"""Conservative, whitespace-only repair for converted Markdown prose.

The parser supplies protected block boundaries. Within a paragraph, rules
classify existing line breaks as paragraph breaks, soft wraps, or uncertain.
Only uncertain boundaries are offered to an optional AI classifier.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Callable

from .mdparse import parse_markdown
from .protect import PATTERNS

END_RE = re.compile(r"[.!?。！？][\"'”’）)]*$")
IMAGE_RE = re.compile(r"^\s*!\[[^\]]*\]\(")
WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]$")
START_RE = re.compile(r"^[\"'“‘(\[]?\**[A-ZÀ-Þ0-9]")
INDEX_ENTRY_RE = re.compile(r"^.{2,100},\s*(?:\d+[A-Za-z]?|[ivxlcdm]+)(?:[–-]\d+)?(?:,.*)?$", re.I)
GLOSSARY_ENTRY_RE = re.compile(r"^\*\*[^*\n]{2,100}:\*\*")
SECTION_RE = re.compile(r"^(?:index|table of contents|contents|glossary|[A-Z])$", re.I)


@dataclass(frozen=True)
class Boundary:
    id: int
    before: str
    after: str


@dataclass
class Result:
    text: str
    split: int = 0
    joined: int = 0
    uncertain: int = 0
    ai_decisions: int = 0


def _visible(line: str, limit: int = 120, *, tail: bool = False) -> str:
    line = line.strip()
    return line[-limit:] if tail else line[:limit]


def _classify(left: str, right: str) -> str:
    a, b = left.strip(), right.strip()
    if IMAGE_RE.match(a) or IMAGE_RE.match(b):
        return "split"
    if left.endswith(("  ", "\\")) or right.startswith((" ", "\t")):
        return "keep"  # Markdown hard break or meaningful indentation
    if not a or not b:
        return "keep"
    if a.endswith(("—", "–")) and b[:1].islower():
        return "join"
    if GLOSSARY_ENTRY_RE.match(b) or b.startswith(("http://", "https://")):
        return "split"
    if INDEX_ENTRY_RE.match(a) and INDEX_ENTRY_RE.match(b):
        return "split"
    if a.endswith(":") and START_RE.match(b) and (len(a) >= 60 or len(b) >= 60):
        return "split"
    if len(a) <= 80 and len(b) >= 80 and (a.isupper() or a.isdigit()):
        return "split"
    if END_RE.search(a):
        if START_RE.match(b):
            if (len(a) >= 80 and len(b) >= 40) or (len(a) >= 40 and len(b) >= 80):
                return "split"
            return "uncertain"
        return "uncertain"
    if a.endswith((":", ";")):
        return "uncertain"
    if a.endswith("-") and b[:1].islower():
        return "keep"  # preserve a possible hyphenated word
    if WORD_RE.search(a) and b[:1].islower():
        return "join"
    return "uncertain"


def _same_characters(source: str, result: str) -> bool:
    """Only whitespace may change; every other code point keeps its order."""
    return re.sub(r"\s", "", source) == re.sub(r"\s", "", result)


def _protected_breaks(raw: str) -> set[int]:
    """Newline ordinals inside code, math, links, or multiline HTML tags."""
    positions = [match.start() for match in re.finditer("\n", raw)]
    protected: set[int] = set()
    patterns = [pattern for _, pattern, _ in PATTERNS[:3]]
    patterns.extend([
        re.compile(r"\]\([^)]*\)", re.DOTALL),
        re.compile(r"<[A-Za-z/][^>]*>", re.DOTALL),
    ])
    for pattern in patterns:
        for match in pattern.finditer(raw):
            first = bisect_right(positions, match.start())
            end = bisect_left(positions, match.end())
            protected.update(range(first, end))
    return protected


def _prepare_once(
    source: str,
    decide: Callable[[list[Boundary]], dict[int, str]] | None = None,
) -> Result:
    if "\r" in source:
        raise ValueError("CR line endings are unsupported; convert a copy to LF first")
    blocks = parse_markdown(source)
    cursor = 0
    pieces: list[str] = []
    entries: list[tuple[str, list[str], list[str]]] = []
    pending: list[Boundary] = []
    next_id = 1

    for index, block in enumerate(blocks):
        at = source.find(block.raw, cursor)
        if at < 0:
            raise ValueError(f"could not locate {block.kind} block at offset {cursor}")
        pieces.append(source[cursor:at])
        if block.kind != "paragraph" or "\n" not in block.raw:
            entries.append((block.raw, [], []))
        else:
            lines = block.raw.split("\n")
            actions: list[str] = []
            preceding = blocks[index - 1] if index else None
            is_index = bool(preceding and preceding.kind == "heading"
                            and SECTION_RE.fullmatch(preceding.meta.get("text", "").strip()))
            protected = _protected_breaks(block.raw)
            for line_index, (left, right) in enumerate(zip(lines, lines[1:])):
                action = ("keep" if line_index in protected else
                          "split" if is_index else _classify(left, right))
                if action == "uncertain":
                    pending.append(Boundary(next_id, _visible(left, tail=True), _visible(right)))
                    action = f"?{next_id}"
                    next_id += 1
                actions.append(action)
            entries.append((block.raw, lines, actions))
        cursor = at + len(block.raw)
    pieces.append(source[cursor:])

    decisions = decide(pending) if pending and decide else {}
    if not isinstance(decisions, dict):
        raise ValueError("AI classifier must return a mapping of boundary IDs to actions")
    valid = {b.id for b in pending}
    if set(decisions) - valid or any(x not in {"split", "join", "keep"} for x in decisions.values()):
        raise ValueError("AI classifier returned an unknown boundary or action")

    result = Result(text="", uncertain=len(pending), ai_decisions=len(decisions))
    out: list[str] = []
    for idx, (raw, lines, actions) in enumerate(entries):
        out.append(pieces[idx])
        if not actions:
            out.append(raw)
            continue
        repaired = [lines[0]]
        for action, line in zip(actions, lines[1:]):
            if action.startswith("?"):
                action = decisions.get(int(action[1:]), "keep")
            if action == "split":
                repaired.append("\n\n")
                result.split += 1
            elif action == "join":
                repaired.append(" ")
                result.joined += 1
            else:
                repaired.append("\n")
            repaired.append(line)
        out.append("".join(repaired))
    out.append(pieces[-1])
    result.text = "".join(out)
    if not _same_characters(source, result.text):
        raise ValueError("preprocessing changed source characters")
    return result


def prepare(
    source: str,
    decide: Callable[[list[Boundary]], dict[int, str]] | None = None,
) -> Result:
    """Run rules to a fixed point, then optionally ask AI about leftovers."""
    text = source
    total = Result(text=source)

    def settle(current: str) -> str:
        for _ in range(12):
            part = _prepare_once(current)
            total.split += part.split
            total.joined += part.joined
            total.uncertain = part.uncertain
            if part.text == current:
                return current
            current = part.text
        raise ValueError("paragraph rules did not converge")

    text = settle(text)
    if decide:
        part = _prepare_once(text, decide)
        total.split += part.split
        total.joined += part.joined
        total.ai_decisions = part.ai_decisions
        text = settle(part.text)
    total.text = text
    if not _same_characters(source, text):
        raise ValueError("preprocessing changed source characters")
    return total


def openai_compatible_classifier(url: str, model: str, *, batch_size: int = 16,
                                 max_boundaries: int = 128, timeout: int = 60,
                                 ) -> Callable[[list[Boundary]], dict[int, str]]:
    """Opt-in classifier for a local OpenAI-compatible chat endpoint.

    Only short windows around uncertain boundaries are sent. The model may
    choose boundary actions; its generated prose is never written to a file.
    """
    parsed_url = urllib.parse.urlsplit(url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("AI endpoint must be local (localhost/127.0.0.1/::1)")
    if batch_size < 1 or max_boundaries < 0:
        raise ValueError("batch_size must be positive and max_boundaries nonnegative")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("AI endpoint redirected; refusing to send text elsewhere")

    opener = urllib.request.build_opener(NoRedirect)

    def classify(boundaries: list[Boundary]) -> dict[int, str]:
        result: dict[int, str] = {}
        # Favor probable prose boundaries. Short title/index fragments can be
        # numerous and are left untouched when the token budget is exhausted.
        ranked = sorted(boundaries, key=lambda b: (
            bool(END_RE.search(b.before)),
            min(len(b.before), 80) + min(len(b.after), 80),
        ), reverse=True)[:max_boundaries]
        selected = sorted(ranked, key=lambda b: b.id)
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            data = [{"id": b.id, "before": b.before, "after": b.after} for b in batch]
            prompt = (
                "Decide whether each existing newline separates two natural paragraphs "
                "(split), is a PDF soft wrap within one paragraph (join), or cannot "
                "be determined (keep). Return ONLY a JSON object mapping each numeric "
                "id to split, join, or keep. Do not reproduce or rewrite the text.\n"
                + json.dumps(data, ensure_ascii=False)
            )
            payload = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max(128, batch_size * 16),
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=timeout) as response:
                content = json.load(response)["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            parsed = json.loads(content)
            allowed = {b.id for b in batch}
            for key, action in parsed.items():
                boundary_id = int(key)
                if boundary_id not in allowed or action not in {"split", "join", "keep"}:
                    raise ValueError("AI returned an invalid boundary decision")
                result[boundary_id] = action
        return result

    return classify
