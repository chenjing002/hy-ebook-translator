"""Inline placeholder protection.

Even inside a prose segment there are spans a translator must not touch: inline
code, inline LaTeX, link and image destinations, footnote references, raw HTML
tags and bare URLs. Each is swapped for a numbered sentinel before the segment
reaches the model, then restored afterwards.

The sentinels use CJK-adjacent brackets (``⟪0⟫``) because translation APIs tend
to copy those through reliably, and restoration is tolerant when an API sometimes
substitutes a visually similar bracket or inserts spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OPEN, CLOSE = "⟪", "⟫"
# Strict form: our own sentinel plus the bracket pairs the model occasionally
# substitutes for it. Narrow on purpose — a loose pattern would rewrite ordinary
# prose like "(2)" into a protected span.
RESTORE_RE = re.compile(r"[⟪⟦《«〈]\s*(\d+)\s*[⟫⟧》»〉⟜⟩⟪]")
# Last-resort form, applied only when a specific index went missing.
LOOSE_TMPL = r"[\[(（<]\s*{idx}\s*[\])）>]"
FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
LETTER_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")

# Order matters: code spans first so that `$x$` inside backticks is not read as
# math, then math, then link/image destinations, then the narrower patterns.
PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("code", re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL), 0),
    ("math", re.compile(r"\$\$.+?\$\$", re.DOTALL), 0),
    ("math", re.compile(r"\\\(.+?\\\)", re.DOTALL), 0),
    # $...$ must not swallow prices like "$5 and $6".
    ("math", re.compile(r"(?<![\w$])\$(?!\s)(?:\\.|[^$\\\n])+?(?<!\s)\$(?![\w$])"), 0),
    ("dest", re.compile(r"(?<=\]) {0,1}\((?:[^()\s]|\([^()]*\))*(?:\s+\"[^\"]*\")?\)"), 0),
    ("ref", re.compile(r"(?<=\])\[[^\]\n]*\]"), 0),
    ("fnref", re.compile(r"\[\^[^\]\n]+\]"), 0),
    ("autolink", re.compile(r"<(?:[A-Za-z][A-Za-z0-9+.\-]*:[^<>\s]+|[^<>@\s]+@[^<>@\s]+)>"), 0),
    ("html", re.compile(r"</?[A-Za-z][^<>\n]*>"), 0),
    ("latex", re.compile(r"\\(?:[A-Za-z]+(?:\{[^{}\n]*\})*|[\\{}$&#_%~^])"), 0),
    ("url", re.compile(r"(?<![(\w])(?:https?://|www\.|mailto:)[^\s<>()\[\]\"']+"), 0),
    ("entity", re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,30}|#\d{1,6}|#[xX][0-9A-Fa-f]{1,6});"), 0),
]


@dataclass
class Masked:
    text: str
    spans: list[str]

    @property
    def needs_translation(self) -> bool:
        """False when the segment is nothing but protected spans and symbols."""
        stripped = RESTORE_RE.sub("", self.text)
        return bool(LETTER_RE.search(stripped) or re.search(r"[\u4e00-\u9fff]", stripped))

    def restore(self, translated: str) -> tuple[str, list[int]]:
        """Put the protected spans back. Returns the text and any lost indices."""
        used: set[int] = set()
        text = translated.translate(FULLWIDTH)

        def sub(m: re.Match[str]) -> str:
            idx = int(m.group(1))
            if idx >= len(self.spans):
                return m.group(0)
            used.add(idx)
            return self.spans[idx]

        text = RESTORE_RE.sub(sub, text)
        # Only now, and only for the indices that vanished, accept a degraded
        # bracket such as "[3]" — scoped this tightly it cannot damage prose.
        for idx in range(len(self.spans)):
            if idx in used:
                continue
            loose = re.compile(LOOSE_TMPL.format(idx=idx))
            m = loose.search(text)
            if m:
                text = text[:m.start()] + self.spans[idx] + text[m.end():]
                used.add(idx)
        missing = [i for i in range(len(self.spans)) if i not in used]
        return text, missing


def mask(text: str) -> Masked:
    spans: list[str] = []
    placeholders: list[str] = []

    def take(m: re.Match[str]) -> str:
        spans.append(m.group(0))
        token = f"{OPEN}{len(spans) - 1}{CLOSE}"
        placeholders.append(token)
        return token

    out = text
    for _name, pattern, _flags in PATTERNS:
        # Never re-scan inside an already-emitted placeholder.
        out = _sub_outside_placeholders(pattern, take, out)
    return Masked(out, spans)


def _sub_outside_placeholders(pattern: re.Pattern[str], repl, text: str) -> str:
    """Apply `pattern` only to the regions of `text` that are not sentinels."""
    guard = re.compile(re.escape(OPEN) + r"\d+" + re.escape(CLOSE))
    out: list[str] = []
    pos = 0
    for g in guard.finditer(text):
        out.append(pattern.sub(repl, text[pos:g.start()]))
        out.append(g.group(0))
        pos = g.end()
    out.append(pattern.sub(repl, text[pos:]))
    return "".join(out)


def strip_placeholders(text: str) -> str:
    return RESTORE_RE.sub("", text)


def has_orphan_placeholder(text: str) -> bool:
    return any(ch in text for ch in "⟪⟫⟦⟧")
