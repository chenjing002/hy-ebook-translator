"""Document translation orchestration.

Segments are packed into batches under a token budget and sent to Hy-MT2 in one
request, each prefixed with a 【n】 marker. Batching is what makes a book
tractable, but it can fail — a model may merge or drop a marker — so every
response is verified and there is a strict fallback ladder:

    batch  ->  retry batch  ->  per-segment  ->  per-segment unmasked  ->  keep source

Nothing is ever silently dropped: anything that falls to the last rung is
recorded as a warning and surfaced by `hy validate`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .engine import Engine
from .mdparse import Block, Unit, document_title, parse_markdown
from .progress import Progress
from .protect import RESTORE_RE, Masked, mask
from .state import DocState, file_sha, segment_key, slugify

SEG_OPEN, SEG_CLOSE = "【", "】"
# Markers are emitted at the start of a line, so anchoring here keeps ordinary
# parenthesised prose from being mistaken for a marker.
SEG_RE = re.compile(r"(?:^|\n)[ \t]*[【\[〔(（]\s*(\d{1,3})\s*[】\]〕)）][ \t:：]*")
SENTENCE_RE = re.compile(
    r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bProf\.)(?<!\bSt\.)"
    r"(?<!\bFig\.)(?<!\bEq\.)(?<!\bvs\.)(?<!\betc\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bcf\.)"
    r"(?<!\bal\.)(?<!\bNo\.)(?<!\bVol\.)(?<!\bpp\.)\s+(?=[\"'“‘(\[A-Z0-9])"
)
CJK_RE = re.compile(r"[㐀-䶿一-鿿]")


@dataclass
class Job:
    """One translatable segment in flight."""

    unit: Unit
    masked: Masked
    tokens: int = 0
    zh: str = ""
    warning: str = ""


@dataclass
class DocResult:
    source: Path
    output: Path
    title: str
    blocks: list[Block]
    zh: dict[str, str]
    warnings: list[dict] = field(default_factory=list)
    cached: int = 0
    translated: int = 0
    skipped: int = 0
    calls: int = 0
    seconds: float = 0.0


class Translator:
    def __init__(self, cfg: Config, engine: Engine, quiet: bool = False) -> None:
        self.cfg = cfg
        self.engine = engine
        self.quiet = quiet

    # -- public API ----------------------------------------------------------
    def translate_document(self, src: Path) -> DocResult:
        started = time.monotonic()
        text = src.read_text(encoding="utf-8")
        blocks = parse_markdown(text)
        units = [u for b in blocks for u in b.iter_units()]
        slug = slugify(src.stem)
        state = DocState.load(self.cfg.state_dir, slug, file_sha(src), self.cfg.fingerprint())

        zh: dict[str, str] = {}
        pending: list[Job] = []
        cached = skipped = 0
        live_keys: set[str] = set()

        for unit in units:
            if not unit.translatable():
                zh[unit.uid] = unit.text
                skipped += 1
                continue
            masked = mask(unit.text)
            if not masked.needs_translation:
                zh[unit.uid] = unit.text
                skipped += 1
                continue
            live_keys.add(segment_key(unit.kind, unit.text))
            hit = state.get(unit.kind, unit.text)
            if hit is not None:
                zh[unit.uid] = hit
                cached += 1
                continue
            pending.append(Job(unit=unit, masked=masked))

        warnings: list[dict] = []
        translated = 0
        if pending:
            self.engine.load(verbose=not self.quiet)
            for job in pending:
                job.tokens = self.engine.count_tokens(job.masked.text) + 4
            batches = self._pack(pending)
            bar = Progress(len(pending), f"  {src.name[:28]:<28}", enabled=not self.quiet)
            for batch in batches:
                self._run_batch(batch, zh)
                for job in batch:
                    zh[job.unit.uid] = job.zh
                    state.put(job.unit.kind, job.unit.text, job.zh)
                    if job.warning:
                        warnings.append({
                            "uid": job.unit.uid, "kind": job.unit.kind,
                            "problem": job.warning, "source": job.unit.text[:160],
                        })
                    translated += 1
                bar.advance(len(batch))
                state.warnings = warnings
                state.save()  # checkpoint: Ctrl-C here loses nothing
            bar.close()

        state.prune(live_keys)
        state.source_sha = file_sha(src)
        state.warnings = warnings
        state.stats = {
            "units": len(units), "cached": cached, "translated": translated,
            "skipped": skipped, "calls": self.engine.calls,
        }
        state.save()

        return DocResult(
            source=src,
            output=self.cfg.outputs_dir / src.name,
            title=document_title(blocks, src.stem),
            blocks=blocks,
            zh=zh,
            warnings=warnings,
            cached=cached,
            translated=translated,
            skipped=skipped,
            calls=self.engine.calls,
            seconds=time.monotonic() - started,
        )

    # -- batching ------------------------------------------------------------
    def _pack(self, jobs: list[Job]) -> list[list[Job]]:
        """Group consecutive segments into requests under the token budget."""
        tc = self.cfg.translate
        batches: list[list[Job]] = []
        current: list[Job] = []
        total = 0
        for job in jobs:
            if job.tokens > tc.max_unit_tokens:
                if current:
                    batches.append(current)
                    current, total = [], 0
                batches.append([job])  # oversized: split by sentence on its own
                continue
            if current and (total + job.tokens > tc.batch_token_budget
                            or len(current) >= tc.max_batch_units):
                batches.append(current)
                current, total = [], 0
            current.append(job)
            total += job.tokens
        if current:
            batches.append(current)
        return batches

    # -- execution -----------------------------------------------------------
    def _run_batch(self, batch: list[Job], zh: dict[str, str]) -> None:
        if len(batch) == 1:
            job = batch[0]
            if job.tokens > self.cfg.translate.max_unit_tokens:
                self._translate_long(job, zh)
            else:
                self._translate_one(job, zh)
            return

        body = "\n".join(
            f"{SEG_OPEN}{i}{SEG_CLOSE} {job.masked.text}" for i, job in enumerate(batch)
        )
        budget = sum(j.tokens for j in batch)
        for _attempt in range(self.cfg.translate.max_retries + 1):
            prompt = self.engine.build_prompt(
                body, delimiters=True, background=self._background(zh)
            )
            raw = self.engine.generate(prompt, max_new_tokens=int(budget * 2.4) + 128)
            parts = _split_segments(raw, len(batch))
            if parts is None:
                continue
            staged: list[str] = []
            for job, part in zip(batch, parts):
                restored, missing = job.masked.restore(part)
                if missing or not part.strip():
                    staged = []
                    break
                staged.append(restored)
            if staged:
                for job, value in zip(batch, staged):
                    job.zh = value
                return
        # Markers proved unreliable for this chunk: one request per segment.
        for job in batch:
            self._translate_one(job, zh)

    def _translate_one(self, job: Job, zh: dict[str, str]) -> None:
        has_placeholders = bool(job.masked.spans)
        for _attempt in range(self.cfg.translate.max_retries + 1):
            prompt = self.engine.build_prompt(
                job.masked.text, delimiters=has_placeholders, background=self._background(zh)
            )
            raw = _strip_stray_markers(
                self.engine.generate(prompt, max_new_tokens=int(job.tokens * 2.4) + 128)
            )
            restored, missing = job.masked.restore(raw)
            if raw.strip() and not missing:
                job.zh = restored
                return
        # Last resort: let the model see the real code and links, not sentinels.
        prompt = self.engine.build_prompt(job.unit.text, delimiters=False)
        raw = _strip_stray_markers(
            self.engine.generate(prompt, max_new_tokens=int(job.tokens * 2.4) + 128)
        )
        if raw.strip():
            job.zh = raw
            job.warning = "translated without placeholder protection"
            return
        job.zh = job.unit.text
        job.warning = "model returned nothing; original text kept"

    def _translate_long(self, job: Job, zh: dict[str, str]) -> None:
        """Split an over-long segment on sentence boundaries, then rejoin."""
        pieces = _split_sentences(job.masked.text)
        if len(pieces) == 1:
            pieces = _hard_split(job.masked.text, self.cfg.translate.max_unit_tokens * 3)
        out: list[str] = []
        for piece in pieces:
            sub = Job(
                unit=Unit(job.unit.uid, job.unit.kind, piece),
                masked=_scoped_mask(piece, job.masked.spans),
                tokens=self.engine.count_tokens(piece) + 4,
            )
            self._translate_one(sub, zh)
            out.append(sub.zh)
            if sub.warning:
                job.warning = sub.warning
        job.zh = _join_zh(out)

    def _background(self, zh: dict[str, str]) -> str:
        """Recent Chinese output, handed back as terminology context."""
        n = self.cfg.translate.context_units
        if n <= 0:
            return ""
        done = [v for v in zh.values() if v and CJK_RE.search(v)]
        if not done:
            return ""
        return " ".join(done[-n:]).strip()[-400:]


# --- helpers ----------------------------------------------------------------


def _scoped_mask(piece: str, spans: list[str]) -> Masked:
    """A Masked view over one sentence of a longer segment.

    The span table is shared with the parent so indices keep their meaning, but
    only the placeholders that actually occur in this piece are required back.
    """
    present = {int(m.group(1)) for m in RESTORE_RE.finditer(piece)}

    class _Scoped(Masked):
        def restore(self, translated: str) -> tuple[str, list[int]]:
            text, missing = Masked.restore(self, translated)
            return text, [i for i in missing if i in present]

    return _Scoped(piece, list(spans))


def _split_segments(raw: str, expected: int) -> list[str] | None:
    """Split a batched response on its 【n】 markers, or None if unreliable."""
    matches = list(SEG_RE.finditer(raw))
    if len(matches) < expected:
        return None
    found: dict[int, str] = {}
    for k, m in enumerate(matches):
        idx = int(m.group(1))
        end = matches[k + 1].start() if k + 1 < len(matches) else len(raw)
        body = raw[m.end():end].strip()
        if found.get(idx):
            continue  # keep the first occurrence of a duplicated marker
        found[idx] = body
    if set(found) != set(range(expected)):
        return None
    if any(not found[i] for i in range(expected)):
        return None
    return [found[i] for i in range(expected)]


def _strip_stray_markers(text: str) -> str:
    return SEG_RE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    return parts or [text]


def _hard_split(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for word in text.split(" "):
        if buf and len(buf) + len(word) + 1 > max_chars:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        out.append(buf)
    return out or [text]


def _join_zh(parts: list[str]) -> str:
    """Join translated pieces: no space between CJK, a space across scripts."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not out:
            out = part
        elif CJK_RE.search(out[-1]) and CJK_RE.search(part[0]):
            out += part
        else:
            out += " " + part
    return out
