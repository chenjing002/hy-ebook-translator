"""Structural validation of a bilingual document.

The guarantee this pipeline makes is that the original survives intact and that
every piece of prose actually got translated. Both are checked here by
re-reading the generated file and comparing it against the source, rather than
by trusting the translator's own bookkeeping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .mdparse import PASSTHROUGH_KINDS, parse_markdown
from .protect import OPEN

CJK_RE = re.compile(r"[㐀-䶿一-鿿]")
URL_RE = re.compile(r"\]\(\s*(<?[^)\s]+>?)")
HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+\S", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"[⟪⟦]\s*\d{1,3}\s*[⟫⟧]|^[ \t]*【\s*\d{1,3}\s*】", re.MULTILINE)
ERROR, WARN, INFO = "error", "warn", "info"


@dataclass
class Issue:
    level: str
    code: str
    detail: str


@dataclass
class Report:
    name: str
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level: str, code: str, detail: str) -> None:
        self.issues.append(Issue(level, code, detail))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(source: Path, output: Path, cfg: Config,
             translator_warnings: list[dict] | None = None) -> Report:
    report = Report(name=source.name)
    if not output.exists():
        report.add(ERROR, "missing-output", f"{output} was not written")
        return report

    src_text = source.read_text(encoding="utf-8")
    out_text = output.read_text(encoding="utf-8")
    src_blocks = parse_markdown(src_text)

    # 1. Every verbatim block must survive byte-for-byte.
    for block in src_blocks:
        if block.kind not in PASSTHROUGH_KINDS or block.kind == "front_matter":
            continue
        if block.raw.strip() and block.raw.strip() not in out_text:
            preview = block.raw.strip().splitlines()[0][:60]
            report.add(ERROR, f"lost-{block.kind}", f"{block.kind} block missing: {preview!r}")

    # 2. Every prose block must still carry its original text.
    missing_source = 0
    for block in src_blocks:
        if block.kind in PASSTHROUGH_KINDS:
            continue
        probe = block.meta.get("text") if block.kind in {"heading", "footnote"} else block.raw
        probe = (probe or "").strip()
        if not probe:
            continue
        if probe not in out_text and _collapse(probe) not in _collapse(out_text):
            missing_source += 1
            if missing_source <= 5:
                report.add(ERROR, "lost-original",
                           f"original {block.kind} not found in output: {probe[:70]!r}")
    if missing_source > 5:
        report.add(ERROR, "lost-original", f"... and {missing_source - 5} more")

    # 3. Link and image destinations must all still be there.
    src_urls = {m.group(1) for m in URL_RE.finditer(src_text)}
    out_urls = {m.group(1) for m in URL_RE.finditer(out_text)}
    lost = sorted(src_urls - out_urls)
    for url in lost[:5]:
        report.add(ERROR, "lost-link", f"destination dropped: {url}")
    if len(lost) > 5:
        report.add(ERROR, "lost-link", f"... and {len(lost) - 5} more destinations")

    # 4. Protection sentinels must never reach the published file.
    leaked = PLACEHOLDER_RE.findall(out_text)
    if leaked:
        report.add(ERROR, "leaked-placeholder",
                   f"{len(leaked)} internal marker(s) left in output, e.g. {leaked[0]!r}")

    # 5. Headings keep the document navigable.
    src_h, out_h = len(HEADING_RE.findall(src_text)), len(HEADING_RE.findall(out_text))
    if out_h < src_h:
        report.add(ERROR, "heading-loss", f"{src_h} headings in source, {out_h} in output")

    # HTML tables must have a translated copy before their marked original.
    # A document can contain plenty of Chinese prose while leaving every table
    # untouched, so the document-wide CJK check below cannot catch this.
    for block in src_blocks:
        if block.kind != "html_table" or not any(u.translatable() for u in block.units):
            continue
        marker = f"<!--en-->\n{block.raw.strip()}"
        at = out_text.find(marker)
        target = out_text[:at].rstrip().rsplit("\n\n", 1)[-1] if at >= 0 else ""
        if at < 0 or "<table" not in target.lower() or not CJK_RE.search(target):
            report.add(ERROR, "untranslated-html-table",
                       f"HTML table has no Chinese counterpart: {block.raw[:70]!r}")

    # 6. The translation itself has to be present.
    prose_blocks = [b for b in src_blocks if b.kind not in PASSTHROUGH_KINDS
                    and any(u.translatable() for u in b.iter_units())]
    cjk_chars = len(CJK_RE.findall(out_text))
    report.stats = {
        "source_bytes": len(src_text.encode()),
        "output_bytes": len(out_text.encode()),
        "blocks": len(src_blocks),
        "prose_blocks": len(prose_blocks),
        "cjk_chars": cjk_chars,
        "headings": out_h,
    }
    if prose_blocks and cjk_chars == 0:
        report.add(ERROR, "no-translation", "output contains no Chinese characters")
    elif prose_blocks and cjk_chars < len(prose_blocks) * 4:
        report.add(WARN, "sparse-translation",
                   f"only {cjk_chars} Chinese characters for {len(prose_blocks)} prose blocks")

    # 7. Inline emphasis and code spans should stay balanced on both sides.
    for name, token in (("code span", "`"), ("bold", "**")):
        if out_text.count(token) % 2 != src_text.count(token) % 2:
            report.add(WARN, "unbalanced-markup", f"odd number of {name} markers in output")

    # 8. Anything the translator itself had to work around.
    for w in translator_warnings or []:
        report.add(WARN, "translator", f"{w.get('problem')}: {str(w.get('source'))[:60]!r}")

    return report


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_report(report: Report, verbose: bool = False) -> str:
    icon = "✓" if report.ok else "✗"
    head = (f"{icon} {report.name}  "
            f"{report.stats.get('blocks', 0)} blocks, "
            f"{report.stats.get('cjk_chars', 0)} Chinese chars")
    lines = [head]
    shown = report.issues if verbose else (report.errors + report.warnings[:6])
    for issue in shown:
        tag = {"error": "ERROR", "warn": "warn ", "info": "info "}[issue.level]
        lines.append(f"    {tag} [{issue.code}] {issue.detail}")
    hidden = len(report.warnings) - 6
    if not verbose and hidden > 0:
        lines.append(f"    warn  ... and {hidden} more warnings (use --verbose)")
    return "\n".join(lines)
