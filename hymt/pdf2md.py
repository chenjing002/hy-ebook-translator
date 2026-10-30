"""PDF → Markdown conversion.

A PDF has no paragraphs, headings or tables, only glyphs at positions, so the
conversion reconstructs them:

* text lines are grouped into paragraphs, soft line breaks are removed and
  end-of-line hyphenation is undone;
* a line that recurs at the top or bottom of most pages is a running header or
  footer and is dropped, as is a bare page number;
* a paragraph that stops without punctuation and continues on the next page
  with a lowercase letter is joined back together.

Two back ends read the file. PyMuPDF (`import fitz`), when installed, also
gives font sizes, so headings are recognised by size, bold and italic spans are
marked, tables are found, and images are extracted. Poppler's `pdftotext`, when
on PATH, gives plain text only; headings are then limited to recognisable
"Chapter N" and numbered-section lines. Neither is a hard dependency.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional; the package was renamed from fitz to pymupdf in 2024
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover - depends on the environment
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,7})\s*(?:of\s+\d+)?\s*$", re.I)
DIGITS_RE = re.compile(r"\d+")
CHAPTER_RE = re.compile(
    r"^\s*(?:chapter|part|appendix|section|book)\s+(?:\d{1,3}|[ivxlc]{1,6}|[a-z])\b[\s.:—–-]*(.{0,80})$",
    re.I)
NUMBERED_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^.!?]{2,70})$")
TERMINAL = tuple('.!?:;"”’)]»…')
INLINE_ESCAPE_RE = re.compile(r"([\\`*_\[\]<$])")
LINE_START_ESCAPE_RE = re.compile(r"^(\s*)(#{1,6}\s|>|[-+*]\s|\d{1,9}[.)]\s|[=-]{3,}\s*$)")
WORD_RE = re.compile(r"[A-Za-z]")

MIN_IMAGE_SIDE = 40      # points; anything smaller is a rule or a bullet
HEADER_ZONE = 0.09       # fraction of the page height counted as header/footer


@dataclass
class Para:
    """A block of text as reconstructed from one page."""
    text: str
    kind: str = "p"       # p | h1..h6 | table | image | code
    top: float = 0.0      # position on the page, for the header/footer test
    bottom: float = 0.0


@dataclass
class ConvertResult:
    markdown: str
    title: str
    authors: list[str]
    pages: int
    backend: str
    images: list[Path]
    warnings: list[str] = field(default_factory=list)


# ── shared reconstruction ────────────────────────────────────────────────────


def join_lines(lines: list[str]) -> str:
    """One paragraph from its printed lines: soft wraps become spaces, a word
    broken at the margin is mended."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        if out.endswith(("-", "‐", "\u00ad")) and len(out) > 1 and out[-2].isalpha() \
                and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return re.sub(r"[ \t]{2,}", " ", out)


def _signature(text: str) -> str:
    return DIGITS_RE.sub("#", " ".join(text.lower().split()))


def running_lines(pages: list[list[Para]], page_heights: list[float]) -> set[str]:
    """Signatures of lines that recur at the top or bottom of pages."""
    if len(pages) < 4:
        return set()
    seen: Counter[str] = Counter()
    for paras, height in zip(pages, page_heights):
        page_sigs: set[str] = set()
        for para in paras:
            if para.kind != "p":
                continue
            in_zone = (height and (para.bottom <= height * HEADER_ZONE
                                   or para.top >= height * (1 - HEADER_ZONE)))
            first_or_last = para is paras[0] or para is paras[-1]
            if not (in_zone or (first_or_last and len(para.text) < 90)):
                continue
            sig = _signature(para.text)
            if sig and len(sig) < 120:
                page_sigs.add(sig)
        seen.update(page_sigs)
    threshold = max(3, len(pages) * 0.35)
    return {sig for sig, n in seen.items() if n >= threshold and not sig.isspace()}


def strip_furniture(pages: list[list[Para]], page_heights: list[float]) -> list[list[Para]]:
    """Remove running headers, footers and bare page numbers."""
    running = running_lines(pages, page_heights)
    cleaned: list[list[Para]] = []
    for paras, height in zip(pages, page_heights):
        kept: list[Para] = []
        for i, para in enumerate(paras):
            if para.kind == "p":
                edge = i == 0 or i == len(paras) - 1
                in_zone = height and (para.bottom <= height * HEADER_ZONE
                                      or para.top >= height * (1 - HEADER_ZONE))
                if (edge or in_zone) and PAGE_NUMBER_RE.match(para.text):
                    continue
                if _signature(para.text) in running and (edge or in_zone):
                    continue
            kept.append(para)
        cleaned.append(kept)
    return cleaned


def merge_pages(pages: list[list[Para]]) -> list[Para]:
    """One flow of paragraphs; a paragraph cut by a page break is rejoined."""
    out: list[Para] = []
    for paras in pages:
        for i, para in enumerate(paras):
            if i == 0 and out and out[-1].kind == "p" and para.kind == "p" \
                    and _continues(out[-1].text, para.text):
                out[-1].text = join_lines([out[-1].text, para.text])
                continue
            out.append(para)
    return out


def _continues(previous: str, following: str) -> bool:
    previous = previous.rstrip()
    if not previous or not following:
        return False
    if previous.endswith(TERMINAL):
        return False
    head = following.lstrip()[:1]
    if previous.endswith("-") or head.islower():
        return True
    return False


def escape_inline(text: str) -> str:
    return INLINE_ESCAPE_RE.sub(r"\\\1", text)


def to_markdown(paras: list[Para], *, title: str, authors: list[str], source: str,
                add_title: bool, provenance: bool) -> str:
    blocks: list[str] = []
    if provenance:
        blocks.append(f"<!-- Generated by hy-ebook-translator pdf2md from {source} on "
                      f"{time.strftime('%Y-%m-%d')} -->")
    if add_title:
        blocks.append(f"# {escape_inline(' '.join(title.split()))}")
        if authors:
            blocks.append("*" + escape_inline(", ".join(authors)) + "*")
    for para in paras:
        text = para.text.strip()
        if not text:
            continue
        if para.kind.startswith("h"):
            level = int(para.kind[1])
            blocks.append(f"{'#' * level} {text}")
        elif para.kind in {"table", "image", "code"}:
            blocks.append(text)
        else:
            blocks.append(LINE_START_ESCAPE_RE.sub(r"\1\\\2", text))
    markdown = "\n\n".join(blocks).rstrip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", markdown)


def detect_heading_plain(text: str) -> str | None:
    """Heading level for a line when no font information is available."""
    if len(text) > 90 or "\n" in text:
        return None
    if CHAPTER_RE.match(text):
        return "h2"
    m = NUMBERED_RE.match(text)
    if m:
        depth = m.group(1).count(".")
        return f"h{min(6, 2 + depth)}"
    return None


# ── PyMuPDF back end ─────────────────────────────────────────────────────────


def _fitz_pages(doc, images_dir: Path | None, images_rel: str,
                warnings: list[str]) -> tuple[list[list[Para]], list[float], list[Path]]:
    """Paragraphs per page, with headings, emphasis, tables and images."""
    size_weight: Counter[float] = Counter()
    raw_pages = []
    for page in doc:
        info = page.get_text("dict", sort=True)
        raw_pages.append((page, info))
        for block in info["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        size_weight[round(span["size"] * 2) / 2] += len(text)
    body_size = size_weight.most_common(1)[0][0] if size_weight else 10.0

    # Distinct larger sizes, biggest first, become heading levels 2, 3, 4 (1 is the title).
    heading_sizes = sorted({s for s in size_weight if s >= body_size * 1.15}, reverse=True)
    level_of = {s: f"h{min(6, 2 + i)}" for i, s in enumerate(heading_sizes[:4])}

    pages: list[list[Para]] = []
    heights: list[float] = []
    files: list[Path] = []
    image_count = 0
    for page, info in raw_pages:
        paras: list[Para] = []
        tables: list[tuple[tuple[float, float, float, float], str]] = []
        if hasattr(page, "find_tables"):
            try:
                for table in page.find_tables().tables:
                    md = _fitz_table(table)
                    if md:
                        tables.append((tuple(table.bbox), md))
            except Exception as exc:  # pragma: no cover - library-version dependent
                warnings.append(f"page {page.number + 1}: table detection failed: {exc}")
        emitted_tables: set[int] = set()

        for block in info["blocks"]:
            bbox = block["bbox"]
            if block.get("type") == 1:
                if images_dir is None:
                    continue
                if bbox[2] - bbox[0] < MIN_IMAGE_SIDE or bbox[3] - bbox[1] < MIN_IMAGE_SIDE:
                    continue
                data = block.get("image")
                ext = block.get("ext") or "png"
                if not data:
                    continue
                image_count += 1
                images_dir.mkdir(parents=True, exist_ok=True)
                name = f"page{page.number + 1:03d}-{image_count:03d}.{ext}"
                target = images_dir / name
                target.write_bytes(data)
                files.append(target)
                rel = f"{images_rel}/{name}" if images_rel else name
                paras.append(Para(f"![]({rel})", "image", bbox[1], bbox[3]))
                continue

            inside = next((i for i, (tb, _) in enumerate(tables) if _within(bbox, tb)), None)
            if inside is not None:
                if inside not in emitted_tables:
                    emitted_tables.add(inside)
                    tb, md = tables[inside]
                    paras.append(Para(md, "table", tb[1], tb[3]))
                continue

            lines_text: list[str] = []
            sizes: list[float] = []
            for line in block["lines"]:
                text = _fitz_line(line["spans"])
                if text.strip():
                    lines_text.append(text)
                    sizes.extend(round(s["size"] * 2) / 2 for s in line["spans"] if s["text"].strip())
            if not lines_text:
                continue
            text = join_lines(lines_text)
            kind = "p"
            if sizes:
                dominant = Counter(sizes).most_common(1)[0][0]
                if dominant in level_of and len(text) <= 120 and len(lines_text) <= 3:
                    kind = level_of[dominant]
                    text = _strip_emphasis(text)
            paras.append(Para(text, kind, bbox[1], bbox[3]))

        for i, (tb, md) in enumerate(tables):
            if i not in emitted_tables:
                paras.append(Para(md, "table", tb[1], tb[3]))
        paras.sort(key=lambda p: (p.top, p.bottom))
        pages.append(paras)
        heights.append(float(page.rect.height))
    return pages, heights, files


def _fitz_line(spans: list[dict]) -> str:
    """One printed line with bold and italic marked, superscripts tagged."""
    parts: list[str] = []
    for span in spans:
        text = escape_inline(span["text"])
        if not text.strip():
            parts.append(text)
            continue
        flags = span.get("flags", 0)
        bold = bool(flags & 16) or "bold" in span.get("font", "").lower()
        italic = bool(flags & 2) or "italic" in span.get("font", "").lower()
        superscript = bool(flags & 1)
        lead, core, trail = re.match(r"^(\s*)(.*?)(\s*)$", text, re.DOTALL).groups()
        if superscript and core:
            core = f"<sup>{core}</sup>"
        else:
            if italic and core:
                core = f"{core}"
            if bold and core:
                core = f"{core}"
        parts.append(lead + core + trail)
    text = "".join(parts)
    # Runs that touch become one run; the stand-ins then become markers.
    text = text.replace("", "").replace("", "")
    text = re.sub(r"(\s+)", r"\1", text)
    text = re.sub(r"(\s+)", r"\1", text)
    return text.replace("", "**").replace("", "*")


def _strip_emphasis(text: str) -> str:
    return re.sub(r"(?<!\\)\*{1,2}", "", text)


def _within(inner, outer, slack: float = 2.0) -> bool:
    return (inner[0] >= outer[0] - slack and inner[1] >= outer[1] - slack
            and inner[2] <= outer[2] + slack and inner[3] <= outer[3] + slack)


def _fitz_table(table) -> str:
    rows = table.extract()
    rows = [[(" ".join(str(c or "").split())).replace("|", "\\|") for c in row] for row in rows]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(escape_inline(c) for c in rows[0]) + " |",
             "|" + "|".join(" --- " for _ in range(width)) + "|"]
    lines.extend("| " + " | ".join(escape_inline(c) for c in r) + " |" for r in rows[1:])
    return "\n".join(lines)


# ── pdftotext back end ───────────────────────────────────────────────────────


def pdftotext_pages(text: str) -> list[list[Para]]:
    """Paragraphs per page from `pdftotext` output (pages end in form feeds)."""
    pages: list[list[Para]] = []
    for chunk in text.split("\f"):
        paras: list[Para] = []
        for raw in re.split(r"\n\s*\n", chunk):
            lines = [ln for ln in raw.split("\n") if ln.strip()]
            if not lines:
                continue
            joined = escape_inline(join_lines(lines))
            kind = detect_heading_plain(joined) or "p"
            paras.append(Para(joined, kind))
        if paras:
            pages.append(paras)
    return pages


def _run_pdftotext(pdf: Path) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext not found")
    out = subprocess.run([exe, "-enc", "UTF-8", "-eol", "unix", str(pdf), "-"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {out.stderr.strip()[:200]}")
    return out.stdout


# ── entry point ──────────────────────────────────────────────────────────────


def available_backends() -> list[str]:
    found = []
    if fitz is not None:
        found.append("pymupdf")
    if shutil.which("pdftotext"):
        found.append("pdftotext")
    return found


def convert_pdf(pdf: Path, *, images_dir: Path | None, images_rel: str = "",
                title: str | None = None, add_title: bool = True, backend: str = "auto",
                provenance: bool = True) -> ConvertResult:
    backends = available_backends()
    if backend == "auto":
        if not backends:
            raise RuntimeError(
                "no PDF reader available: install PyMuPDF (`pip install pymupdf`) "
                "or poppler (`brew install poppler` for pdftotext)")
        backend = backends[0]
    elif backend not in backends:
        raise RuntimeError(f"back end {backend!r} is not available (found: {backends or 'none'})")

    warnings: list[str] = []
    authors: list[str] = []
    meta_title = ""
    files: list[Path] = []
    if backend == "pymupdf":
        doc = fitz.open(str(pdf))
        meta = doc.metadata or {}
        meta_title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        authors = [a.strip() for a in re.split(r"[;,]| and ", author) if a.strip()] if author else []
        pages, heights, files = _fitz_pages(doc, images_dir, images_rel.rstrip("/"), warnings)
        page_count = doc.page_count
        doc.close()
    else:
        pages = pdftotext_pages(_run_pdftotext(pdf))
        heights = [0.0] * len(pages)
        page_count = len(pages)
        if images_dir is not None:
            warnings.append("pdftotext extracts no images; install PyMuPDF for those")

    pages = strip_furniture(pages, heights)
    paras = merge_pages(pages)
    book_title = title or _usable_title(meta_title) or pdf.stem.replace("_", " ")
    markdown = to_markdown(paras, title=book_title, authors=authors, source=pdf.name,
                           add_title=add_title, provenance=provenance)
    return ConvertResult(markdown, book_title, authors, page_count, backend, files, warnings)


def _usable_title(title: str) -> str:
    """PDF metadata often holds a file name or an application's default."""
    if not title or re.search(r"\.(pdf|docx?|indd|tex|qxd)$", title, re.I):
        return ""
    if re.fullmatch(r"(untitled|microsoft word.*|document\d*|print|slide ?\d*)", title, re.I):
        return ""
    return title if WORD_RE.search(title) or len(title) > 3 else ""
