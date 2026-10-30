"""Scanned PDF → Markdown with Apple Vision OCR.

A scanned book is a stack of page images, so nothing in the file says where a
paragraph, heading or table is. This module rebuilds them from the geometry of
the recognised text:

* each page is rendered with PyMuPDF and read by macOS Vision (the same
  recogniser Preview and Live Text use), which returns one box per text line
  or table cell;
* running headers, page numbers and side tabs are dropped by position and by
  how often the same text recurs on other pages;
* the PDF outline (bookmarks) names the chapters and sections, matched to the
  recognised lines of the pages it points to; lines that look like Chinese
  chapter, section or "一、" headings are recognised by pattern as well;
* a first-line indent, a short last line or a wide gap starts a paragraph, and
  a paragraph that runs to the foot of a page continues at the top of the
  next; CJK lines are joined without a space;
* three or more consecutive rows of short cells form a table, rebuilt as a
  pipe table from the cell positions;
* a wide blank-looking gap that nevertheless holds ink (a photograph, a
  chart) is cropped from the page render and saved as a figure;
* pages of dotted-leader contents entries are skipped, since the Markdown
  carries its own headings.

Per-page OCR output is cached as JSON, so a run can resume and the layout
pass can be repeated without recognising the pages again. Vision needs macOS
and the PyObjC bindings (`pip install pyobjc-framework-Vision
pyobjc-framework-Quartz`), and the renderer needs PyMuPDF (`pip install
pymupdf`).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover - depends on the environment
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

DEFAULT_LANGS = ("zh-Hans", "en-US")
DEFAULT_DPI = 220
HEADER_ZONE = 0.10        # fraction of the page height treated as the header band
FOOTER_ZONE = 0.92        # below this (top-left origin) is the footer band
FIGURE_MIN_GAP = 0.07     # a text-free gap at least this tall may hold a figure
FIGURE_DPI = 120
TABLE_MIN_ROWS = 3

CJK_RE = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")
CN_NUM = "一二三四五六七八九十百零〇两"
CHAPTER_RE = re.compile(rf"^第[{CN_NUM}\d]+[章篇编卷部]\s*\S")
SECTION_RE = re.compile(rf"^第[{CN_NUM}\d]+节\s*\S")
SUB1_RE = re.compile(rf"^[{CN_NUM}]+、\S")
SUB2_RE = re.compile(rf"^[（(][{CN_NUM}]+[)）]\S")
SPECIAL_HEAD_RE = re.compile(r"^(概\s*述|专\s*记|附\s*录|凡\s*例|序|前\s*言|后\s*记|编后记|大事记|索\s*引)$")
PAGE_NUMBER_RE = re.compile(r"^[\s\d\-–—·.]{1,9}$")
CONTINUED_RE = re.compile(r"^[（(〔【\[]\s*续\s*(前|上|表)?\s*[)）〕】\]]\s*")
TOC_LINE_RE = re.compile(r"(?:[.。·•…‥:：_\-]\s?){3,}\s*\d{1,4}\s*$|\s\d{1,4}\s*$")
TERMINAL = "。！？；：”’」』）)!?;…"
LINE_START_ESCAPE_RE = re.compile(r"^(\s*)(#{1,6}\s|>|[-+*]\s|\d{1,9}[.)]\s|[=-]{3,}\s*$)")
NORMALISE_RE = re.compile(r"[\s•·・.,，、:：;；—\-_()（）《》〈〉【】\[\]\"'“”‘’]")


# --------------------------------------------------------------------------- OCR

def vision_available() -> str:
    """Empty string when Vision can be used, otherwise the reason it cannot."""
    if os.uname().sysname != "Darwin":
        return "Apple Vision OCR is only available on macOS"
    if fitz is None:
        return "PyMuPDF is not installed (pip install pymupdf)"
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except ImportError:
        return ("the PyObjC Vision bindings are not installed "
                "(pip install pyobjc-framework-Vision pyobjc-framework-Quartz)")
    return ""


def ocr_png(png: bytes, langs=DEFAULT_LANGS) -> list[dict]:
    """Recognise text in a PNG; one dict per line with a normalised box.

    Boxes use Vision's convention: fractions of the image with the origin at
    the bottom-left corner.
    """
    import Quartz
    import Vision
    from Foundation import NSData

    data = NSData.dataWithBytes_length_(png, len(png))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(list(langs))
    request.setUsesLanguageCorrection_(True)
    try:
        request.setRevision_(Vision.VNRecognizeTextRequestRevision3)
    except Exception:  # older macOS: keep the default revision
        pass
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision failed: {error}")
    lines = []
    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        best = candidates[0]
        box = obs.boundingBox()
        lines.append({
            "t": str(best.string()), "c": round(float(best.confidence()), 3),
            "x": round(box.origin.x, 4), "y": round(box.origin.y, 4),
            "w": round(box.size.width, 4), "h": round(box.size.height, 4),
        })
    return lines


def _cache_path(cache_dir: Path, page: int) -> Path:
    return cache_dir / f"p{page:04d}.json"


def _ocr_worker(job: tuple) -> int:
    pdf, pages, cache_dir, dpi, langs = job
    doc = fitz.open(str(pdf))
    done = 0
    for page in pages:
        path = _cache_path(cache_dir, page)
        if path.exists():
            continue
        pix = doc[page].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        lines = ocr_png(pix.tobytes("png"), langs)
        record = {"page": page, "width": pix.width, "height": pix.height, "lines": lines}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        done += 1
    doc.close()
    return done


def run_ocr(pdf: Path, cache_dir: Path, pages: list[int], *, dpi: int = DEFAULT_DPI,
            langs=DEFAULT_LANGS, workers: int = 3, progress=None) -> None:
    """Recognise every page not yet in the cache. Safe to interrupt and rerun."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pages if not _cache_path(cache_dir, p).exists()]
    if not todo:
        return
    workers = max(1, min(workers, len(todo)))
    # Round-robin so every worker sees a similar mix of dense and sparse pages.
    jobs = [(pdf, todo[i::workers], cache_dir, dpi, tuple(langs)) for i in range(workers)]
    if workers == 1:
        _ocr_worker(jobs[0])
        return
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        async_results = [pool.apply_async(_ocr_worker, (job,)) for job in jobs]
        started = time.time()
        last = -1
        while any(not r.ready() for r in async_results):
            time.sleep(1.0)
            if progress is not None:
                finished = sum(1 for p in todo if _cache_path(cache_dir, p).exists())
                if finished != last:
                    progress(finished, len(todo), time.time() - started)
                    last = finished
        for r in async_results:
            r.get()


# ------------------------------------------------------------------------ geometry

@dataclass
class Line:
    text: str
    x0: float
    x1: float
    top: float      # top-left origin, fractions of the page
    bottom: float
    conf: float = 1.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Block:
    """A unit of page content in reading order."""
    kind: str                       # p | h2..h6 | table | image
    text: str = ""
    indented: bool = False          # first line was indented (new paragraph)
    ends_full: bool = False         # last line reached the right margin
    top: float = 0.0
    bottom: float = 0.0
    grid: list[list[str]] | None = None   # table cells, for joining across pages


@dataclass
class Layout:
    """Document-wide measurements taken from every page."""
    line_h: float                   # typical body line height
    left: float                     # body left margin
    right: float                    # body right margin
    char_w: float                   # approximate width of one CJK character
    header_texts: Counter = field(default_factory=Counter)
    title_norm: str = ""


def normalise(text: str) -> str:
    return NORMALISE_RE.sub("", text).lower()


def char_count(text: str) -> float:
    """Width of a line in CJK character units (Latin glyphs count as half)."""
    return sum(1.0 if CJK_RE.match(ch) else 0.5 for ch in text if not ch.isspace())


@dataclass
class Metrics:
    """Margins and glyph size of one page; scans shift and scale page by page."""
    left: float
    right: float
    char_w: float
    line_h: float


def page_metrics(lines: list[Line], layout: Layout) -> Metrics:
    long = [l for l in lines if l.width > 0.5 and HEADER_ZONE <= l.top and l.bottom <= FOOTER_ZONE]
    if len(long) < 3:
        return Metrics(layout.left, layout.right, layout.char_w, layout.line_h)
    lefts = sorted(l.x0 for l in long)
    rights = sorted(l.x1 for l in long)
    left = lefts[len(lefts) // 10]
    right = rights[-1 - len(rights) // 10]
    widths = sorted(l.width / max(char_count(l.text), 1) for l in long)
    char_w = widths[len(widths) // 2]
    line_h = statistics.median(l.height for l in long)
    return Metrics(left, right, char_w, line_h)


def load_page(path: Path) -> list[Line]:
    record = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for item in record.get("lines", []):
        text = item["t"].strip()
        if not text:
            continue
        top = 1.0 - (item["y"] + item["h"])
        lines.append(Line(text, item["x"], item["x"] + item["w"], top, top + item["h"],
                          item.get("c", 1.0)))
    lines.sort(key=lambda l: (l.top, l.x0))
    return lines


def measure(pages: dict[int, list[Line]], title: str) -> Layout:
    heights, lefts, rights = [], Counter(), Counter()
    headers: Counter = Counter()
    for lines in pages.values():
        for line in lines:
            if line.top < HEADER_ZONE or line.bottom > FOOTER_ZONE:
                headers[normalise(line.text)] += 1
            if line.width > 0.5 and HEADER_ZONE <= line.top and line.bottom <= FOOTER_ZONE:
                heights.append(line.height)
                lefts[round(line.x0, 3)] += 1
                rights[round(line.x1, 3)] += 1
    line_h = statistics.median(heights) if heights else 0.012
    left = lefts.most_common(1)[0][0] if lefts else 0.08
    right = rights.most_common(1)[0][0] if rights else 0.92
    # A CJK glyph is about as wide as the line box is tall, less the leading.
    char_w = line_h * 0.9
    widths = sorted(l.width / max(char_count(l.text), 1)
                    for lines in pages.values() for l in lines if l.width > 0.5)
    if widths:
        char_w = widths[len(widths) // 2]
    return Layout(line_h, left, right, char_w, headers, normalise(title))


# ----------------------------------------------------------------- page analysis

def strip_furniture(lines: list[Line], layout: Layout) -> list[Line]:
    kept = []
    for line in lines:
        norm = normalise(line.text)
        short = len(norm) <= 2
        if line.top < HEADER_ZONE:
            if (layout.header_texts[norm] >= 3 or norm == layout.title_norm or short
                    or PAGE_NUMBER_RE.match(line.text) or CONTINUED_RE.match(line.text)
                    or (layout.title_norm and norm in layout.title_norm)):
                continue
        if line.bottom > FOOTER_ZONE and (PAGE_NUMBER_RE.match(line.text) or short
                                          or layout.header_texts[norm] >= 3):
            continue
        if line.top < 0.17 and line.width < 0.5 and norm and (
                layout.header_texts[norm] >= 3 or norm == layout.title_norm):
            continue   # a running header on a page scanned lower than the others
        if line.x0 > 0.9 and line.width < 0.08 and len(norm) <= 3:
            continue   # vertical tab on the fore-edge
        if CONTINUED_RE.match(line.text):
            line.text = CONTINUED_RE.sub("", line.text).strip()
            if not line.text:
                continue
        kept.append(line)
    return kept


CONTENTS_NUMBER_RE = re.compile(r"^[\s.。·•…‥⋯:：,，=_\-~～]*\d{1,4}\s*$")
CONTENTS_ENTRY_RE = re.compile(
    rf"^(第[{CN_NUM}\d]+[章节篇编卷部]|[{CN_NUM}]+、|[（(][{CN_NUM}]+[)）]|专\s*记|附\s*录|概\s*述|"
    r"(?i:chapter|section|part|appendix|preface|overview|introduction)\b)")


def is_contents_page(rows: list[list[Line]]) -> bool:
    """A printed table of contents: heading-like entries beside page numbers.

    The leaders between an entry and its number come back from OCR in odd
    pieces ("⋯56", "•116", "：118"), sometimes as their own cell, so cells are
    counted rather than parsed. A statistical table also holds many small
    numbers, but its text cells are not chapter and section titles.
    """
    if len(rows) < 8:
        return False
    numbers = entries = 0
    for row in rows:
        for cell in row:
            text = cell.text.strip()
            if CONTENTS_NUMBER_RE.match(text):
                numbers += 1
            else:
                stripped = re.sub(r"[\s.。·•…‥⋯:：,，=_\-~～\d]+$", "", text)
                if CONTENTS_ENTRY_RE.match(stripped) and TOC_LINE_RE.search(text):
                    numbers += 1
                    entries += 1
                elif CONTENTS_ENTRY_RE.match(stripped):
                    entries += 1
    if entries >= 0.4 * len(rows) and numbers >= 0.3 * len(rows):
        return True
    # Numbers lost to the OCR: still a contents page when nearly every row is a
    # title and nothing on the page is as long as a line of prose.
    long_lines = sum(1 for row in rows for cell in row if cell.width > 0.6)
    return entries >= 0.6 * len(rows) and long_lines == 0


def group_rows(lines: list[Line]) -> list[list[Line]]:
    """Lines on one baseline form a row, left to right.

    Scans are slightly skewed, so a row is followed cell by cell: a line joins
    a row when it overlaps its horizontally nearest member by more than half a
    line height and does not sit above or below another member.
    """
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: (l.top, l.x0)):
        best = None
        for row in rows[-12:]:
            nearest = min(row, key=lambda m: abs(m.cx - line.cx))
            overlap = min(nearest.bottom, line.bottom) - max(nearest.top, line.top)
            if overlap < 0.55 * min(line.height, nearest.height):
                continue
            if any(min(m.x1, line.x1) - max(m.x0, line.x0) > 0.3 * min(m.width, line.width)
                   for m in row):
                continue
            best = row
        if best is not None:
            best.append(line)
        else:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: l.x0)
    rows.sort(key=lambda r: min(l.top for l in r))
    return rows


def row_box(row: list[Line]) -> tuple[float, float, float, float]:
    return (min(l.x0 for l in row), max(l.x1 for l in row),
            min(l.top for l in row), max(l.bottom for l in row))


def find_tables(rows: list[list[Line]], layout: Layout) -> list[tuple[int, int]]:
    """Index ranges [start, end) of rows that make up tables."""
    def tabular(row: list[Line]) -> bool:
        if len(row) < 2:
            return False
        return max(l.width for l in row) < 0.45 * (layout.right - layout.left) + 0.05

    spans = []
    i = 0
    while i < len(rows):
        if not tabular(rows[i]):
            i += 1
            continue
        j = i
        count = 0
        while j < len(rows):
            if tabular(rows[j]):
                count += 1
                j += 1
                continue
            # One short single cell inside the table (a wrapped cell) is allowed.
            if (len(rows[j]) == 1 and rows[j][0].width < 0.3 and j + 1 < len(rows)
                    and tabular(rows[j + 1]) and count >= 1):
                j += 1
                continue
            break
        if count >= TABLE_MIN_ROWS:
            spans.append((i, j))
        i = j
    return spans


def cluster_columns(cells: list[Line], tolerance: float) -> list[float]:
    centres = sorted(c.cx for c in cells)
    clusters: list[list[float]] = []
    for c in centres:
        if clusters and c - clusters[-1][-1] <= tolerance:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return [sum(cl) / len(cl) for cl in clusters]


NUMERIC_CELL_RE = re.compile(r"^[\d\s.,%~～\-–—+±/:：年月日万亿元]+$")


def numeric_row(cells: list[str]) -> bool:
    filled = [c for c in cells if c]
    numeric = sum(1 for c in filled if NUMERIC_CELL_RE.match(c))
    return len(filled) >= 2 and numeric >= max(1, len(filled) / 3)


def table_grid(rows: list[list[Line]], layout: Layout) -> list[list[str]]:
    cells = [c for row in rows for c in row]
    columns = cluster_columns(cells, 2.2 * layout.char_w)
    grid: list[list[str]] = []
    for row in rows:
        out = [""] * len(columns)
        for cell in row:
            col = min(range(len(columns)), key=lambda k: abs(columns[k] - cell.cx))
            text = cell.text.replace("|", "\\|")
            out[col] = join_lines([out[col], text]) if out[col] else text
        grid.append(out)
    keep = [k for k in range(len(columns)) if any(r[k] for r in grid)]
    grid = [[r[k] for k in keep] for r in grid]
    if not grid:
        return grid
    # Header lines stacked above the first data row ("货运量" over "（万吨）") are
    # one header; a wrapped note that continues alone under its cell is one cell.
    first_data = next((i for i, r in enumerate(grid[:5]) if numeric_row(r)), None)
    header_rows = first_data if first_data else 1
    if first_data is None and len(grid) > 2:
        # No numbers to go by: a second row that fills gaps left in the first
        # ("正职 / 副职" under "职务") is still part of the header.
        if grid[0][0] and not grid[1][0] and any(grid[1][k] and not grid[0][k] for k in range(len(grid[0]))):
            header_rows = 2
    if 1 < header_rows <= 4:
        header = ["" for _ in grid[0]]
        for r in grid[:header_rows]:
            for k, c in enumerate(r):
                header[k] = join_lines([header[k], c]) if header[k] else c
        grid = [header] + grid[header_rows:]
    merged: list[list[str]] = []
    for r in grid:
        filled = [k for k, c in enumerate(r) if c]
        if merged and len(filled) == 1 and filled[0] != 0 and merged[-1][filled[0]] \
                and not numeric_row(r):
            k = filled[0]
            merged[-1][k] = join_lines([merged[-1][k], r[k]])
            continue
        merged.append(r)
    return merged


def grid_markdown(grid: list[list[str]]) -> str:
    if not grid or len(grid[0]) < 2:
        return "\n".join(" ".join(c for c in r if c) for r in grid)
    width = len(grid[0])
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * width]
    for r in grid[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def table_markdown(rows: list[list[Line]], layout: Layout) -> str:
    return grid_markdown(table_grid(rows, layout))


def heading_level(text: str, line: Line, layout: "Layout | Metrics", centred: bool) -> int | None:
    compact = text.replace(" ", "").replace("　", "")
    if len(compact) > 40 or text[-1:] in TERMINAL:
        return None
    short = line.width < 0.6 * (layout.right - layout.left)
    big = line.height > 1.25 * layout.line_h
    if CHAPTER_RE.match(compact) and (centred or big) and short:
        return 3
    if SECTION_RE.match(compact) and (centred or big) and short:
        return 4
    if SPECIAL_HEAD_RE.match(text) and (centred or big):
        return 3
    if SUB1_RE.match(compact) and short and len(compact) <= 24:
        return 5
    if SUB2_RE.match(compact) and short and len(compact) <= 24:
        return 6
    return None


def join_lines(parts: list[str]) -> str:
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and not (CJK_RE.match(out[-1]) or CJK_RE.match(part[0])):
            if out.endswith("-") and out[-2:-1].isalpha() and part[0].islower():
                out = out[:-1]
            else:
                out += " "
        out += part
    return out


def tidy_heading(text: str) -> str:
    text = text.strip()
    # "概  述" / "第四章  其他通信": collapse spaced-out characters but keep one gap
    # between a chapter number and its title.
    m = re.match(rf"^(第[{CN_NUM}\d]+[章节篇编卷部])\s*(.*)$", text)
    if m:
        return f"{m.group(1)} {re.sub(r'\s+', '', m.group(2))}".strip()
    if len(text.replace(" ", "")) <= 6:
        return text.replace(" ", "")
    return re.sub(r"\s+", " ", text)


class PageAnalyser:
    def __init__(self, layout: Layout, toc: dict[int, list[tuple[int, str]]],
                 figure_sink=None, keep_contents: bool = False):
        self.layout = layout
        self.toc = toc
        self.figure_sink = figure_sink      # callable(page, top, bottom) -> str | None
        self.keep_contents = keep_contents
        self.recent_headings: list[str] = []

    def drop_picture_text(self, page: int, rows: list[list[Line]], metrics: Metrics) -> list[list[Line]]:
        """Remove lines read off signs and banners inside photographs.

        Such a line is short, has open space above or below it, and has ink
        beside it on the same band; a caption under a picture has blank paper
        beside it.
        """
        inside = getattr(self.figure_sink, "inside_picture", None)
        if inside is None or len(rows) < 1:
            return rows
        body_w = metrics.right - metrics.left
        kept = []
        for i, row in enumerate(rows):
            x0, x1, top, bottom = row_box(row)
            above = top - (row_box(rows[i - 1])[3] if i else HEADER_ZONE)
            below = (row_box(rows[i + 1])[2] if i + 1 < len(rows) else FOOTER_ZONE) - bottom
            if (x1 - x0) < 0.7 * body_w and max(above, below) > 0.03 and inside(page, x0, x1, top, bottom):
                continue
            kept.append(row)
        return kept

    def analyse(self, page: int, lines: list[Line], prev_blocks: list[Block]) -> list[Block]:
        layout = self.layout
        lines = strip_furniture(lines, layout)
        entries = list(self.toc.get(page, []))
        text_len = sum(len(l.text) for l in lines)

        blocks: list[Block] = []
        if entries and (text_len < 150 or any(level == 1 for level, _ in entries)) \
                and text_len < 400:
            # A part-title or chapter-opening page carrying little else.
            if page == 0 and self.figure_sink is not None:
                ref = self.figure_sink(page, HEADER_ZONE, FOOTER_ZONE)
                if ref:
                    blocks.append(Block("image", ref))
            for level, title in entries:
                blocks.append(Block(f"h{min(level + 1, 6)}", tidy_heading(title)))
                self.recent_headings.append(normalise(title))
            return blocks
        metrics = page_metrics(lines, layout)
        rows = group_rows(lines)
        if not self.keep_contents and is_contents_page(rows):
            return []
        rows = self.drop_picture_text(page, rows, metrics)
        tables = find_tables(rows, layout)
        table_at = {start: end for start, end in tables}

        # Outline entries for this page: match them to lines, else emit at the top.
        pending = [(level, title, normalise(title)) for level, title in entries]
        matched: dict[int, tuple[int, str]] = {}
        for idx, row in enumerate(rows):
            norm = normalise(join_lines([l.text for l in row]))
            for k, (level, title, tnorm) in enumerate(pending):
                if norm and (norm == tnorm or (len(norm) >= 2 and len(tnorm) >= 2 and
                                                (norm in tnorm or tnorm in norm))):
                    matched[idx] = (level, title)
                    pending.pop(k)
                    break
        for level, title, tnorm in pending:
            if tnorm in self.recent_headings[-6:]:
                continue
            blocks.append(Block(f"h{min(level + 1, 6)}", tidy_heading(title)))
            self.recent_headings.append(tnorm)

        content_top = HEADER_ZONE
        para: list[str] | None = None
        para_meta: Block | None = None
        last_bottom = content_top
        last_line: Line | None = None
        last_centred = False

        def flush():
            nonlocal para, para_meta
            if para and para_meta is not None:
                para_meta.text = join_lines(para)
                blocks.append(para_meta)
            para, para_meta = None, None

        def figure_between(top: float, bottom: float):
            if self.figure_sink is None or bottom - top < FIGURE_MIN_GAP:
                return
            ref = self.figure_sink(page, top, bottom)
            if ref:
                flush()
                blocks.append(Block("image", ref, top=top, bottom=bottom))

        idx = 0
        while idx < len(rows):
            if idx in table_at:
                end = table_at[idx]
                flush()
                x0, x1, top, bottom = row_box([l for r in rows[idx:end] for l in r])
                figure_between(last_bottom, top)
                grid = table_grid(rows[idx:end], layout)
                blocks.append(Block("table", grid_markdown(grid), top=top, bottom=bottom,
                                    grid=grid))
                last_bottom = bottom
                last_line = None
                idx = end
                continue
            row = rows[idx]
            x0, x1, top, bottom = row_box(row)
            text = join_lines([l.text for l in row]) if len(row) > 1 else row[0].text
            if len(row) > 1:
                # Side-by-side fragments on one baseline (a label and a unit line).
                text = "　".join(l.text for l in row)
            line = row[0]
            figure_between(last_bottom, top)

            body_w = metrics.right - metrics.left
            centred = (abs((x0 + x1) / 2 - (metrics.left + metrics.right) / 2) < 2 * metrics.char_w
                       and (x1 - x0) < 0.75 * body_w and x0 - metrics.left > 2.5 * metrics.char_w)
            indented = metrics.char_w * 1.1 < x0 - metrics.left < metrics.char_w * 4.5
            gap = top - last_bottom if last_line is not None else 0.0

            if idx in matched:
                flush()
                level, title = matched[idx]
                blocks.append(Block(f"h{min(level + 1, 6)}", tidy_heading(title)))
                self.recent_headings.append(normalise(title))
                last_bottom, last_line, last_centred = bottom, line, False
                idx += 1
                continue

            level = heading_level(text, line, metrics, centred)
            if level is not None:
                norm = normalise(text)
                flush()
                if norm not in self.recent_headings[-4:]:
                    blocks.append(Block(f"h{level}", tidy_heading(text), top=top, bottom=bottom))
                    self.recent_headings.append(norm)
                last_bottom, last_line, last_centred = bottom, line, False
                idx += 1
                continue

            prev_short = (last_line is not None and
                          last_line.x1 < metrics.right - 1.5 * metrics.char_w)
            starts_new = (para is None or indented or centred or last_centred
                          or gap > 1.8 * metrics.line_h
                          or (prev_short and (para_meta is None or para[-1][-1:] in TERMINAL))
                          or (last_line is not None and len(row) > 1))
            if starts_new:
                flush()
                para = [text]
                para_meta = Block("p", indented=indented or centred, top=top, bottom=bottom)
            else:
                para.append(text)
                para_meta.bottom = bottom
            para_meta.ends_full = x1 >= metrics.right - 1.5 * metrics.char_w
            last_centred = centred
            last_bottom, last_line = bottom, line
            idx += 1
        flush()
        figure_between(last_bottom, FOOTER_ZONE)
        if len(self.recent_headings) > 50:
            del self.recent_headings[:-20]
        return blocks


def merge_pages(all_blocks: list[list[Block]]) -> list[Block]:
    """Join a paragraph cut by a page break with its continuation."""
    merged: list[Block] = []
    for blocks in all_blocks:
        if not blocks:
            continue
        first = blocks[0]
        if (merged and merged[-1].kind == "p" and first.kind == "p"
                and not first.indented and merged[-1].ends_full):
            merged[-1].text = join_lines([merged[-1].text, first.text])
            merged[-1].ends_full = first.ends_full
            blocks = blocks[1:]
        elif (merged and merged[-1].kind == "table" and first.kind == "table"
              and merged[-1].grid and first.grid
              and len(merged[-1].grid[0]) == len(first.grid[0])
              and first.top < HEADER_ZONE + 0.1):
            # The same table continued on the next page, with or without its
            # header repeated.
            prev = merged[-1]
            rows = first.grid[1:] if first.grid[0] == prev.grid[0] or not numeric_row(first.grid[0]) \
                else first.grid
            prev.grid = prev.grid + rows
            prev.text = grid_markdown(prev.grid)
            prev.bottom = first.bottom
            blocks = blocks[1:]
        merged.extend(blocks)
    return merged


def to_markdown(blocks: list[Block], *, title: str, source: str, add_title: bool,
                provenance: bool) -> str:
    out: list[str] = []
    if provenance:
        out.append(f"<!-- Generated by hy-ebook-translator ocr2md (Apple Vision) from {source} "
                   f"on {time.strftime('%Y-%m-%d')} -->")
    if add_title:
        out.append(f"# {' '.join(title.split())}")
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if block.kind.startswith("h"):
            out.append(f"{'#' * int(block.kind[1])} {text}")
        elif block.kind in {"table", "image"}:
            out.append(text)
        else:
            out.append(LINE_START_ESCAPE_RE.sub(r"\1\\\2", text))
    markdown = "\n\n".join(out).rstrip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", markdown)


# ------------------------------------------------------------------- figures

class FigureCropper:
    """Decides whether a text-free gap holds a picture and saves it if so."""

    def __init__(self, doc, images_dir: Path | None, images_rel: str):
        self.doc = doc
        self.images_dir = images_dir
        self.images_rel = images_rel.rstrip("/")
        self.count = 0
        self.files: list[Path] = []
        self._probe_cache: dict[int, tuple[bytes, int, int]] = {}

    def _probe(self, page: int) -> tuple[bytes, int, int]:
        if page not in self._probe_cache:
            pix = self.doc[page].get_pixmap(dpi=36, colorspace=fitz.csGRAY)
            self._probe_cache = {page: (bytes(pix.samples), pix.width, pix.height)}
        return self._probe_cache[page]

    def ink_box(self, page: int, top: float, bottom: float) -> tuple[float, float, float, float] | None:
        """Bounding box of the ink in a gap, or None when it is only a rule or blank.

        A thin rule under a running header darkens one or two pixel rows; a
        picture or chart darkens most of them. A scan often catches the book's
        fore-edge as a dark strip along one side of the page, so a narrow run
        of ink touching the edge is not a picture.
        """
        samples, w, h = self._probe(page)
        y0, y1 = int(top * h), int(bottom * h)
        x0, x1 = int(0.06 * w), int(0.90 * w)
        if y1 - y0 < 3 or x1 <= x0:
            return None
        width = x1 - x0
        col_ink = [0] * width
        row_dark: list[list[int]] = []
        for y in range(y0, y1):
            row = samples[y * w + x0: y * w + x1]
            dark = [i for i, v in enumerate(row) if v < 150]
            row_dark.append(dark)
            for i in dark:
                col_ink[i] += 1
        # Runs of columns that hold ink on a fair share of the rows.
        threshold = max(2, 0.15 * (y1 - y0))
        runs: list[list[int]] = []
        for i, n in enumerate(col_ink):
            if n >= threshold:
                if runs and i - runs[-1][1] <= 0.02 * w:
                    runs[-1][1] = i
                else:
                    runs.append([i, i])
        runs = [r for r in runs
                if not ((r[0] <= 1 or r[1] >= width - 2) and r[1] - r[0] < 0.4 * w)]
        if not runs:
            return None
        c0, c1 = runs[0][0], runs[-1][1] + 1
        if (c1 - c0) < 0.2 * w:
            return None
        inked_rows = [y0 + k for k, dark in enumerate(row_dark)
                      if sum(1 for i in dark if c0 <= i < c1) > 0.03 * (c1 - c0)]
        if len(inked_rows) < max(3, 0.2 * (y1 - y0)):
            return None
        return ((x0 + c0) / w, (x0 + c1) / w, inked_rows[0] / h, (inked_rows[-1] + 1) / h)

    def inside_picture(self, page: int, x0: float, x1: float, top: float, bottom: float) -> bool:
        samples, w, h = self._probe(page)
        y0, y1 = max(0, int((top - 0.003) * h)), min(h, int((bottom + 0.003) * h) + 1)
        spans = [(int(0.06 * w), int((x0 - 0.015) * w)), (int((x1 + 0.015) * w), int(0.90 * w))]
        dark = total = 0
        for y in range(y0, y1):
            for a, b in spans:
                if b - a < 3:
                    continue
                row = samples[y * w + a: y * w + b]
                dark += sum(1 for v in row if v < 150)
                total += len(row)
        return total > 0 and dark / total > 0.08

    def __call__(self, page: int, top: float, bottom: float) -> str | None:
        pad = 0.004
        top, bottom = max(0.0, top + pad), min(1.0, bottom - pad)
        box = self.ink_box(page, top, bottom)
        if box is None:
            return None
        self.count += 1
        if self.images_dir is None:
            return None
        bx0, bx1, btop, bbottom = box
        margin = 0.01
        rect = self.doc[page].rect
        clip = fitz.Rect(rect.x0 + max(0.0, bx0 - margin) * rect.width,
                         rect.y0 + max(top, btop - margin) * rect.height,
                         rect.x0 + min(1.0, bx1 + margin) * rect.width,
                         rect.y0 + min(bottom, bbottom + margin) * rect.height)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        name = f"page{page + 1:04d}_{self.count:03d}.jpg"
        path = self.images_dir / name
        pix = self.doc[page].get_pixmap(dpi=FIGURE_DPI, clip=clip, colorspace=fitz.csRGB)
        pix.save(str(path), output="jpeg", jpg_quality=72)
        self.files.append(path)
        rel = f"{self.images_rel}/{name}" if self.images_rel else name
        rel = rel.replace(" ", "%20")
        return f"![第{page + 1}页图](<{rel}>)" if " " in rel else f"![第{page + 1}页图]({rel})"


# ------------------------------------------------------------------- driver

@dataclass
class ConvertResult:
    markdown: str
    title: str
    page_count: int
    files: list[Path]
    warnings: list[str]
    figures: int = 0
    tables: int = 0
    headings: int = 0


def outline_by_page(doc, pages: dict[int, list[Line]] | None = None,
                    reach: int = 3) -> dict[int, list[tuple[int, str]]]:
    """Outline entries keyed by 0-based page.

    Bookmark targets are often a page or two off, so when the recognised pages
    are given each entry is moved to the nearest page within `reach` whose text
    actually contains the title.
    """
    toc: dict[int, list[tuple[int, str]]] = {}
    norm_pages: dict[int, list[str]] = {}
    if pages:
        norm_pages = {p: [normalise(l.text) for l in lines if HEADER_ZONE <= l.top]
                      for p, lines in pages.items()}

    def found_on(page: int, tnorm: str) -> bool:
        # Only a line that is the whole title counts; a mention inside a
        # sentence or a running header is not where the heading stands.
        return tnorm in norm_pages.get(page, ())

    last_page = -1
    for level, title, page in doc.get_toc():
        title = " ".join(title.split())
        if page < 1 or not title:
            continue
        target = page - 1
        if norm_pages:
            tnorm = normalise(title)
            for delta in sorted(range(-reach, reach + 1), key=lambda d: (abs(d), d)):
                candidate = target + delta
                if candidate < last_page or candidate not in norm_pages:
                    continue
                if found_on(candidate, tnorm):
                    target = candidate
                    break
        target = max(target, last_page)
        last_page = target
        toc.setdefault(target, []).append((level, title))
    return toc


def convert_scanned_pdf(pdf: Path, *, cache_dir: Path, images_dir: Path | None,
                        images_rel: str = "", title: str | None = None,
                        add_title: bool = True, provenance: bool = True,
                        dpi: int = DEFAULT_DPI, langs=DEFAULT_LANGS, workers: int = 3,
                        pages: list[int] | None = None, keep_contents: bool = False,
                        progress=None) -> ConvertResult:
    reason = vision_available()
    if reason:
        raise RuntimeError(reason)
    doc = fitz.open(str(pdf))
    page_list = pages if pages is not None else list(range(doc.page_count))
    run_ocr(pdf, cache_dir, page_list, dpi=dpi, langs=langs, workers=workers,
            progress=progress)

    ocr_pages = {p: load_page(_cache_path(cache_dir, p)) for p in page_list}
    meta_title = ((doc.metadata or {}).get("title") or "").strip()
    book_title = title or (meta_title if not meta_title.lower().endswith(".pdf") else "") \
        or pdf.stem.replace("_", " ")
    layout = measure(ocr_pages, book_title)
    toc = outline_by_page(doc, ocr_pages)
    cropper = FigureCropper(doc, images_dir, images_rel)
    analyser = PageAnalyser(layout, toc, figure_sink=cropper, keep_contents=keep_contents)

    all_blocks: list[list[Block]] = []
    previous: list[Block] = []
    recent_texts: list[str] = []
    duplicates: list[int] = []
    for p in page_list:
        # Some scans repeat a page (a chapter-opening spread bound in twice).
        signature = normalise("".join(l.text for l in ocr_pages[p]))
        if len(signature) > 80 and signature in recent_texts:
            duplicates.append(p)
            continue
        recent_texts = (recent_texts + [signature])[-4:]
        blocks = analyser.analyse(p, ocr_pages[p], previous)
        all_blocks.append(blocks)
        previous = blocks
    blocks = merge_pages(all_blocks)
    markdown = to_markdown(blocks, title=book_title, source=pdf.name,
                           add_title=add_title, provenance=provenance)
    warnings: list[str] = []
    if duplicates:
        shown = ", ".join(str(p + 1) for p in duplicates[:8])
        warnings.append(f"{len(duplicates)} page(s) repeated an earlier page and were "
                        f"skipped (e.g. page {shown})")
    low = sum(1 for lines in ocr_pages.values() for l in lines if l.conf < 0.5)
    total = sum(len(lines) for lines in ocr_pages.values())
    if total and low / total > 0.2:
        warnings.append(f"{low} of {total} recognised lines have low confidence; "
                        "a higher --dpi may help")
    result = ConvertResult(markdown, book_title, doc.page_count, cropper.files, warnings,
                           figures=len(cropper.files),
                           tables=sum(1 for b in blocks if b.kind == "table"),
                           headings=sum(1 for b in blocks if b.kind.startswith("h")))
    doc.close()
    return result
