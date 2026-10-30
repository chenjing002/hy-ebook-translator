"""Convert a scanned PDF to Markdown with Apple Vision OCR (macOS only).

Examples:
  python3 scripts/ocr_pdf_to_markdown.py scan.pdf
  python3 scripts/ocr_pdf_to_markdown.py scan.pdf -o "inputs/Book Title.md"
  python3 scripts/ocr_pdf_to_markdown.py scan.pdf --pages 100-120 -o /tmp/sample.md
  python3 scripts/ocr_pdf_to_markdown.py scan.pdf --workers 4 --dpi 250

Each page is rendered with PyMuPDF and read by the macOS Vision text
recogniser; headings come from the PDF outline and from Chinese chapter and
section patterns, paragraphs from indents and line geometry, tables from cell
positions, and pictures are cropped from text-free gaps that contain ink.
Per-page OCR results are cached in `.state/ocr/<pdf stem>/` so a run resumes
after an interruption and the layout pass can be rerun without recognising
again. The Markdown is written next to the PDF unless -o or --output-dir says
otherwise; figures go to `<markdown stem>_files/` beside it.

Needs: pip install pymupdf pyobjc-framework-Vision pyobjc-framework-Quartz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hymt.ocr2md import DEFAULT_DPI, DEFAULT_LANGS, convert_scanned_pdf, vision_available


def parse_pages(spec: str | None, count: int) -> list[int] | None:
    """"5-20,30" (1-based, inclusive) → 0-based page indices."""
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a) if a else 1
            hi = int(b) if b else count
            pages.extend(range(lo - 1, min(hi, count)))
        elif part:
            pages.append(int(part) - 1)
    return sorted({p for p in pages if 0 <= p < count})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="scanned PDF file")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("-o", "--output", type=Path, help="Markdown file to write")
    where.add_argument("--output-dir", type=Path, help="directory for <pdf stem>.md and its figures")
    parser.add_argument("--title", help="title for the top heading (default: PDF metadata, else file name)")
    parser.add_argument("--no-title", action="store_true", help="do not add a title heading")
    parser.add_argument("--no-images", action="store_true", help="do not crop figures from the pages")
    parser.add_argument("--pages", help="1-based page selection, e.g. 1-50 or 3,7,100-120")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"render resolution for OCR (default {DEFAULT_DPI})")
    parser.add_argument("--langs", default=",".join(DEFAULT_LANGS),
                        help="Vision recognition languages, most likely first (default zh-Hans,en-US)")
    parser.add_argument("--workers", type=int, default=3, help="parallel OCR processes (default 3)")
    parser.add_argument("--cache-dir", type=Path, help="per-page OCR cache (default .state/ocr/<pdf stem>)")
    parser.add_argument("--keep-contents", action="store_true", help="keep printed table-of-contents pages")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing Markdown file")
    args = parser.parse_args(argv)

    reason = vision_available()
    if reason:
        print(f"error: {reason}", file=sys.stderr)
        return 2
    import pymupdf  # noqa: F401  (checked by vision_available)

    pdf = args.pdf
    if not pdf.is_file():
        print(f"error: {pdf} is not a file", file=sys.stderr)
        return 2
    if args.output:
        out = args.output
    elif args.output_dir:
        out = args.output_dir / f"{pdf.stem}.md"
    else:
        out = pdf.with_suffix(".md")
    if out.exists() and not args.overwrite:
        print(f"error: {out} exists (use --overwrite)", file=sys.stderr)
        return 1

    count = pymupdf.open(str(pdf)).page_count
    pages = parse_pages(args.pages, count)
    cache_dir = args.cache_dir or (ROOT / ".state" / "ocr" / pdf.stem)
    images_dir = None if args.no_images else out.parent / f"{out.stem}_files"
    if images_dir is not None and images_dir.is_dir() and args.overwrite:
        for stale in images_dir.glob("page[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9].jpg"):
            stale.unlink()

    def progress(done: int, total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        print(f"\r  OCR {done}/{total} pages  {elapsed:5.0f}s elapsed  ~{eta:4.0f}s left   ",
              end="", file=sys.stderr, flush=True)

    started = time.time()
    print(f"{pdf.name}: {len(pages) if pages is not None else count} page(s), cache {cache_dir}")
    result = convert_scanned_pdf(
        pdf, cache_dir=cache_dir, images_dir=images_dir, images_rel=images_dir.name if images_dir else "",
        title=args.title, add_title=not args.no_title, dpi=args.dpi,
        langs=[s.strip() for s in args.langs.split(",") if s.strip()], workers=args.workers,
        pages=pages, keep_contents=args.keep_contents, progress=progress)
    print(file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.markdown, encoding="utf-8")
    print(f"wrote {out}  ({len(result.markdown):,} chars, {result.headings} headings, "
          f"{result.tables} tables, {result.figures} figures) in {time.time() - started:.0f}s")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
