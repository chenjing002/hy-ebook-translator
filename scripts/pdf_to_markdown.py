"""Convert a PDF to a Markdown file, with its images beside it.

Examples:
  python3 scripts/pdf_to_markdown.py paper.pdf
  python3 scripts/pdf_to_markdown.py book.pdf -o "inputs/Book Title.md"
  python3 scripts/pdf_to_markdown.py inputs/pdfs/*.pdf --output-dir inputs/
  python3 scripts/pdf_to_markdown.py book.pdf --backend pdftotext --no-images

PyMuPDF (`pip install pymupdf`) gives the best result: headings by font size,
bold and italic, tables and images. Without it, poppler's `pdftotext` is used
for text only. The Markdown is written next to the PDF unless -o or
--output-dir says otherwise; images go to `<markdown stem>_files/` beside it.
A converted PDF usually still benefits from `scripts/preprocess_markdown.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hymt.pdf2md import available_backends, convert_pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", type=Path, nargs="+", help="PDF file(s)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("-o", "--output", type=Path, help="Markdown file to write (one PDF only)")
    where.add_argument("--output-dir", type=Path, help="directory for <pdf stem>.md and its images")
    parser.add_argument("--title", help="title for the top heading (default: PDF metadata, else file name)")
    parser.add_argument("--no-title", action="store_true", help="do not add a title heading and author line")
    parser.add_argument("--no-images", action="store_true", help="do not extract images")
    parser.add_argument("--backend", choices=["auto", "pymupdf", "pdftotext"], default="auto",
                        help=f"PDF reader to use (available: {', '.join(available_backends()) or 'none'})")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing Markdown file")
    args = parser.parse_args(argv)
    if args.output and len(args.pdfs) != 1:
        parser.error("-o/--output takes exactly one PDF; use --output-dir for several")

    status = 0
    for pdf in args.pdfs:
        if not pdf.is_file():
            print(f"!  not a file: {pdf}", file=sys.stderr)
            status = 1
            continue
        target = args.output or (args.output_dir or pdf.parent) / f"{pdf.stem}.md"
        if target.exists() and not args.overwrite:
            print(f"!  {target} exists; use --overwrite or -o", file=sys.stderr)
            status = 1
            continue
        images_name = f"{target.stem}_files"
        images_dir = None if args.no_images else target.parent / images_name
        try:
            result = convert_pdf(
                pdf, images_dir=images_dir, images_rel=images_name, title=args.title,
                add_title=not args.no_title, backend=args.backend,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"!  {pdf}: {exc}", file=sys.stderr)
            status = 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.markdown, encoding="utf-8")
        headings = sum(1 for ln in result.markdown.split("\n") if ln.startswith("#"))
        print(f"{pdf.name} → {target}  [{result.backend}]")
        print(f"    {result.title!r} · {result.pages} pages · {headings} headings · "
              f"{len(result.images)} images · {len(result.markdown):,} characters")
        for warning in result.warnings[:10]:
            print(f"    warning: {warning}")
    return status


if __name__ == "__main__":
    sys.exit(main())
