"""Convert an EPUB to a single Markdown file, with its images beside it.

Examples:
  python3 scripts/epub_to_markdown.py book.epub
  python3 scripts/epub_to_markdown.py book.epub -o "inputs/Book Title.md"
  python3 scripts/epub_to_markdown.py inputs/epubs/*.epub --output-dir inputs/
  python3 scripts/epub_to_markdown.py book.epub --no-images --no-title

The Markdown is written next to the EPUB unless -o or --output-dir says
otherwise. Images go to `<markdown stem>_files/` beside the Markdown file and
are referenced relatively, so the pair can be moved together. Only the Python
standard library is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hymt.epub2md import convert_epub


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("epubs", type=Path, nargs="+", help="EPUB file(s)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("-o", "--output", type=Path,
                       help="Markdown file to write (one EPUB only)")
    where.add_argument("--output-dir", type=Path,
                       help="directory for <epub stem>.md and its images")
    parser.add_argument("--title", help="book title for the top heading (default: EPUB metadata)")
    parser.add_argument("--no-title", action="store_true",
                        help="do not add a title heading and author line; keep heading levels as in the EPUB")
    parser.add_argument("--no-images", action="store_true", help="drop images instead of extracting them")
    parser.add_argument("--keep-nav", action="store_true",
                        help="keep the EPUB navigation document (table of contents) in the output")
    parser.add_argument("--include-nonlinear", action="store_true",
                        help='include spine items marked linear="no"')
    parser.add_argument("--overwrite", action="store_true", help="replace an existing Markdown file")
    args = parser.parse_args(argv)
    if args.output and len(args.epubs) != 1:
        parser.error("-o/--output takes exactly one EPUB; use --output-dir for several")

    status = 0
    for epub in args.epubs:
        if not epub.is_file():
            print(f"!  not a file: {epub}", file=sys.stderr)
            status = 1
            continue
        if args.output:
            target = args.output
        else:
            target = (args.output_dir or epub.parent) / f"{epub.stem}.md"
        if target.exists() and not args.overwrite:
            print(f"!  {target} exists; use --overwrite or -o", file=sys.stderr)
            status = 1
            continue
        images_name = f"{target.stem}_files"
        images_dir = None if args.no_images else target.parent / images_name
        try:
            result = convert_epub(
                epub, images_dir=images_dir, images_rel=images_name,
                title=args.title, add_title=not args.no_title,
                keep_nav=args.keep_nav, include_nonlinear=args.include_nonlinear,
            )
        except (ValueError, KeyError, OSError) as exc:
            print(f"!  {epub}: {exc}", file=sys.stderr)
            status = 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.markdown, encoding="utf-8")
        headings = sum(1 for ln in result.markdown.split("\n") if ln.startswith("#"))
        print(f"{epub.name} → {target}")
        print(f"    {result.title!r} · {result.documents} documents · {headings} headings · "
              f"{len(result.images)} images · {len(result.markdown):,} characters")
        for warning in result.warnings[:10]:
            print(f"    warning: {warning}")
        if len(result.warnings) > 10:
            print(f"    … and {len(result.warnings) - 10} more warnings")
    return status


if __name__ == "__main__":
    sys.exit(main())
