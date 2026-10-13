"""Add paragraph breaks to a PDF-converted Markdown book without changing words.

Only long prose blocks and the line-oriented index are touched. HTML tables,
math, code, headings and images remain byte-for-byte intact. A copy of the
original is saved under .state/ before the file is changed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hymt.mdparse import Block, parse_markdown
from hymt.translate import SENTENCE_RE


def prepare(text: str, target_chars: int = 1200) -> tuple[str, int]:
    blocks = parse_markdown(text)
    out: list[str] = []
    cursor = 0
    changed = 0
    for i, block in enumerate(blocks):
        at = text.find(block.raw, cursor)
        if at < 0:
            raise ValueError(f"could not locate {block.kind} block at {cursor}")
        out.append(text[cursor:at])
        replacement = block.raw
        if block.kind == "paragraph":
            if i and _is_index_heading(blocks[i - 1]):
                replacement = "\n\n".join(block.raw.split("\n"))
            elif len(block.raw) > target_chars * 2:
                replacement = _break_prose(block.raw, target_chars)
        if replacement != block.raw:
            changed += 1
        out.append(replacement)
        cursor = at + len(block.raw)
    out.append(text[cursor:])
    prepared = "".join(out)
    if _words(prepared) != _words(text):
        raise ValueError("preprocessing changed non-whitespace content")
    return prepared, changed


def _is_index_heading(block: Block) -> bool:
    return block.kind == "heading" and block.meta.get("text", "").strip().lower() == "index"


def _break_prose(raw: str, target_chars: int) -> str:
    out: list[str] = []
    cursor = last_break = 0
    for match in SENTENCE_RE.finditer(raw):
        if match.start() - last_break < target_chars:
            continue
        out.append(raw[cursor:match.start()])
        out.append("\n\n")
        cursor = match.end()
        last_break = cursor
    out.append(raw[cursor:])
    return "".join(out)


def _words(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--target-chars", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = args.path.resolve()
    if not path.is_file():
        parser.error(f"missing input: {path}")
    original = path.read_text(encoding="utf-8")
    prepared, changed = prepare(original, args.target_chars)
    print(f"{changed} long paragraphs or index blocks prepared")
    if args.dry_run or prepared == original:
        return
    backup = ROOT / ".state" / f"{path.name}.before-paragraph-breaks"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(prepared, encoding="utf-8")
    print(f"updated {path}; original saved to {backup}")


if __name__ == "__main__":
    main()
