"""Preview or save whitespace-only Markdown paragraph repairs.

Examples:
  python3 scripts/preprocess_markdown.py inputs/
  python3 scripts/preprocess_markdown.py inputs/ --output-dir preprocessed/
  python3 scripts/preprocess_markdown.py inputs/ --output-dir preprocessed/ --overwrite
  python3 scripts/preprocess_markdown.py inputs/book.md --apply
  python3 scripts/preprocess_markdown.py inputs/book.md --ai-url http://127.0.0.1:1234/v1/chat/completions --ai-model local-model --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hymt.preprocess import openai_compatible_classifier, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="Markdown files or directories")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--apply", action="store_true",
                        help="update source files after saving .state backups")
    output.add_argument("--output-dir", type=Path,
                        help="save repaired copies in this directory, leaving sources untouched")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing copies in --output-dir")
    parser.add_argument("--ai-url", help="local OpenAI-compatible chat completions URL")
    parser.add_argument("--ai-model", help="model name exposed by the local endpoint")
    parser.add_argument("--ai-limit", type=int, default=128,
                        help="maximum uncertain boundaries sent to AI per file (default: 128)")
    parser.add_argument("--review-file", type=Path,
                        help="apply validated sampled AI boundary decisions from JSON")
    args = parser.parse_args()
    if bool(args.ai_url) != bool(args.ai_model):
        parser.error("--ai-url and --ai-model must be used together")
    if args.overwrite and not args.output_dir:
        parser.error("--overwrite requires --output-dir")
    ai_decide = (openai_compatible_classifier(args.ai_url, args.ai_model,
                                              max_boundaries=args.ai_limit)
                 if args.ai_url else None)
    review = json.loads(args.review_file.read_text(encoding="utf-8")) if args.review_file else {}
    if args.review_file and review.get("version") != 1:
        parser.error("unsupported review file version")
    reviewed_files = review.get("files", {})

    files: set[Path] = set()
    for path in args.paths:
        if path.is_dir():
            files.update(p for p in path.rglob("*.md") if p.is_file() and not p.name.startswith("."))
        elif path.is_file() and path.suffix.lower() == ".md":
            files.add(path)
        else:
            parser.error(f"not a Markdown file or directory: {path}")
    if args.output_dir:
        destination = args.output_dir.resolve()
        if any(destination == p.resolve() or destination in p.resolve().parents for p in files):
            parser.error("output directory cannot be a source file or its parent")
        if any(destination == p.resolve() or destination in p.resolve().parents
               for p in args.paths if p.is_dir()):
            parser.error("output directory must be outside the scanned source directory")

    planned = []
    for path in sorted(files):
        source = path.read_text(encoding="utf-8")
        reviewed = reviewed_files.get(path.name, {})
        decisions = reviewed.get("decisions", {})
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if decisions and reviewed.get("source_sha256") != source_sha:
            parser.error(f"review file source hash differs: {path}")

        def decide(boundaries):
            chosen = {}
            missing = set(decisions) - {str(b.id) for b in boundaries}
            if missing:
                parser.error(f"reviewed boundary IDs are no longer available: {path}")
            for boundary in boundaries:
                signature = hashlib.sha256(
                    (boundary.before + "\n" + boundary.after).encode("utf-8")
                ).hexdigest()
                record = decisions.get(str(boundary.id))
                if record:
                    if record.get("signature") != signature:
                        parser.error(f"reviewed boundary has changed: {path} #{boundary.id}")
                    chosen[boundary.id] = record["action"]
            if ai_decide:
                remaining = [b for b in boundaries if b.id not in chosen]
                chosen.update(ai_decide(remaining))
            return chosen

        result = prepare(source, decide if decisions or ai_decide else None)
        print(f"{path}: split={result.split}, joined={result.joined}, "
              f"uncertain={result.uncertain}, ai={result.ai_decisions}")
        planned.append((path, source, result.text))
    if args.apply:
        for path, source, prepared in planned:
            if prepared == source:
                continue
            try:
                relative = path.resolve().relative_to(ROOT)
            except ValueError:
                suffix = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
                relative = Path(f"{path.stem}-{suffix}{path.suffix}")
            backup = ROOT / ".state" / "preprocess-backups" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(prepared, encoding="utf-8")
            print(f"  updated; original: {backup}")
    elif args.output_dir:
        names = [path.name for path, _, _ in planned]
        if len(names) != len(set(names)):
            parser.error("source files share a name; use separate output directories")
        for path, _, prepared in planned:
            target = destination / path.name
            if target.exists() and target.read_text(encoding="utf-8") != prepared and not args.overwrite:
                parser.error(f"output already exists with different contents: {target}")
        destination.mkdir(parents=True, exist_ok=True)
        for path, _, prepared in planned:
            target = destination / path.name
            target.write_text(prepared, encoding="utf-8")
            print(f"  saved {target}")


if __name__ == "__main__":
    main()
