"""Command line interface.

    python3 -m hymt run                scan → translate → validate → save → publish
    python3 -m hymt translate FILE     one document, no publishing
    python3 -m hymt validate           re-check the documents already generated
    python3 -m hymt site               rebuild library.json
    python3 -m hymt publish            commit and push
    python3 -m hymt serve              preview the site locally
    python3 -m hymt status             what is configured and what is done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .api_engine import APIEngine
from .config import load_config
from .pipeline import DocOutcome, run, scan, translate_one
from .publish import enable_pages, is_repo, pending_changes, publish as git_publish
from .site import build_library, write_site
from .state import DocState, file_sha, slugify
from .validate import format_report, validate

BULLET = "  •"


# ── helpers ──────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hy",
        description="Markdown preprocessing, API translation and bilingual publishing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    p.add_argument("--version", action="version", version=f"hy-ebook-translator {__version__}")
    p.add_argument("--config", type=Path, help="additional TOML settings file")
    p.add_argument("--api-url", help="override [api].url")
    p.add_argument("--api-model", help="override [api].model")
    p.add_argument("--fallback-model", action="append", help="model to use when the prior model reaches its cap")
    p.add_argument("--api-key-env", help="override [api].key_env")
    p.add_argument("--max-total-tokens", type=int, help="cap this project's recorded API token use")
    p.add_argument("--input-dir", help="override [paths].inputs, e.g. preprocessed")
    p.add_argument("--output-dir", help="override [paths].outputs, e.g. output")
    p.add_argument("--target-lang", help="override [translate].target_lang")
    p.add_argument("--source-lang", help="override [translate].source_lang")
    p.add_argument("--order", choices=["zh-first", "en-first"], help="override [layout].order")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("run", help="the whole workflow")
    s.add_argument("only", nargs="*", help="limit to these input files")
    s.add_argument("--force", action="store_true", help="ignore cached translations")
    s.add_argument("--no-publish", action="store_true", help="do not touch git")
    s.add_argument("--no-push", action="store_true", help="commit but do not push")
    s.add_argument("--dry-run", action="store_true", help="show what would be committed")
    s.add_argument("--strict", action="store_true", help="fail if validation finds errors")
    s.add_argument("--verbose", action="store_true", help="list every validation warning")

    s = sub.add_parser("translate", help="translate documents without publishing")
    s.add_argument("only", nargs="*", help="limit to these input files")
    s.add_argument("--force", action="store_true", help="ignore cached translations")
    s.add_argument("--verbose", action="store_true")

    sub.add_parser("scan", help="list inputs and their state")

    s = sub.add_parser("validate", help="re-check generated documents")
    s.add_argument("only", nargs="*")
    s.add_argument("--strict", action="store_true")
    s.add_argument("--verbose", action="store_true")

    sub.add_parser("site", help="rebuild library.json")

    s = sub.add_parser("publish", help="commit and push the library")
    s.add_argument("--no-push", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--pages", action="store_true", help="also enable GitHub Pages")

    s = sub.add_parser("serve", help="serve the site on localhost")
    s.add_argument("-p", "--port", type=int, default=8000)

    sub.add_parser("status", help="configuration and progress")

    s = sub.add_parser("clean", help="discard cached translation state")
    s.add_argument("only", nargs="*")
    return p


def config_from(args: argparse.Namespace):
    return load_config(extra_path=args.config, overrides={
        "api.url": args.api_url,
        "api.model": args.api_model,
        "api.fallback_models": args.fallback_model,
        "api.key_env": args.api_key_env,
        "api.max_total_tokens": args.max_total_tokens,
        "paths.inputs": args.input_dir,
        "paths.outputs": args.output_dir,
        "translate.target_lang": args.target_lang,
        "translate.source_lang": args.source_lang,
        "layout.order": args.order,
    })


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in {"B", "KB"} else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.0f} B"


def input_label(cfg) -> Path:
    try:
        return cfg.inputs_dir.relative_to(cfg.root)
    except ValueError:
        return cfg.inputs_dir


def hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def report_outcomes(outcomes: list[DocOutcome], verbose: bool) -> None:
    for o in outcomes:
        print(format_report(o.report, verbose=verbose))
        r = o.result
        print(f"    {r.translated} translated, {r.cached} reused, {r.skipped} verbatim"
              f"  ·  {r.output.relative_to(r.output.parent.parent)}"
              f"  ·  {hms(r.seconds)}")


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_scan(args, cfg) -> int:
    files = scan(cfg)
    if not files:
        print(f"No Markdown in {input_label(cfg)}/ — copy some .md files there.")
        return 0
    print(f"{len(files)} input file(s) in {input_label(cfg)}/\n")
    fp = cfg.fingerprint()
    for src in files:
        out = cfg.outputs_dir / src.name
        state = DocState.load(cfg.state_dir, slugify(src.stem), file_sha(src), fp)
        done = len(state.segments)
        if not out.exists():
            mark, note = "new ", "not translated yet"
        elif state.source_sha != file_sha(src):
            mark, note = "edit", f"source changed · {done} segments cached"
        else:
            mark, note = "done", f"{done} segments cached"
        print(f"  [{mark}] {src.name:<44} {human(src.stat().st_size):>9}  {note}")
    return 0


def cmd_translate(args, cfg) -> int:
    files = scan(cfg, args.only)
    if not files:
        print("Nothing to translate.")
        return 0
    engine = APIEngine(cfg)
    engine.load()
    outcomes: list[DocOutcome] = []
    print(f"Translating {len(files)} document(s) → {cfg.translate.target_lang}\n")
    for src in files:
        outcome = translate_one(cfg, engine, src, force=args.force, quiet=cfg_quiet(args))
        outcomes.append(outcome)
        report_outcomes([outcome], args.verbose)
    if cfg.output_dir == "outputs":
        write_site(cfg)
    print(f"{engine.calls} API requests · {engine.prompt_tokens:,} input + "
          f"{engine.output_tokens:,} output tokens this run")
    return 0 if all(o.ok for o in outcomes) else 1


def cmd_run(args, cfg) -> int:
    files = scan(cfg, args.only)
    if not files:
        print(f"Nothing in {input_label(cfg)}/ to translate.")
        return 0
    engine = APIEngine(cfg)
    engine.load()
    print(f"Translating {len(files)} document(s) with API model {cfg.api.model}")
    print(f"{cfg.translate.source_lang} → {cfg.translate.target_lang}, "
          f"{cfg.layout.order} bilingual layout\n")

    report = run(
        cfg, engine,
        only=args.only, force=args.force,
        do_publish=not args.no_publish, push=not args.no_push,
        dry_run=args.dry_run, quiet=cfg_quiet(args), strict=args.strict,
    )
    print()
    report_outcomes(report.outcomes, args.verbose)

    print(f"\n{len(report.outcomes)} document(s) in {hms(report.seconds)} · "
          f"{report.translated} segments translated, {report.cached} reused")
    if engine.loaded:
        print(f"    {engine.calls} API requests, "
              f"{engine.prompt_tokens:,} prompt + {engine.output_tokens:,} output tokens")

    pub = report.published
    if pub:
        for message in pub.messages:
            print(f"    {message}")
        if pub.committed:
            print(f"    committed {pub.commit} ({len(pub.changed)} files)")
        if pub.pushed:
            print(f"    pushed to {pub.remote} [{pub.branch}]")
            print(f"    site: {pub.pages_url}")

    errors = sum(len(o.report.errors) for o in report.outcomes)
    if errors and args.strict:
        print(f"\n{errors} validation error(s); --strict so exiting non-zero", file=sys.stderr)
        return 1
    return 0


def cmd_validate(args, cfg) -> int:
    files = scan(cfg, args.only)
    if not files:
        print("No inputs to compare against.")
        return 0
    bad = 0
    fp = cfg.fingerprint()
    for src in files:
        out = cfg.outputs_dir / src.name
        state = DocState.load(cfg.state_dir, slugify(src.stem), file_sha(src), fp)
        report = validate(src, out, cfg, state.warnings)
        print(format_report(report, verbose=args.verbose))
        bad += len(report.errors)
    return 1 if bad and args.strict else 0


def cmd_site(args, cfg) -> int:
    written = write_site(cfg)
    library = build_library(cfg)
    print(f"library.json · {len(library['documents'])} document(s)")
    for doc in library["documents"]:
        print(f"{BULLET} {doc['title_zh']}  ({doc['cjk_chars']:,} 汉字, {doc['minutes']} min)")
    for path in written:
        print(f"    wrote {path.relative_to(cfg.root)}")
    return 0


def cmd_publish(args, cfg) -> int:
    write_site(cfg)
    docs = [p.name for p in sorted(cfg.outputs_dir.glob("*.md"))]
    if not docs:
        print(f"{cfg.output_dir}/ is empty — nothing to publish.")
        return 0
    result = git_publish(cfg, docs, push=not args.no_push, dry_run=args.dry_run)
    for message in result.messages:
        print(f"{BULLET} {message}")
    if result.changed and args.dry_run:
        for path in result.changed[:40]:
            print(f"    {path}")
    if result.committed:
        print(f"{BULLET} committed {result.commit} ({len(result.changed)} files)")
    if result.pushed:
        print(f"{BULLET} pushed to {result.remote} [{result.branch}]")
    if args.pages:
        ok, message = enable_pages(cfg)
        print(f"{BULLET} {message}")
    if result.pushed or result.pages_url:
        owner, _, name = cfg.publish.repo.partition("/")
        print(f"{BULLET} site: https://{owner}.github.io/{name}/")
    return 0 if (result.pushed or args.no_push or args.dry_run or not result.changed) else 1


def cmd_serve(args, cfg) -> int:
    import http.server
    import socketserver

    write_site(cfg)
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(cfg.root), **kw)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Reading site on http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def cmd_status(args, cfg) -> int:
    print(f"hy-ebook-translator {__version__}\n")
    print(f"  api          {cfg.api.url or 'not configured'}")
    print(f"  api model    {cfg.api.model or 'not configured'}")
    print(f"  direction    {cfg.translate.source_lang} → {cfg.translate.target_lang}")
    print(f"  layout       {cfg.layout.order}, headings {cfg.layout.heading_style}")
    print(f"  inputs       {len(scan(cfg))} file(s)")
    print(f"  outputs      {len(list(cfg.outputs_dir.glob('*.md')))} file(s)")
    print(f"  repository   {cfg.publish.repo} [{cfg.publish.branch}]"
          f"{'' if is_repo(cfg) else '  (not initialised)'}")
    if is_repo(cfg):
        changes = pending_changes(cfg)
        print(f"  git          {len(changes)} uncommitted path(s)")
    return 0


def cmd_clean(args, cfg) -> int:
    targets = scan(cfg, args.only) or []
    if args.only and targets:
        removed = 0
        for src in targets:
            path = cfg.state_dir / f"{slugify(src.stem)}.json"
            if path.exists():
                path.unlink()
                removed += 1
        print(f"removed {removed} state file(s)")
        return 0
    if cfg.state_dir.exists():
        caches = [p for p in cfg.state_dir.glob("*.json") if p.name != "api-usage.json"]
        count = len(caches)
        for path in caches:
            path.unlink()
        print(f"removed {count} translation state file(s); API usage and documents untouched")
    return 0


def cfg_quiet(args) -> bool:
    return bool(getattr(args, "quiet", False))


COMMANDS = {
    "run": cmd_run, "translate": cmd_translate, "scan": cmd_scan,
    "validate": cmd_validate, "site": cmd_site, "publish": cmd_publish,
    "serve": cmd_serve, "status": cmd_status, "clean": cmd_clean,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    cfg = config_from(args)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        return COMMANDS[args.command](args, cfg)
    except (RuntimeError, ValueError) as exc:
        print(f"!  {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — progress is saved; re-run to continue", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
