"""Command line interface.

    ./hy setup              download the model weights (resumable)
    ./hy run                scan → translate → validate → save → publish
    ./hy translate FILE     one document, no publishing
    ./hy validate           re-check what is already in outputs/
    ./hy site               rebuild library.json
    ./hy publish            commit and push
    ./hy serve              preview the site locally
    ./hy status             what is configured and what is done
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__, download as dl
from .config import load_config
from .engine import Engine, model_info, preflight, total_memory_gb
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
        description="Local Markdown translation and publishing pipeline (Hy-MT2 on Apple MLX).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    p.add_argument("--version", action="version", version=f"hy-ebook-translator {__version__}")
    p.add_argument("--model", help="override [model].repo")
    p.add_argument("--target-lang", help="override [translate].target_lang")
    p.add_argument("--source-lang", help="override [translate].source_lang")
    p.add_argument("--order", choices=["zh-first", "en-first"], help="override [layout].order")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("setup", help="download model weights and check this machine")
    s.add_argument("--repo", help="a specific weight repository to fetch")
    s.add_argument("--endpoint", help="pin a download host, e.g. https://hf-mirror.com")

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

    sub.add_parser("status", help="configuration, weights and progress")

    s = sub.add_parser("clean", help="discard cached translation state")
    s.add_argument("only", nargs="*")
    return p


def config_from(args: argparse.Namespace):
    return load_config(overrides={
        "model.repo": args.model,
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


def hms(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def warn_preflight(cfg) -> None:
    for note in preflight(cfg):
        print(f"!  {note}", file=sys.stderr)


def report_outcomes(outcomes: list[DocOutcome], verbose: bool) -> None:
    for o in outcomes:
        print(format_report(o.report, verbose=verbose))
        r = o.result
        print(f"    {r.translated} translated, {r.cached} reused, {r.skipped} verbatim"
              f"  ·  {r.output.relative_to(r.output.parent.parent)}"
              f"  ·  {hms(r.seconds)}")


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_setup(args, cfg) -> int:
    import os

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
    repo = args.repo or cfg.model.repo
    print(f"hy-ebook-translator {__version__}")
    print(f"{BULLET} python  {sys.version.split()[0]}")
    try:
        import mlx.core as mx
        import mlx_lm
        print(f"{BULLET} mlx-lm  {mlx_lm.__version__}  (Metal: {mx.metal.is_available()})")
    except Exception as exc:
        print(f"!  mlx-lm is not importable: {exc}", file=sys.stderr)
        print("   pip install -r requirements.txt", file=sys.stderr)
        return 1
    print(f"{BULLET} memory  {total_memory_gb():.0f} GB unified")
    warn_preflight(cfg)

    if dl.is_complete(cfg.root, repo):
        print(f"{BULLET} weights already complete: {dl.local_dir(cfg.root, repo)}")
        return 0
    try:
        remaining = dl.missing_bytes(cfg.root, repo)
        print(f"{BULLET} {human(remaining)} still to download for {repo}")
    except Exception:
        pass
    try:
        dl.fetch(cfg.root, repo)
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"\n!  {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_scan(args, cfg) -> int:
    files = scan(cfg)
    if not files:
        print(f"No Markdown in {cfg.inputs_dir.relative_to(cfg.root)}/ — copy some .md files there.")
        return 0
    print(f"{len(files)} input file(s) in {cfg.inputs_dir.relative_to(cfg.root)}/\n")
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
    warn_preflight(cfg)
    engine = Engine(cfg)
    outcomes: list[DocOutcome] = []
    print(f"Translating {len(files)} document(s) → {cfg.translate.target_lang}\n")
    for src in files:
        outcomes.append(translate_one(cfg, engine, src, force=args.force, quiet=cfg_quiet(args)))
    report_outcomes(outcomes, args.verbose)
    write_site(cfg)
    return 0 if all(o.ok for o in outcomes) else 1


def cmd_run(args, cfg) -> int:
    files = scan(cfg, args.only)
    if not files:
        print(f"Nothing in {cfg.inputs_dir.relative_to(cfg.root)}/ to translate.")
        return 0
    warn_preflight(cfg)
    engine = Engine(cfg)
    print(f"Translating {len(files)} document(s) with {cfg.model.repo}")
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
        print(f"    {engine.calls} model calls, "
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
        print("outputs/ is empty — nothing to publish.")
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
    print(f"  model        {cfg.model.repo}")
    local = dl.local_dir(cfg.root, cfg.model.repo)
    if local.exists():
        have = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        try:
            missing = dl.missing_bytes(cfg.root, cfg.model.repo)
        except Exception:
            missing = 0
        state = "complete" if missing == 0 else f"{human(missing)} remaining"
        print(f"  weights      {local.relative_to(cfg.root)}  ({human(have)}, {state})")
    elif model_info(cfg.model.repo).downloaded:
        print("  weights      in the HuggingFace cache")
    else:
        print("  weights      not downloaded — run ./hy setup")
    print(f"  direction    {cfg.translate.source_lang} → {cfg.translate.target_lang}"
          f"  (prompt in {cfg.translate.prompt_lang})")
    print(f"  layout       {cfg.layout.order}, headings {cfg.layout.heading_style}")
    print(f"  memory       {total_memory_gb():.0f} GB unified")
    print(f"  inputs       {len(scan(cfg))} file(s)")
    print(f"  outputs      {len(list(cfg.outputs_dir.glob('*.md')))} file(s)")
    print(f"  repository   {cfg.publish.repo} [{cfg.publish.branch}]"
          f"{'' if is_repo(cfg) else '  (not initialised)'}")
    if is_repo(cfg):
        changes = pending_changes(cfg)
        print(f"  git          {len(changes)} uncommitted path(s)")
    warn_preflight(cfg)
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
        count = len(list(cfg.state_dir.glob("*.json")))
        for path in cfg.state_dir.glob("*.json"):
            path.unlink()
        print(f"removed {count} state file(s); outputs/ untouched")
    return 0


def cfg_quiet(args) -> bool:
    return bool(getattr(args, "quiet", False))


COMMANDS = {
    "setup": cmd_setup, "run": cmd_run, "translate": cmd_translate, "scan": cmd_scan,
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
    except KeyboardInterrupt:
        print("\ninterrupted — progress is saved; re-run to continue", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
