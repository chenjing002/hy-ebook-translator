"""The whole workflow: scan → translate → validate → save → publish."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .assemble import assemble, build_header
from .config import Config
from .api_engine import APIEngine
from .publish import PublishResult, publish as git_publish
from .site import write_site
from .state import slugify
from .translate import DocResult, Translator
from .validate import Report, validate


@dataclass
class DocOutcome:
    result: DocResult
    report: Report

    @property
    def ok(self) -> bool:
        return self.report.ok


@dataclass
class RunReport:
    outcomes: list[DocOutcome] = field(default_factory=list)
    site_files: list[Path] = field(default_factory=list)
    published: PublishResult | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(o.ok for o in self.outcomes)

    @property
    def translated(self) -> int:
        return sum(o.result.translated for o in self.outcomes)

    @property
    def cached(self) -> int:
        return sum(o.result.cached for o in self.outcomes)


def scan(cfg: Config, only: list[str] | None = None) -> list[Path]:
    """Markdown files waiting in `inputs/`, in a stable order."""
    if not cfg.inputs_dir.exists():
        return []
    files = sorted(
        p for p in cfg.inputs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".markdown"}
        and not p.name.startswith(".")
    )
    if not only:
        return files
    picked: list[Path] = []
    missing: list[str] = []
    for requested in only:
        name = Path(requested).name
        match = next((p for p in files if p.name == name or p.stem == name), None)
        if match is None:
            missing.append(requested)
        elif match not in picked:
            picked.append(match)
    if missing:
        raise SystemExit(f"no input matched: {', '.join(missing)}")
    return picked


def translate_one(cfg: Config, engine: APIEngine, src: Path, *, force: bool = False,
                  quiet: bool = False) -> DocOutcome:
    if force:
        (cfg.state_dir / f"{slugify(src.stem)}.json").unlink(missing_ok=True)

    translator = Translator(cfg, engine, quiet=quiet)
    result = translator.translate_document(src)

    stats = {
        "units": len(result.zh), "translated": result.translated,
        "cached": result.cached, "skipped": result.skipped,
    }
    header = build_header(result.title, src.name, cfg, stats)
    built = assemble(result.blocks, result.zh, cfg, header=header)

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    result.output.write_text(built.text, encoding="utf-8")

    report = validate(src, result.output, cfg, result.warnings)
    report.stats["pairs"] = built.pairs
    report.stats["verbatim"] = built.passthrough
    return DocOutcome(result=result, report=report)


def run(cfg: Config, engine: APIEngine, *, only: list[str] | None = None,
        force: bool = False, do_publish: bool = True, push: bool = True,
        dry_run: bool = False, quiet: bool = False, strict: bool = False) -> RunReport:
    started = time.monotonic()
    report = RunReport()
    sources = scan(cfg, only)
    if not sources:
        return report

    for src in sources:
        outcome = translate_one(cfg, engine, src, force=force, quiet=quiet)
        report.outcomes.append(outcome)

    report.site_files = write_site(cfg)

    if do_publish and (not strict or report.ok):
        names = [o.result.output.name for o in report.outcomes]
        report.published = git_publish(cfg, names, push=push, dry_run=dry_run)

    report.seconds = time.monotonic() - started
    return report
