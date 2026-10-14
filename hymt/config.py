"""Configuration loading.

`config.toml` holds the defaults that ship with the repo; `config.local.toml`
(git-ignored) overlays machine-specific overrides. CLI flags win over both.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class APIConfig:
    url: str = ""
    model: str = ""
    fallback_models: list[str] = field(default_factory=list)
    key_env: str = "TRANSLATION_API_KEY"
    timeout_seconds: int = 120
    temperature: float = 0.0
    max_output_tokens: int = 3072
    max_total_tokens: int = 0  # per-model project-local cap; 0 disables it


@dataclass
class TranslateConfig:
    source_lang: str = "English"
    target_lang: str = "Simplified Chinese"
    batch_token_budget: int = 640
    max_batch_units: int = 10
    max_unit_tokens: int = 480
    context_units: int = 1
    max_retries: int = 2
    workers: int = 1


@dataclass
class LayoutConfig:
    order: str = "zh-first"
    heading_style: str = "italic-below"
    markers: bool = True


@dataclass
class PublishConfig:
    repo: str = "chenjing002/hy-ebook-translator"
    branch: str = "main"
    protocol: str = "ssh"
    site_title: str = "双语书架"
    site_subtitle: str = "Bilingual Library"
    include_inputs: bool = False


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    root: Path = ROOT
    input_dir: str = "inputs"
    output_dir: str = "outputs"
    # [titles]: shelf labels per document file name, {"zh": …, "en": …}. A
    # converted book often opens with front matter or a split title line, so
    # the heading found in the file is not always the title of the book.
    titles: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def inputs_dir(self) -> Path:
        return self.root / self.input_dir

    @property
    def outputs_dir(self) -> Path:
        return self.root / self.output_dir

    @property
    def state_dir(self) -> Path:
        return self.root / ".state"

    def fingerprint(self) -> str:
        """Identity of every setting that can change a translation's text.

        Cached segments are invalidated when this changes, but cosmetic layout
        settings are deliberately excluded: re-laying-out a document must not
        force a re-translation.
        """
        import hashlib
        import json

        api_text_settings = {
            "url": self.api.url,
            "model": self.api.model,
            "fallback_models": self.api.fallback_models,
            "temperature": self.api.temperature,
        }
        payload = {
            "api": api_text_settings,
            "translate": {
                "source_lang": self.translate.source_lang,
                "target_lang": self.translate.target_lang,
                "max_unit_tokens": self.translate.max_unit_tokens,
                "context_units": self.translate.context_units,
            },
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _as_dict(obj: Any) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _overlay(section: Any, data: dict[str, Any], where: str) -> None:
    known = {f.name: f.type for f in fields(section)}
    for key, value in data.items():
        if key not in known:
            raise SystemExit(f"config error: unknown key [{where}].{key}")
        setattr(section, key, value)


def _overlay_titles(cfg: Config, data: dict[str, Any]) -> None:
    """Merge a [titles] table, one document at a time."""
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise SystemExit(
                f'config error: [titles]."{name}" must be a table of zh/en titles'
            )
        unknown = set(entry) - {"zh", "en"}
        if unknown:
            raise SystemExit(
                f'config error: unknown key [titles]."{name}".{sorted(unknown)[0]}'
            )
        cfg.titles.setdefault(name, {}).update(
            {k: str(v).strip() for k, v in entry.items()}
        )


def load_config(root: Path | None = None, overrides: dict[str, Any] | None = None,
                extra_path: Path | None = None) -> Config:
    cfg = Config(root=root or ROOT)
    paths = [cfg.root / "config.toml", cfg.root / "config.local.toml"]
    if extra_path is not None:
        paths.append(extra_path if extra_path.is_absolute() else cfg.root / extra_path)
    for path in paths:
        if not path.exists():
            if extra_path is not None and path == paths[-1]:
                raise SystemExit(f"config error: missing file {path}")
            continue
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        for section in ("api", "translate", "layout", "publish"):
            if section in data:
                _overlay(getattr(cfg, section), data[section], section)
        if "paths" in data:
            for key in data["paths"]:
                if key not in {"inputs", "outputs"}:
                    raise SystemExit(f"config error: unknown key [paths].{key}")
            cfg.input_dir = data["paths"].get("inputs", cfg.input_dir)
            cfg.output_dir = data["paths"].get("outputs", cfg.output_dir)
        if "titles" in data:
            _overlay_titles(cfg, data["titles"])

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        if section == "paths" and key == "inputs":
            cfg.input_dir = value
        elif section == "paths" and key == "outputs":
            cfg.output_dir = value
        else:
            _overlay(getattr(cfg, section), {key: value}, section)
    return cfg
