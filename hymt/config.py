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
class ModelConfig:
    repo: str = "mlx-community/Hy-MT2-7B-8bit"
    temperature: float = 0.7
    top_p: float = 0.6
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_new_tokens: int = 3072


@dataclass
class TranslateConfig:
    source_lang: str = "English"
    target_lang: str = "Simplified Chinese"
    prompt_lang: str = "zh"
    batch_token_budget: int = 640
    max_batch_units: int = 10
    max_unit_tokens: int = 480
    context_units: int = 1
    max_retries: int = 2


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
    model: ModelConfig = field(default_factory=ModelConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    root: Path = ROOT

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

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

        payload = {
            "model": _as_dict(self.model),
            "translate": _as_dict(self.translate),
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


def load_config(root: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config(root=root or ROOT)
    for name in ("config.toml", "config.local.toml"):
        path = cfg.root / name
        if not path.exists():
            continue
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        for section in ("model", "translate", "layout", "publish"):
            if section in data:
                _overlay(getattr(cfg, section), data[section], section)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        _overlay(getattr(cfg, section), {key: value}, section)
    return cfg
