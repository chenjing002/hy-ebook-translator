"""Local MLX inference wrapper around Hy-MT2.

All generation happens on-device through `mlx-lm`. The prompt templates are the
ones published with the upstream Tencent Hunyuan release: the plain translation
template, the delimiter-preserving template (used whenever a request carries
segment markers or protected placeholders) and the background-context template.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import download as dl
from .config import Config

# Upstream templates. `{target}` is filled with a Chinese language name for the
# Chinese prompt and an English one for the English prompt.
TEMPLATES = {
    "zh": {
        "plain": "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}",
        "delim": (
            "请将以下文本准确翻译为{target}。\n"
            "你必须在译文中保留等量的分隔符，绝对不可遗漏、转义或翻译该符号，并注意分隔符的位置。\n\n"
            "{text}"
        ),
        "context": (
            "【背景信息】\n{background}\n\n"
            "请结合背景信息将以下文本准确翻译为{target}。\n"
            "你必须在译文中保留等量的分隔符，绝对不可遗漏、转义或翻译该符号，并注意分隔符的位置。\n\n"
            "【待翻译文本】\n{text}"
        ),
    },
    "en": {
        "plain": (
            "Translate the following text into {target}. Note that you should only output "
            "the translated result without any additional explanation:\n\n{text}"
        ),
        "delim": (
            "Please accurately translate the following text into {target}.\n"
            "You must retain the exact same number of delimiters in the translation. Strictly do "
            "not omit, escape, or translate these symbols, and pay close attention to their "
            "placement.\n\n{text}"
        ),
        "context": (
            "[Background Information]\n{background}\n\n"
            "Please accurately translate the following text into {target}, taking the provided "
            "background information into consideration. You must retain the exact same number of "
            "delimiters in the translation.\n\n[Source Text]\n{text}"
        ),
    },
}

LANG_NAMES = {
    "Simplified Chinese": {"zh": "简体中文", "en": "Simplified Chinese"},
    "Traditional Chinese": {"zh": "繁体中文", "en": "Traditional Chinese"},
    "English": {"zh": "英语", "en": "English"},
    "Japanese": {"zh": "日语", "en": "Japanese"},
    "Korean": {"zh": "韩语", "en": "Korean"},
    "French": {"zh": "法语", "en": "French"},
    "German": {"zh": "德语", "en": "German"},
    "Spanish": {"zh": "西班牙语", "en": "Spanish"},
    "Russian": {"zh": "俄语", "en": "Russian"},
}


@dataclass
class ModelInfo:
    repo: str
    local_path: Path | None
    size_bytes: int
    downloaded: bool


def _hub_dir(repo: str) -> Path:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if cache.name != "huggingface":
        cache = cache / "huggingface"
    return cache / "hub" / ("models--" + repo.replace("/", "--"))


def model_info(repo: str) -> ModelInfo:
    d = _hub_dir(repo)
    if not d.exists():
        return ModelInfo(repo, None, 0, False)
    snapshots = sorted((d / "snapshots").glob("*")) if (d / "snapshots").exists() else []
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())
    weights = bool(snapshots) and any(snapshots[-1].glob("*.safetensors"))
    config = bool(snapshots) and (snapshots[-1] / "config.json").exists()
    return ModelInfo(repo, snapshots[-1] if snapshots else None, size, weights and config)


def total_memory_gb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        return int(out.stdout.strip()) / 1e9
    except Exception:
        return 0.0


def download(root: Path, repo: str, quiet: bool = False) -> Path:
    """Fetch weights into `models/<name>/`, resuming any partial file."""
    return dl.fetch(root, repo, quiet=quiet)


def preflight(cfg: Config) -> list[str]:
    """Non-fatal warnings about this machine's ability to run the chosen weights."""
    warnings: list[str] = []
    if sys.platform != "darwin":
        warnings.append("MLX requires macOS on Apple Silicon; this platform is unsupported.")
    ram = total_memory_gb()
    # bf16 7B needs roughly 15 GB resident, 8-bit ~8 GB, 4-bit ~4.5 GB.
    needed = 16.0 if cfg.model.repo.endswith(("7B", "7B-bf16")) else (
        8.5 if "8bit" in cfg.model.repo else 5.0
    )
    if ram and ram < needed + 2:
        alt = "mlx-community/Hy-MT2-7B-8bit" if "8bit" not in cfg.model.repo else "mlx-community/Hy-MT2-7B-4bit"
        warnings.append(
            f"{cfg.model.repo} needs ~{needed:.0f} GB resident but this Mac has "
            f"{ram:.0f} GB; expect heavy swapping. Consider [model].repo = \"{alt}\"."
        )
    free = shutil.disk_usage(cfg.root).free / 1e9
    if free < 6:
        warnings.append(f"Only {free:.1f} GB free on disk; weights may not fit.")
    return warnings


class Engine:
    """Lazily-loaded Hy-MT2 translator."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._sampler = None
        self._logits_processors: list = []
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    # ---- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, verbose: bool = True) -> None:
        if self._model is not None:
            return
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        repo = self.cfg.model.repo
        target = dl.resolve(self.cfg.root, repo)
        if target == repo and not model_info(repo).downloaded:
            download(self.cfg.root, repo, quiet=not verbose)
            target = dl.resolve(self.cfg.root, repo)
        elif target != repo and not dl.is_complete(self.cfg.root, repo):
            # A half-downloaded shard would otherwise surface as an opaque
            # safetensors parse error deep inside mlx-lm.
            local = dl.local_dir(self.cfg.root, repo)
            try:
                remaining = dl.missing_bytes(self.cfg.root, repo) / 1e9
                detail = f" — {remaining:.2f} GB still missing"
            except Exception:
                detail = ""
            raise SystemExit(
                f"weights in {local.name} are incomplete{detail}.\n"
                "Run `./hy setup` to resume the download, then try again."
            )
        if verbose:
            print(f"→ loading {self.cfg.model.repo} into unified memory …", flush=True)
        self._model, self._tokenizer = load(target)
        _register_stop_tokens(self._tokenizer, target)
        m = self.cfg.model
        self._sampler = make_sampler(temp=m.temperature, top_p=m.top_p, top_k=m.top_k)
        self._logits_processors = make_logits_processors(
            repetition_penalty=m.repetition_penalty or None
        )
        if verbose:
            print("  model ready", flush=True)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ---- prompting ---------------------------------------------------------
    def target_name(self) -> str:
        lang = self.cfg.translate.target_lang
        return LANG_NAMES.get(lang, {}).get(self.cfg.translate.prompt_lang, lang)

    def build_prompt(self, text: str, *, delimiters: bool = False, background: str = "") -> str:
        lang = self.cfg.translate.prompt_lang if self.cfg.translate.prompt_lang in TEMPLATES else "zh"
        tpl = TEMPLATES[lang]
        if background:
            body = tpl["context"].format(target=self.target_name(), background=background, text=text)
        elif delimiters:
            body = tpl["delim"].format(target=self.target_name(), text=text)
        else:
            body = tpl["plain"].format(target=self.target_name(), text=text)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False
        )

    # ---- generation --------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        from mlx_lm import generate

        self.load()
        budget = min(max_new_tokens or self.cfg.model.max_new_tokens, self.cfg.model.max_new_tokens)
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=budget,
            sampler=self._sampler,
            logits_processors=self._logits_processors,
            verbose=False,
        )
        self.calls += 1
        self.prompt_tokens += len(self.tokenizer.encode(prompt, add_special_tokens=False))
        self.output_tokens += len(self.tokenizer.encode(text, add_special_tokens=False))
        return clean_output(text)


def _register_stop_tokens(tokenizer, model_path: str) -> None:
    """Hy-MT2 stops on `<|eos|>` *or* `<|extra_5|>`.

    Only the first is declared in tokenizer_config.json, so mlx-lm would miss
    the second and keep generating to the token limit. config.json carries the
    full list; merge it in.
    """
    config = Path(model_path) / "config.json"
    if not config.exists():
        return
    try:
        declared = json.loads(config.read_text()).get("eos_token_id")
    except (json.JSONDecodeError, OSError):
        return
    ids = declared if isinstance(declared, list) else [declared]
    known = getattr(tokenizer, "eos_token_ids", None)
    merged = set(known or ()) | {int(i) for i in ids if isinstance(i, int)}
    if merged:
        tokenizer.eos_token_ids = merged


_STRIP_PREFIXES = (
    "翻译结果：", "翻译结果:", "译文：", "译文:", "翻译：", "翻译:",
    "Translation:", "Translated text:", "以下是翻译结果：",
)


def clean_output(text: str) -> str:
    """Trim the occasional conversational wrapper from a translation."""
    out = text.strip()
    for token in ("<|eos|>", "<|endoftext|>", "<|extra_0|>", "<|startoftext|>"):
        out = out.replace(token, "")
    out = out.strip()
    for prefix in _STRIP_PREFIXES:
        if out.startswith(prefix):
            out = out[len(prefix):].lstrip()
    # A whole-response code fence is never part of a translation.
    if out.startswith("```") and out.rstrip().endswith("```") and out.count("```") == 2:
        body = out.split("\n", 1)
        if len(body) == 2:
            out = body[1].rstrip()[:-3].rstrip()
    return out.strip()
