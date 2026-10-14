"""Configurable OpenAI-compatible translation API adapter.

No endpoint or model is selected by default. The adapter makes requests only
after both are configured; credentials are read from an environment variable.
"""

from __future__ import annotations

import json
import http.client
import math
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import Config


class APIEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.loaded = False
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.active_model = cfg.api.model
        self.models_used: set[str] = set()
        self._usage_path = cfg.state_dir / "api-usage.json"
        self._usage = self._read_prior_usage()
        self._reserved: dict[str, int] = {}
        self._lock = threading.Lock()

    def _read_prior_usage(self) -> dict[str, int]:
        try:
            data = json.loads(self._usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if data.get("url") != self.cfg.api.url:
            return {}
        if isinstance(data.get("models"), dict):
            return {str(k): int(v) for k, v in data["models"].items()}
        if data.get("model"):
            return {data["model"]: int(data.get("total_tokens") or 0)}
        return {}

    def _save_usage(self) -> None:
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"url": self.cfg.api.url, "models": self._usage}
        fd, name = tempfile.mkstemp(dir=self._usage_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(name, self._usage_path)
        except BaseException:
            os.unlink(name)
            raise

    def load(self, verbose: bool = False) -> None:
        api = self.cfg.api
        if not api.url or not api.model:
            raise RuntimeError("translation API is not configured; set [api].url and [api].model")
        parsed = urllib.parse.urlsplit(api.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("[api].url must be an HTTP(S) chat completions endpoint")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("remote translation API endpoints must use HTTPS")
        if api.key_env and not os.environ.get(api.key_env) and parsed.hostname not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise RuntimeError(f"translation API key environment variable is unset: {api.key_env}")
        self.loaded = True

    def count_tokens(self, text: str) -> int:
        # Deliberately conservative estimate until a provider tokenizer is
        # selected. Actual usage is read from the response when available.
        return max(1, math.ceil(sum(1 if ord(ch) > 127 else 0.3 for ch in text)))

    def build_prompt(self, text: str, *, delimiters: bool = False,
                     background: str = "") -> str:
        direction = f"{self.cfg.translate.source_lang} to {self.cfg.translate.target_lang}"
        instruction = (
            f"Translate the text from {direction}. Return only the translation. "
            "Preserve Markdown syntax, numbers, URLs, and all placeholders exactly."
        )
        if delimiters:
            instruction += " Preserve every numbered segment marker and its order exactly."
        if background:
            instruction += f"\nPrevious translated context (do not translate again):\n{background}"
        return f"{instruction}\n\nText to translate:\n{text}"

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.load()
        api = self.cfg.api
        budget = min(max_new_tokens or api.max_output_tokens, api.max_output_tokens)
        reserve = self.count_tokens(prompt) + budget
        with self._lock:
            models = [api.model, *api.fallback_models]
            if api.max_total_tokens:
                available = [m for m in models if
                             self._usage.get(m, 0) + self._reserved.get(m, 0) + reserve
                             <= api.max_total_tokens]
                if not available:
                    raise RuntimeError(
                        f"project API token budget reached for all models ({api.max_total_tokens:,} each); "
                        "resume after increasing --max-total-tokens"
                    )
                model = available[0]
            else:
                model = api.model
            self._reserved[model] = self._reserved.get(model, 0) + reserve
            self.active_model = model
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": api.temperature,
            "max_tokens": budget,
        }
        if model.startswith("hy-mt2-"):
            body["enable_thinking"] = False
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api.key_env and os.environ.get(api.key_env):
            headers["Authorization"] = f"Bearer {os.environ[api.key_env]}"
        request = urllib.request.Request(api.url, data=payload, headers=headers)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, redirect_headers, newurl):
                raise RuntimeError("translation API redirected; request was stopped")

        try:
            for attempt in range(8):
                try:
                    with urllib.request.build_opener(NoRedirect).open(
                        request, timeout=api.timeout_seconds
                    ) as response:
                        data = json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code not in {429, 500, 502, 503, 504} or attempt == 7:
                        raise RuntimeError(f"translation API returned HTTP {exc.code}") from exc
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    if exc.code == 429:
                        delay = max(delay, min(15 * (attempt + 1), 90))
                    else:
                        delay = max(delay, min(2 ** attempt, 30))
                    time.sleep(min(delay, 120))
                except urllib.error.URLError as exc:
                    if attempt == 7:
                        raise RuntimeError(f"translation API request failed: {exc.reason}") from exc
                    time.sleep(min(2 ** attempt, 30))
                except (http.client.HTTPException, TimeoutError, ConnectionError,
                        json.JSONDecodeError) as exc:
                    if attempt == 7:
                        raise RuntimeError("translation API response was interrupted after retries") from exc
                    time.sleep(min(2 ** attempt, 30))
            try:
                output = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("translation API response has no message content") from exc
            if not isinstance(output, str):
                raise RuntimeError("translation API returned non-text content")
            usage = data.get("usage") or {}
            prompt_used = usage.get("prompt_tokens") or self.count_tokens(prompt)
            output_used = usage.get("completion_tokens") or self.count_tokens(output)
            with self._lock:
                self.calls += 1
                self.prompt_tokens += prompt_used
                self.output_tokens += output_used
                self._usage[model] = self._usage.get(model, 0) + prompt_used + output_used
                self.models_used.add(model)
                if api.max_total_tokens:
                    self._save_usage()
            return output
        finally:
            with self._lock:
                self._reserved[model] -= reserve
