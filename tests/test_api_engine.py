"""API adapter tests run without a network connection."""

import io
import http.client
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hymt.api_engine import APIEngine
from hymt.config import load_config


class TestAPIEngine(unittest.TestCase):
    def test_missing_endpoint_fails_before_translation(self):
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            APIEngine(load_config()).load()

    def test_remote_http_is_rejected(self):
        cfg = load_config()
        cfg.api.url = "http://example.com/v1/chat/completions"
        cfg.api.model = "example"
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            APIEngine(cfg).load()

    def test_chat_request_and_usage(self):
        cfg = load_config()
        cfg.api.url = "http://127.0.0.1:1234/v1/chat/completions"
        cfg.api.model = "example"
        cfg.api.key_env = ""
        sent = []

        class Opener:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                data = {"choices": [{"message": {"content": "译文"}}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
                return io.BytesIO(json.dumps(data).encode())

        engine = APIEngine(cfg)
        prompt = engine.build_prompt("Original text", delimiters=True)
        with patch("hymt.api_engine.urllib.request.build_opener", return_value=Opener()):
            output = engine.generate(prompt, max_new_tokens=50)
        self.assertEqual(output, "译文")
        self.assertEqual(sent[0]["model"], "example")
        self.assertEqual(sent[0]["max_tokens"], 50)
        self.assertIn("Original text", sent[0]["messages"][0]["content"])
        self.assertEqual((engine.calls, engine.prompt_tokens, engine.output_tokens), (1, 12, 3))

    def test_api_identity_invalidates_translation_cache(self):
        cfg = load_config()
        before = cfg.fingerprint()
        cfg.api.model = "another-model"
        self.assertNotEqual(cfg.fingerprint(), before)

    def test_hy_mt2_disables_thinking_and_honors_budget(self):
        cfg = load_config()
        cfg.api.url = "http://127.0.0.1:1234/v1/chat/completions"
        cfg.api.model = "hy-mt2-pro"
        cfg.api.key_env = ""
        cfg.api.max_total_tokens = 1
        with self.assertRaisesRegex(RuntimeError, "budget reached"):
            APIEngine(cfg).generate("Translate this", max_new_tokens=10)
        cfg.api.max_total_tokens = 0
        sent = []

        class Opener:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                return io.BytesIO(json.dumps({"choices": [{"message": {"content": "译文"}}]}).encode())

        with patch("hymt.api_engine.urllib.request.build_opener", return_value=Opener()):
            APIEngine(cfg).generate("Translate this", max_new_tokens=10)
        self.assertIs(sent[0]["enable_thinking"], False)

    def test_falls_back_to_next_model_without_losing_usage_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = load_config(root=Path(folder))
            cfg.api.url = "http://127.0.0.1:1234/v1/chat/completions"
            cfg.api.model = "hy-mt2-pro"
            cfg.api.fallback_models = ["hy-mt2-plus"]
            cfg.api.key_env = ""
            cfg.api.max_total_tokens = 100
            sent = []

            class Opener:
                def open(self, request, timeout):
                    sent.append(json.loads(request.data)["model"])
                    data = {"choices": [{"message": {"content": "译文"}}],
                            "usage": {"prompt_tokens": 60, "completion_tokens": 20}}
                    return io.BytesIO(json.dumps(data).encode())

            with patch("hymt.api_engine.urllib.request.build_opener", return_value=Opener()):
                engine = APIEngine(cfg)
                engine.generate("one", max_new_tokens=20)
                engine.generate("two", max_new_tokens=20)
            self.assertEqual(sent, ["hy-mt2-pro", "hy-mt2-plus"])
            usage = json.loads((cfg.state_dir / "api-usage.json").read_text())["models"]
            self.assertEqual(usage, {"hy-mt2-pro": 80, "hy-mt2-plus": 80})
            before = cfg.fingerprint()
            cfg.api.max_total_tokens = 200
            self.assertEqual(cfg.fingerprint(), before)

    def test_interrupted_response_retries_without_caching_partial_text(self):
        cfg = load_config()
        cfg.api.url = "http://127.0.0.1:1234/v1/chat/completions"
        cfg.api.model = "example"
        cfg.api.key_env = ""
        cfg.api.max_total_tokens = 0
        attempts = []

        class Interrupted:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                raise http.client.IncompleteRead(b"partial")

        class Opener:
            def open(self, request, timeout):
                attempts.append(1)
                if len(attempts) == 1:
                    return Interrupted()
                return io.BytesIO(b'{"choices":[{"message":{"content":"done"}}]}')

        with patch("hymt.api_engine.urllib.request.build_opener", return_value=Opener()), \
                patch("hymt.api_engine.time.sleep"):
            engine = APIEngine(cfg)
            self.assertEqual(engine.generate("text", max_new_tokens=10), "done")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(engine.calls, 1)


if __name__ == "__main__":
    unittest.main()
