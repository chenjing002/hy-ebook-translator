"""Tests for whitespace-only repair of converted Markdown."""

import io
import json
import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hymt.preprocess import Boundary, openai_compatible_classifier, prepare


def characters(text: str) -> str:
    return re.sub(r"\s", "", text)


class TestPreprocess(unittest.TestCase):
    def test_repairs_obvious_paragraphs_and_soft_wraps(self):
        source = (
            "# Chapter\n"
            "This is a long paragraph with enough words to make its ending unambiguous. "
            "It explains the first idea in a complete sentence.\n"
            "The second paragraph begins with a different idea and continues for a good "
            "while, so that it is clearly a separate paragraph.\n"
            "This sentence runs over the page and keeps going\n"
            "on the next line without ending the thought.\n"
        )
        result = prepare(source)
        self.assertIn("sentence.\n\nThe second", result.text)
        self.assertIn("going on the next", result.text)
        self.assertEqual(characters(result.text), characters(source))
        self.assertEqual(prepare(result.text).text, result.text)

    def test_protected_blocks_are_exact(self):
        protected = (
            "```python\nprint('one')\nprint('two')\n```\n"
            "<table><tr><td>First\nSecond</td></tr></table>\n"
            "$$\na + b\n$$\n"
        )
        source = protected + "A complete sentence.\n![Image](https://example.com/a.png)\nCaption.\n"
        result = prepare(source)
        self.assertTrue(result.text.startswith(protected))
        self.assertIn("sentence.\n\n![Image]", result.text)
        self.assertEqual(characters(result.text), characters(source))

    def test_multiline_inline_code_is_untouched(self):
        source = "This is `a code\nspan` inside the sentence.\nAnother sentence.\n"
        result = prepare(source)
        self.assertIn("`a code\nspan`", result.text)

    def test_ai_only_selects_boundary_action(self):
        source = "A short sentence.\nA short following sentence.\n"
        seen = []

        def decide(boundaries):
            seen.extend(boundaries)
            return {boundaries[0].id: "split"}

        result = prepare(source, decide)
        self.assertEqual(len(seen), 1)
        self.assertEqual(result.ai_decisions, 1)
        self.assertEqual(result.text, "A short sentence.\n\nA short following sentence.\n")
        self.assertEqual(characters(result.text), characters(source))

    def test_index_entries_are_separate(self):
        source = "# Index\nAcme Steel, 12\nBeta Holdings, 34\n  bonds, 35\n"
        result = prepare(source)
        self.assertIn("Acme Steel, 12\n\nBeta Holdings, 34\n\n  bonds, 35", result.text)
        self.assertEqual(characters(result.text), characters(source))

    def test_line_oriented_sections_and_glossary_entries(self):
        source = (
            "# Table of Contents\nIntroduction\nChapter One\n"
            "# F\nFair Debt Collection Practices Act, 350\nFairfax Financial, 169\n"
            "**acquisition:** A purchase of a company.\n"
            "**adverse opinion:** An auditor's report.\n"
        )
        result = prepare(source)
        self.assertIn("Introduction\n\nChapter One", result.text)
        self.assertIn("Act, 350\n\nFairfax", result.text)
        self.assertIn("a company.\n\n**adverse opinion:**", result.text)

    def test_colon_introduced_quote_and_dash_continuation(self):
        source = (
            "A long passage explains the finding, then introduces the following quotation from the study:\n"
            "The quoted passage begins on its own line and runs long enough to stand alone.\n"
            "But do not mention the obvious objection—you—\n"
            "is it?—may wish to leave that out.\n"
        )
        result = prepare(source)
        self.assertIn("study:\n\nThe quoted passage", result.text)
        self.assertIn("you— is it?", result.text)

    def test_invalid_ai_decision_fails_closed(self):
        with self.assertRaises(ValueError):
            prepare("A short sentence.\nAnother sentence.\n",
                    lambda boundaries: {boundaries[0].id: "rewrite"})

    def test_ai_budget_can_disable_calls(self):
        classifier = openai_compatible_classifier(
            "http://127.0.0.1:1234/v1/chat/completions", "local", max_boundaries=0)
        result = prepare("A short sentence.\nAnother sentence.\n", classifier)
        self.assertEqual(result.ai_decisions, 0)

    def test_ai_endpoint_must_really_be_local(self):
        with self.assertRaises(ValueError):
            openai_compatible_classifier(
                "http://localhost:1234@remote.example/v1/chat/completions", "local")

    def test_ai_sends_only_budgeted_boundary_windows(self):
        sent = []

        class Opener:
            def open(self, request, timeout):
                sent.append(json.loads(request.data))
                reply = {"choices": [{"message": {"content": '{"2":"split"}'}}]}
                return io.BytesIO(json.dumps(reply).encode())

        with patch("hymt.preprocess.urllib.request.build_opener", return_value=Opener()):
            classifier = openai_compatible_classifier(
                "http://localhost:1234/v1/chat/completions", "local", max_boundaries=1)
            decisions = classifier([
                Boundary(1, "short title", "short title"),
                Boundary(2, "A complete sentence.", "Another paragraph begins."),
            ])
        self.assertEqual(decisions, {2: "split"})
        self.assertEqual(len(sent), 1)
        self.assertIn('"id": 2', sent[0]["messages"][0]["content"])
        self.assertNotIn('"id": 1', sent[0]["messages"][0]["content"])

    def test_review_file_writes_copy_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "book.md"
            original = "A short sentence.\nAnother short sentence.\n"
            source.write_text(original, encoding="utf-8")
            pending = []
            prepare(original, lambda items: (pending.extend(items), {})[1])
            boundary = pending[0]
            signature = hashlib.sha256(
                (boundary.before + "\n" + boundary.after).encode()
            ).hexdigest()
            review = root / "review.json"
            review.write_text(json.dumps({"version": 1, "files": {"book.md": {
                "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "decisions": {str(boundary.id): {"signature": signature, "action": "split"}},
            }}}), encoding="utf-8")
            output_dir = root / "prepared"
            script = Path(__file__).resolve().parent.parent / "scripts" / "preprocess_markdown.py"
            completed = subprocess.run(
                [sys.executable, str(script), str(source), "--review-file", str(review),
                 "--output-dir", str(output_dir)], capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(source.read_text(), original)
            self.assertEqual((output_dir / "book.md").read_text(),
                             "A short sentence.\n\nAnother short sentence.\n")


if __name__ == "__main__":
    unittest.main()
