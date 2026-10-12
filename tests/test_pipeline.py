"""Pipeline tests.

These run without model weights: a fake engine stands in for Hy-MT2 and speaks
the same protocol (segment markers in, segment markers out), which is what lets
the parser, the placeholder protection, the assembler and the validator be
tested end to end.

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hymt.assemble import assemble, build_header  # noqa: E402
from hymt.config import load_config  # noqa: E402
from hymt.mdparse import parse_markdown, render_blocks  # noqa: E402
from hymt.pipeline import run, translate_one  # noqa: E402
from hymt.protect import mask  # noqa: E402
from hymt.site import build_library  # noqa: E402
from hymt.translate import SEG_RE, _join_zh, _split_segments, _split_sentences  # noqa: E402
from hymt.validate import ERROR, Report, validate  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CJK_RE = re.compile(r"[一-鿿]")


# ── a stand-in for the model ─────────────────────────────────────────────────


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text) // 3)))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return messages[0]["content"]


class FakeEngine:
    """Answers like Hy-MT2 would: Chinese text, markers and placeholders kept."""

    def __init__(self, cfg, drop_marker: bool = False, drop_placeholder: bool = False):
        self.cfg = cfg
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.loaded = True
        self.drop_marker = drop_marker
        self.drop_placeholder = drop_placeholder
        self._tokenizer = FakeTokenizer()

    def load(self, verbose=False):
        return None

    @property
    def tokenizer(self):
        return self._tokenizer

    def count_tokens(self, text):
        return max(1, len(text) // 3)

    def build_prompt(self, text, *, delimiters=False, background=""):
        return text

    def generate(self, prompt, max_new_tokens=None):
        self.calls += 1
        body = prompt.split("\n\n")[-1]
        segments = [m for m in SEG_RE.finditer(body)]
        if not segments:
            return self._translate(body)
        out = []
        for k, m in enumerate(segments):
            end = segments[k + 1].start() if k + 1 < len(segments) else len(body)
            piece = body[m.end():end].strip()
            idx = m.group(1)
            if self.drop_marker and k == 1:
                out.append(self._translate(piece))          # marker omitted
            else:
                out.append(f"【{idx}】 {self._translate(piece)}")
        return "\n".join(out)

    def _translate(self, text):
        if self.drop_placeholder:
            text = re.sub(r"⟪\d+⟫", "", text)
        # Keep every placeholder, prove the text changed, and emit real CJK.
        return "译文：" + text + "（中文）"


# ── unit level ───────────────────────────────────────────────────────────────


class TestMarkdownParser(unittest.TestCase):
    def setUp(self):
        self.src = (FIXTURES / "complex.md").read_text()
        self.blocks = parse_markdown(self.src)

    def test_block_kinds_detected(self):
        kinds = {b.kind for b in self.blocks}
        for expected in {"front_matter", "heading", "paragraph", "math", "table",
                         "list", "blockquote", "code", "html", "hr", "footnote", "linkdef"}:
            self.assertIn(expected, kinds, f"{expected} not recognised")

    def test_round_trip_is_lossless(self):
        """Rebuilding each block with its original text reproduces that block.

        Verbatim kinds must match byte for byte. Prose kinds are compared with
        whitespace collapsed, because the parser joins soft-wrapped lines and
        normalises table cell padding — both semantically identical in Markdown.
        """
        lookup = {u.uid: u.text for b in self.blocks for u in b.iter_units()}
        get = lambda uid: lookup[uid]  # noqa: E731
        exact = {"code", "math", "html", "hr", "linkdef", "front_matter"}
        for block in self.blocks:
            rebuilt = block.render_target(get)
            with self.subTest(kind=block.kind, at=block.raw[:40]):
                if block.kind in exact:
                    self.assertEqual(rebuilt, block.raw)
                elif block.kind == "heading":
                    self.assertIn(block.meta["text"], rebuilt)
                elif block.kind == "blockquote":
                    # Joining a soft-wrapped quoted paragraph legitimately
                    # collapses two "> " prefixes into one.
                    self.assertEqual(_unquote(rebuilt), _unquote(block.raw))
                else:
                    self.assertEqual(_flat(rebuilt), _flat(block.raw))

    def test_every_source_line_is_accounted_for(self):
        """No line of the source may be dropped by the block scanner."""
        covered = "\n".join(b.raw for b in self.blocks)
        for line in self.src.split("\n"):
            if line.strip():
                self.assertIn(line, covered, f"line lost by the parser: {line!r}")

    def test_code_blocks_are_byte_exact(self):
        for block in self.blocks:
            if block.kind == "code":
                self.assertIn(block.raw, self.src)
                self.assertEqual(block.units, [])

    def test_math_not_translated(self):
        math = [b for b in self.blocks if b.kind == "math"]
        self.assertGreaterEqual(len(math), 2)
        for block in math:
            self.assertEqual(list(block.iter_units()), [])

    def test_table_cells_become_units(self):
        table = next(b for b in self.blocks if b.kind == "table")
        texts = [u.text for u in table.iter_units()]
        self.assertIn("Property", texts)
        self.assertIn("quadratic in sequence length", texts)
        # The delimiter row is preserved verbatim, not turned into text.
        self.assertTrue(any(r["type"] == "delim" for r in table.meta["rows"]))

    def test_fence_inside_list_item_is_preserved(self):
        lists = [b for b in self.blocks if b.kind == "list"]
        raws = [seg for b in lists for seg in b.meta["segments"] if seg["type"] == "raw"]
        self.assertTrue(any("def attention" in "\n".join(seg["lines"]) for seg in raws))


class TestConvertedHtml(unittest.TestCase):
    """Markdown produced by a PDF or EPUB converter.

    Such a document has almost no blank lines: a whole `<table>…</table>` sits
    on one line and the next paragraph begins on the line below it. Ending an
    HTML block only at a blank line would hand the rest of the chapter to the
    converter's table.
    """

    def test_single_line_element_does_not_swallow_what_follows(self):
        src = (
            "<table><tr><td>Revenue</td><td>12.4</td></tr></table>\n"
            "Source: Company 10-K.\n"
            "## Not All Sales Are Final\n"
            "Revenue recognition deserves a sceptical reading.\n"
        )
        blocks = parse_markdown(src)
        kinds = [b.kind for b in blocks]
        self.assertEqual(kinds, ["html", "paragraph", "heading", "paragraph"])
        self.assertEqual(blocks[0].raw, "<table><tr><td>Revenue</td><td>12.4</td></tr></table>")

    def test_tags_that_never_close_do_not_hold_the_block_open(self):
        """`<t>` as an empty-cell placeholder, `<tbody>` with no end tag."""
        src = (
            "<table><tbody><tr><td><t></td><td><t></td></tr></table>\n"
            "# What Is Profit?\n"
            "Profit is less self-evident than it looks.\n"
        )
        blocks = parse_markdown(src)
        self.assertEqual([b.kind for b in blocks], ["html", "heading", "paragraph"])

    def test_nested_same_name_elements_are_matched(self):
        src = (
            '<div style="text-align: center;"><div>Sources: Annual Reports</div> </div>\n'
            "Because margins are fictitious, the shares are worth close to zero.\n"
        )
        blocks = parse_markdown(src)
        self.assertEqual([b.kind for b in blocks], ["html", "paragraph"])

    def test_multi_line_element_still_runs_to_its_closing_tag(self):
        src = (
            '<div class="note">\n'
            "Several lines of raw HTML.\n"
            "<span>more</span>\n"
            "</div>\n"
            "Ordinary prose resumes here.\n"
        )
        blocks = parse_markdown(src)
        self.assertEqual([b.kind for b in blocks], ["html", "paragraph"])
        self.assertEqual(blocks[0].raw.count("\n"), 3)

    def test_void_element_is_its_own_block(self):
        src = "![x](a.png)\n\n<img src=\"figure.png\">\nThe figure above shows the trend.\n"
        blocks = parse_markdown(src)
        self.assertEqual([b.kind for b in blocks], ["paragraph", "html", "paragraph"])

    def test_unclosed_opener_still_ends_at_a_blank_line(self):
        src = "<section>\nstill inside the section\n\nA new paragraph.\n"
        blocks = parse_markdown(src)
        self.assertEqual([b.kind for b in blocks], ["html", "paragraph"])
        self.assertIn("still inside", blocks[0].raw)

    def test_comment_block_ends_at_its_terminator(self):
        src = "<!-- a note\nspanning lines -->\nProse directly below.\n"
        blocks = parse_markdown(src)
        self.assertEqual([b.kind for b in blocks], ["html", "paragraph"])

    def test_converted_document_keeps_its_headings(self):
        """Every heading in a blank-line-free document must survive parsing."""
        src = "".join(
            f"<table><tr><td>row {n}</td></tr></table>\n"
            f"## Section {n}\n"
            f"Body text for section {n}.\n"
            for n in range(12)
        )
        blocks = parse_markdown(src)
        self.assertEqual(sum(1 for b in blocks if b.kind == "heading"), 12)
        self.assertEqual(sum(1 for b in blocks if b.kind == "paragraph"), 12)


class TestProtection(unittest.TestCase):
    CASES = [
        "A paragraph with `code`, $x^2$, a [link](http://x.com) and **bold** text.",
        "See ![diagram](imgs/a.png \"Title\") and [docs][ref] plus a footnote[^1].",
        "Inline math $\\alpha + \\beta$ costs $5 and $6 at <https://x.com> &mdash; done.",
        "Use `pip install mlx-lm`, then read https://huggingface.co/mlx-community.",
        "The <strong>HTML</strong> tag and \\ref{eq:1} reference.",
    ]

    def test_mask_restore_is_identity(self):
        for case in self.CASES:
            with self.subTest(case=case):
                m = mask(case)
                restored, missing = m.restore(m.text)
                self.assertEqual(restored, case)
                self.assertEqual(missing, [])

    def test_prices_are_not_math(self):
        m = mask("It costs $5 and $6 today.")
        self.assertEqual(m.spans, [])

    def test_code_only_segment_needs_no_translation(self):
        self.assertFalse(mask("`x = 2`").needs_translation)
        self.assertTrue(mask("Set `x = 2` first.").needs_translation)

    def test_degraded_bracket_is_recovered(self):
        m = mask("Install `mlx-lm` from [here](http://x.com).")
        restored, missing = m.restore("从 ⟪0⟫ 安装，见 [这里][1]。")
        self.assertEqual(missing, [])
        self.assertIn("http://x.com", restored)

    def test_lost_placeholder_is_reported(self):
        m = mask("Install `mlx-lm` now.")
        _, missing = m.restore("现在安装。")
        self.assertEqual(missing, [0])

    def test_ordinary_parentheses_survive(self):
        m = mask("Step `one`, then two, then three.")
        restored, _ = m.restore("先 ⟪0⟫，然后 (2)，再 (3)。")
        self.assertIn("(2)", restored)
        self.assertIn("(3)", restored)


class TestSegmentSplitting(unittest.TestCase):
    def test_well_formed(self):
        self.assertEqual(_split_segments("【0】 甲\n【1】 乙", 2), ["甲", "乙"])

    def test_degraded_brackets(self):
        self.assertEqual(_split_segments("[0] 甲\n(1) 乙", 2), ["甲", "乙"])

    def test_missing_marker_rejected(self):
        self.assertIsNone(_split_segments("【0】 只有一段", 2))

    def test_empty_segment_rejected(self):
        self.assertIsNone(_split_segments("【0】 甲\n【1】", 2))

    def test_abbreviations_do_not_split(self):
        parts = _split_sentences("See Fig. 3 and e.g. this. Then Dr. Who arrives. Done!")
        self.assertEqual(len(parts), 3)

    def test_cjk_join_has_no_stray_spaces(self):
        self.assertEqual(_join_zh(["你好", "世界"]), "你好世界")
        self.assertEqual(_join_zh(["你好", "MLX"]), "你好 MLX")


# ── document level ───────────────────────────────────────────────────────────


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hy-test-"))
        for name in ("inputs", "outputs", ".state"):
            (self.tmp / name).mkdir()
        shutil.copy(FIXTURES / "complex.md", self.tmp / "inputs" / "complex.md")
        shutil.copy(Path(__file__).resolve().parent.parent / "config.toml", self.tmp / "config.toml")
        self.cfg = load_config(root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **kwargs):
        engine = FakeEngine(self.cfg, **kwargs)
        src = self.tmp / "inputs" / "complex.md"
        return translate_one(self.cfg, engine, src), engine

    def test_pipeline_produces_valid_bilingual_document(self):
        outcome, _ = self._run()
        self.assertTrue(outcome.report.ok, _why(outcome.report))
        text = outcome.result.output.read_text()
        self.assertTrue(CJK_RE.search(text))
        # Both languages present for prose.
        self.assertIn("Attention is a mechanism for routing information.", text)
        self.assertIn("译文", text)

    def test_strict_run_does_not_publish_invalid_translation(self):
        engine = FakeEngine(self.cfg)
        invalid = Report("complex.md")
        invalid.add(ERROR, "quality", "translation failed validation")
        with patch("hymt.pipeline.validate", return_value=invalid), \
             patch("hymt.pipeline.git_publish") as publish:
            report = run(self.cfg, engine, strict=True)
        self.assertFalse(report.ok)
        publish.assert_not_called()

    def test_originals_preserved_byte_exact(self):
        outcome, _ = self._run()
        text = outcome.result.output.read_text()
        src = (self.tmp / "inputs" / "complex.md").read_text()
        for block in parse_markdown(src):
            if block.kind in {"code", "math", "html", "hr", "linkdef"}:
                self.assertIn(block.raw.strip(), text, f"{block.kind} block lost")

    def test_no_internal_markers_leak(self):
        outcome, _ = self._run()
        text = outcome.result.output.read_text()
        for token in ("⟪", "⟫", "⟦", "⟧"):
            self.assertNotIn(token, text)
        self.assertIsNone(re.search(r"^\s*【\d+】", text, re.MULTILINE))

    def test_translation_comes_first(self):
        outcome, _ = self._run()
        text = outcome.result.output.read_text()
        zh = text.index("译文：Attention is a mechanism")
        en = text.index("> Attention is a mechanism for routing information.")
        self.assertLess(zh, en, "Chinese block should precede the English original")

    def test_en_first_layout(self):
        self.cfg.layout.order = "en-first"
        outcome, _ = self._run()
        text = outcome.result.output.read_text()
        self.assertLess(text.index("> Attention is a mechanism"),
                        text.index("译文：Attention is a mechanism"))

    def test_resume_reuses_cached_segments(self):
        first, engine_a = self._run()
        self.assertGreater(first.result.translated, 0)
        second, engine_b = self._run()
        self.assertEqual(second.result.translated, 0)
        self.assertGreater(second.result.cached, 0)
        self.assertEqual(engine_b.calls, 0, "a resumed run must not call the model")
        self.assertEqual(first.result.output.read_text(), second.result.output.read_text())

    def test_batch_failure_falls_back_to_single_segments(self):
        outcome, engine = self._run(drop_marker=True)
        self.assertTrue(outcome.report.ok, _why(outcome.report))
        self.assertTrue(CJK_RE.search(outcome.result.output.read_text()))

    def test_dropped_placeholder_is_recorded_not_hidden(self):
        outcome, _ = self._run(drop_placeholder=True)
        problems = [w["problem"] for w in outcome.result.warnings]
        self.assertTrue(problems, "a lost placeholder must surface as a warning")
        # The published file still has every link destination.
        self.assertTrue(outcome.report.ok, _why(outcome.report))

    def test_validator_catches_a_damaged_output(self):
        outcome, _ = self._run()
        out = outcome.result.output
        damaged = out.read_text().replace("```python", "```").replace(
            "https://example.com/spec", "https://example.com/WRONG")
        out.write_text(damaged)
        report = validate(self.tmp / "inputs" / "complex.md", out, self.cfg)
        self.assertFalse(report.ok)
        codes = {i.code for i in report.errors}
        self.assertTrue({"lost-code", "lost-link", "lost-linkdef"} & codes, codes)

    def test_oversized_segment_is_split_and_rejoined(self):
        """A paragraph past max_unit_tokens goes down the sentence-split path."""
        long_para = " ".join(
            f"Sentence number {i} explains `token_{i}` and links to [doc {i}](http://x/{i})."
            for i in range(60)
        )
        src = self.tmp / "inputs" / "long.md"
        src.write_text(f"# Long\n\n{long_para}\n")
        engine = FakeEngine(self.cfg)
        outcome = translate_one(self.cfg, engine, src)
        self.assertTrue(outcome.report.ok, _why(outcome.report))
        text = outcome.result.output.read_text()
        # Every protected span must come back, in every piece.
        for i in range(60):
            self.assertIn(f"`token_{i}`", text)
            self.assertIn(f"http://x/{i}", text)
        self.assertGreater(engine.calls, 1, "a long paragraph should take several calls")

    def test_library_manifest(self):
        self._run()
        library = build_library(self.cfg)
        self.assertEqual(len(library["documents"]), 1)
        doc = library["documents"][0]
        self.assertEqual(doc["file"], "complex.md")
        self.assertGreater(doc["cjk_chars"], 0)
        self.assertGreater(doc["sections"], 0)
        self.assertEqual(doc["title"], "Attention and Memory")

    def test_heading_styles(self):
        for style in ("italic-below", "inline", "none"):
            with self.subTest(style=style):
                self.cfg.layout.heading_style = style
                shutil.rmtree(self.tmp / ".state", ignore_errors=True)
                (self.tmp / ".state").mkdir(exist_ok=True)
                outcome, _ = self._run()
                text = outcome.result.output.read_text()
                self.assertIn("Attention and Memory", text)
                if style == "inline":
                    self.assertIn("·", text)


class TestAssembler(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def _build(self, src, **over):
        for key, value in over.items():
            setattr(self.cfg.layout, key, value)
        blocks = parse_markdown(src)
        zh = {u.uid: "译文" + u.text for b in blocks for u in b.iter_units()}
        return assemble(blocks, zh, self.cfg).text

    def test_untranslated_block_is_not_duplicated(self):
        blocks = parse_markdown("ACRONYM LIST\n")
        zh = {u.uid: u.text for b in blocks for u in b.iter_units()}
        out = assemble(blocks, zh, self.cfg).text
        self.assertEqual(out.count("ACRONYM LIST"), 1)

    def test_footnote_label_appears_once(self):
        out = self._build("Body.\n\n[^1]: A note.\n")
        self.assertEqual(out.count("[^1]:"), 1)

    def test_markers_can_be_disabled(self):
        self.assertNotIn("<!--en-->", self._build("Hello there.\n", markers=False))
        self.assertIn("<!--en-->", self._build("Hello there.\n", markers=True))

    def test_header_records_provenance(self):
        header = build_header("T", "a.md", self.cfg, {"units": 3, "translated": 3})
        self.assertIn("hy-ebook-translator", header)
        self.assertIn(self.cfg.model.repo, header)


def _flat(text: str) -> str:
    """Collapse all whitespace so soft wraps and cell padding do not matter."""
    return re.sub(r"\s+", " ", text).strip()


def _unquote(text: str) -> str:
    return _flat(re.sub(r"^\s*>[ \t]?", "", text, flags=re.MULTILINE))


def _why(report) -> str:
    return "\n".join(f"{i.level} [{i.code}] {i.detail}" for i in report.issues) or "no issues"


if __name__ == "__main__":
    unittest.main(verbosity=2)
