# hy-ebook-translator

Markdown book preparation, API translation, and bilingual publishing. The
existing reading site is at <https://chenjing002.github.io/hy-ebook-translator/>.

The pipeline parses Markdown into typed blocks, sends only translatable prose
to a configured API, rebuilds bilingual Markdown, validates it, and can publish
the result to GitHub Pages. The translation API is **unconfigured by default**.

```
inputs/*.epub, *.pdf → convert → inputs/*.md → preprocess → preprocessed/*.md
                                → translate → output/*.md → site
```

## Convert an EPUB or PDF to Markdown

Two converters turn an ebook into one Markdown file with its images in a
`<name>_files/` directory beside it, referenced relatively so the pair can be
moved together:

```bash
python3 scripts/epub_to_markdown.py book.epub -o "inputs/Book Title.md"
python3 scripts/pdf_to_markdown.py book.pdf -o "inputs/Book Title.md"
```

With `--output-dir` instead of `-o`, several files are converted at once and
each Markdown file takes the ebook's name. Both add a title heading and author
line from the file's metadata (`--title` overrides it, `--no-title` omits it),
and both write a provenance comment the reader hides. `--no-images` drops the
images; `--overwrite` replaces an existing Markdown file.

The EPUB converter uses only the standard library. It follows the spine order,
skips the navigation document and non-linear items, keeps headings, emphasis,
lists, quotes, tables, figures, code, footnotes and external links, and renders
publisher class names such as `Italic`, `Semibold`, `Subscript` and `AllCaps`.
When the book's chapters are `h1` headings they move down one level so the
title stays alone at the top. A table with spanning cells is kept as HTML.

The PDF converter reads with PyMuPDF (`pip install pymupdf`) when installed,
and otherwise with poppler's `pdftotext`. PyMuPDF gives headings by font size,
bold and italic, detected tables and extracted images; `pdftotext` gives text
only, with headings limited to "Chapter N" and numbered-section lines. Both
remove running headers, footers and page numbers, undo end-of-line
hyphenation, and rejoin paragraphs cut by a page break. A converted PDF
usually still benefits from the whitespace repair described next.

## OCR a scanned PDF (macOS)

A scanned book has no text layer worth keeping, or one produced by a weak
OCR engine. On macOS, `scripts/ocr_pdf_to_markdown.py` reads every page with
Apple's Vision text recogniser (the engine behind Live Text) and rebuilds the
document from the geometry of the recognised lines:

```bash
pip install pymupdf pyobjc-framework-Vision pyobjc-framework-Quartz
python3 scripts/ocr_pdf_to_markdown.py scan.pdf -o "inputs/Book Title.md"
python3 scripts/ocr_pdf_to_markdown.py scan.pdf --pages 100-120 -o /tmp/sample.md
```

Headings come from the PDF outline (matched to the page where the title is
actually printed, since bookmarks are often a page off) and from Chinese
chapter, section and `一、` patterns; paragraphs from first-line indents and
line geometry, joined across page breaks; tables from three or more rows of
aligned cells, with stacked header rows merged and a table continued on the
next page joined to the first; photographs and charts are cropped from
text-free gaps that contain ink and saved to `<name>_files/`. Running
headers, page numbers, printed contents pages, text read off signs inside
photographs and pages bound in twice are dropped. Per-page recognition is
cached in `.state/ocr/<pdf stem>/`, so an interrupted run resumes and the
layout pass can be repeated in seconds after a rule change. `--dpi` (default
220), `--langs` (default `zh-Hans,en-US`), `--workers` and `--keep-contents`
adjust the run. On this machine Vision needs about 0.7 s per page; three
worker processes convert a 1,200-page book in roughly 7 minutes.

Vision and the PaddleOCR-based RapidOCR engine read the same test pages with
the same accuracy on prose; Vision was chosen because it runs several times
faster and returns one box per table cell, which the table reconstruction
depends on.

A converted book can also be published without translation: copy the Markdown
file and its `_files/` directory into `output/`, give it a `[titles]` entry in
`config.toml`, and run `python3 -m hymt publish`. The reader shows a
single-language document as it is; `hy validate` will report that document as
untranslated, which is accurate.

## Prepare converted Markdown

PDF and EPUB conversions often leave one newline between natural paragraphs,
wrap a sentence across lines, or merge a whole index into one Markdown block.
Preview the rule-based repair:

```bash
python3 scripts/preprocess_markdown.py inputs/
```

Save new copies while keeping the originals:

```bash
python3 scripts/preprocess_markdown.py inputs/ --output-dir preprocessed/
```

Use `--overwrite` with `--output-dir` to regenerate existing copies after a rule
change. `preprocessed/` is local and git-ignored. The program changes only
whitespace inside ordinary prose blocks; it verifies that every non-whitespace
character remains in order. Code, math, tables, HTML, headings, and inline
code/math/link spans are protected. It is safe to rerun: a second pass makes no
further changes.

The sampled AI review and its reproducible boundary decisions are in
`docs/preprocess-sample-review.md` and
`docs/preprocess-reviewed-decisions.json`. Add `--review-file
docs/preprocess-reviewed-decisions.json` when generating the current copies.

An optional local OpenAI-compatible chat endpoint can classify uncertain
line boundaries. It receives only short excerpts around those boundaries and
returns `split`, `join`, or `keep`, never replacement prose:

```bash
python3 scripts/preprocess_markdown.py inputs/ \
  --ai-url http://127.0.0.1:1234/v1/chat/completions \
  --ai-model YOUR_LOCAL_MODEL --ai-limit 128 \
  --output-dir preprocessed/ --overwrite
```

No AI service is required for the rules pass. The limit is per file. Ambiguous
boundaries outside the limit stay as they are. The separate
`scripts/prepare_book.py` can still split exceptionally long paragraphs at
sentence boundaries; it is a different, optional operation.

## Translation API

An OpenAI-compatible chat completions endpoint can be configured in
`config.local.toml` (which is git-ignored):

```toml
[api]
url = "https://YOUR_PROVIDER/v1/chat/completions"
model = "YOUR_MODEL"
key_env = "TRANSLATION_API_KEY"

[paths]
inputs = "preprocessed"
```

Set the named environment variable to the API key. The key is never stored in
the config or repository. A remote endpoint must use HTTPS. The adapter uses
approximate token counts to pack requests and reads actual usage from API
responses when provided. No provider SDK or local model package is required.

For the processed books, `config.hy-mt2.toml` selects Tencent TokenHub's
Hy-MT2-Pro, then Plus, then Lite. It caps this project's recorded usage at
990,000 tokens per model, below each 1,000,000-token free allowance. The
credential remains in `TRANSLATION_API_KEY`; the config contains no key. It
reads from `preprocessed/` and writes bilingual files to `output/`, preserving
the source files. Short paragraphs are sent together with numbered markers to
reduce prompt overhead; every returned paragraph is checked and cached
separately in `.state/` for resuming after interruption. The request disables
thinking. This project-local cap cannot see usage from other applications
using the same Tencent account.

```bash
export TRANSLATION_API_KEY="..."
python3 -m hymt --config config.hy-mt2.toml translate
```

Running the same command again reuses completed paragraphs. `output/` is the
published document directory: its bilingual Markdown is what the reading site
serves. If all three project caps are reached, the command stops and keeps its
checkpoints; increasing the cap later does not invalidate cached translations.

```bash
python3 -m hymt --input-dir preprocessed scan
python3 -m hymt --input-dir preprocessed translate "book.md"
python3 -m hymt --input-dir preprocessed run --no-publish
```

The command is also installed as `hy` by `pip install -e .`. `run` publishes by
default; use `--no-publish` to keep the generated output local. API settings can
also be supplied for one command with global `--api-url`, `--api-model`, and
`--api-key-env` flags before the subcommand.

## Markdown handling

`hymt/mdparse.py` recognises headings, paragraphs, code fences, display math,
tables, lists, blockquotes, HTML, footnotes, and link definitions. Code, math,
non-table HTML, and link definitions pass through verbatim. HTML table text is
translated cell by cell. `hymt/protect.py` masks inline code, math, link
destinations, HTML tags, URLs, and entities before a translation request, then
restores them after the response.

The translator batches segments, checks every marker and placeholder, retries
invalid responses, and saves segment-level progress in `.state/`. The API URL,
model priority, and translation settings are part of the cache identity. The assembler
puts the translation and original beside each other, with invisible
`<!--en-->` markers for the reader's language toggle. The validator checks
that original blocks, links, headings, and translated content remain present.

## Other commands

| Command | Purpose |
|---|---|
| `python3 -m hymt scan` | List source files and translation state |
| `python3 -m hymt validate` | Check existing outputs against sources |
| `python3 -m hymt site` | Rebuild `library.json` |
| `python3 -m hymt publish` | Commit and push the reading site |
| `python3 -m hymt serve` | Preview the site locally |
| `python3 -m hymt status` | Show API, document, and repository status |
| `python3 -m hymt clean` | Remove segment cache, preserving outputs |

`config.toml` holds defaults; `config.local.toml` holds private overrides.
`inputs/`, `preprocessed/` and `.state/` are git-ignored. `output/` holds the
published bilingual files; `[paths].outputs` names it, and `library.json`
records that name so the reader fetches from the right place. Python 3.11+ is
required; runtime dependencies are from the standard library. Run the test
suite with:

```bash
python3 -m unittest discover -s tests -q
```

Pipeline code: MIT. Translated documents inherit the licence of their source
material.
