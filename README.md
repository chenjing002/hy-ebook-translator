# hy-ebook-translator

A local Markdown translation and publishing pipeline for macOS. English goes in,
bilingual Chinese–English Markdown comes out, and the result is published as a
static reading site on GitHub Pages.

Translation runs entirely on-device with [Tencent Hunyuan's Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2)
through [Apple MLX](https://github.com/ml-explore/mlx-lm). No text is sent to a
cloud API.

**Read the library →** <https://chenjing002.github.io/hy-ebook-translator/>

```
inputs/*.md  →  parse  →  translate  →  assemble  →  validate  →  outputs/*.md  →  git push  →  Pages
```

## Quick start

```bash
pip install -r requirements.txt    # or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./hy setup                         # download the model weights (resumable)
cp samples/mlx-primer.md inputs/   # or drop in your own .md files
./hy run                           # scan → translate → validate → save → publish
./hy serve                         # preview the site at http://127.0.0.1:8000
```

`./hy run` is the whole workflow. Everything else is for inspecting or redoing
one stage of it.

## Commands

| Command | What it does |
|---|---|
| `./hy setup` | Checks the machine and downloads weights into `models/`. Resumable — re-run it after an interruption and it continues from where it stopped. |
| `./hy scan` | Lists `inputs/` with the state of each file: new, edited, or already translated. |
| `./hy run [FILE…]` | The full pipeline. `--force` ignores the cache, `--no-publish` skips git, `--no-push` commits without pushing, `--strict` exits non-zero on validation errors. |
| `./hy translate [FILE…]` | Translate and validate without touching git. |
| `./hy validate` | Re-check the generated files against their sources. `--verbose` lists every warning. |
| `./hy site` | Rebuild `library.json`, the shelf manifest. |
| `./hy publish [--pages]` | Commit and push. `--pages` also enables GitHub Pages through the `gh` CLI. |
| `./hy serve` | Serve the site locally. |
| `./hy status` | Model, weights, direction, counts, git state. |
| `./hy clean [FILE…]` | Discard cached translation state. `outputs/` is left alone. |

Global flags override configuration for one run: `--model`, `--target-lang`,
`--source-lang`, `--order`.

## What the output looks like

Each block appears twice — the Chinese translation first, then the English
original:

```markdown
## 统一内存

<!--en-->
*Unified Memory*

在大多数加速器上，数组要么位于主机内存，要么位于设备内存……

<!--en-->
On most accelerators an array lives in either host memory or device memory,
and moving it between them is an explicit, expensive step.
```

The `<!--en-->` comments are invisible in every Markdown renderer, including
GitHub's. The reading site uses them to style the original more quietly and to
offer a 双语 / 中文 / EN toggle.

Blocks with no prose — fenced code, display math, raw HTML, horizontal rules,
link definitions — are emitted **once**, byte for byte. Duplicating a code
listing would add noise rather than fidelity.

## How structure is preserved

Markdown is never handed to the model as raw text. Each document is parsed into
typed blocks first, and only the prose is translated.

**Block level.** `hymt/mdparse.py` splits the document into front matter,
headings (ATX and setext), paragraphs, fenced code, display math (`$$`, `\[`,
and `\begin{...}` environments), pipe tables, lists, blockquotes, HTML blocks,
thematic breaks, footnote definitions and link reference definitions. Each block
knows how to rebuild itself from translated segments, so a table keeps its
column count and alignment row, a nested list keeps its indentation and
markers, and a fenced code block inside a list item survives untouched.

**Converted documents.** Markdown that came out of a PDF or EPUB converter
breaks the usual assumptions: a whole `<table>…</table>` sits on one line, the
next paragraph begins on the line below it, and the file may hold only a few
dozen blank lines in several thousand. CommonMark ends an HTML block at the
next blank line, which in such a file hands the rest of the chapter to the
converter's table. The parser instead closes an HTML block as soon as the
element that opened it is closed, matching by tag name so that the stray
openers these converters emit — `<t>` standing in for an empty cell, a
`<tbody>` with no end tag — cannot hold the block open. On a 937 KB book
converted from PDF that is the difference between 50 headings recognised and
all 226.

**Inline level.** Within a prose segment, `hymt/protect.py` replaces everything
the model must not touch with a numbered sentinel: inline code, inline math,
link and image destinations, reference labels, footnote references, autolinks,
raw HTML tags, LaTeX commands, bare URLs and HTML entities. The model sees

> Install ⟪0⟫ and read ⟪1⟫ for details.

and the spans are restored afterwards. Restoration is tolerant of a model that
substitutes a look-alike bracket, and strict about reporting a sentinel that
never came back.

**Verification.** Every response is checked before it is accepted, with a
fallback ladder:

```
batch of segments  →  retry  →  one request per segment  →  segment without sentinels  →  keep the original
```

Anything that reaches the last rung is recorded as a warning and reported by
`./hy validate`. Nothing is dropped silently.

## Long documents

Segments are packed into one request up to a token budget
(`[translate].batch_token_budget`), each tagged with a `【n】` marker that the
model is instructed to preserve. A segment longer than
`[translate].max_unit_tokens` is split on sentence boundaries — abbreviations
like `e.g.`, `Fig.` and `Dr.` are not treated as sentence ends — and rejoined
without stray spaces between Chinese characters.

Each batch is checkpointed to `.state/<document>.json` as soon as it completes.
Interrupt a run at any point and the next one picks up where it left off.
Translations are keyed by a hash of the **segment content**, not its position,
so editing one paragraph of a long book re-translates that paragraph only, and
reordering chapters costs nothing.

The preceding translated segment is passed back to the model as background
context (`[translate].context_units`), which keeps terminology consistent across
a long document.

## Validation

`./hy validate` re-reads each generated file and compares it against its source:

- every verbatim block present byte for byte
- every original prose block still present
- every link and image destination intact
- no internal sentinel left in the output
- heading count not reduced
- Chinese actually present, in proportion to the prose
- balanced inline code and emphasis markers
- every warning the translator recorded

## The reading site

GitHub Pages serves the repository root. `index.html` is the library,
`reader.html` renders one document, and the Markdown it fetches is the very same
file in `outputs/` — there is only ever one copy of a translation.

- Markdown via [marked](https://marked.js.org/), math via
  [KaTeX](https://katex.org/), code via
  [highlight.js](https://highlightjs.org/) — all pinned with subresource
  integrity hashes.
- Math is lifted out of the Markdown before it is parsed and rendered back
  afterwards, using the same rule as `hymt/protect.py`: a `$…$` span may not be
  padded with spaces unless what it holds is unmistakably TeX, and it is
  neither long nor full of markup. KaTeX's own auto-render treats every `$` as
  a delimiter, which in a book on financial statements renders the prose
  between two dollar amounts as algebra. Extracting first also keeps Markdown
  from eating `\$`, `&` and the subscripts in `$a_i$`.
- Typography tuned for mixed scripts: a serif CJK stack at a generous line
  height for the translation, a smaller quieter setting for the original.
- Language toggle, text size, table of contents with scroll spy, reading
  progress, light/dark, and a layout that works at phone width.

To enable Pages the first time:

```bash
./hy publish --pages          # needs the gh CLI, or do it in repository settings
```

## Configuration

`config.toml` holds the defaults; copy it to `config.local.toml` to override
anything without touching the tracked file.

```toml
[model]
repo = "mlx-community/Hy-MT2-7B-8bit"
temperature = 0.7      # upstream-recommended sampling for the 1.8B/7B models
top_p = 0.6
top_k = 20
repetition_penalty = 1.05

[translate]
source_lang = "English"
target_lang = "Simplified Chinese"
prompt_lang = "zh"     # Hy-MT2's Chinese instruction template
batch_token_budget = 640
context_units = 1

[layout]
order = "zh-first"            # or "en-first"
heading_style = "italic-below"   # or "inline", "none"
markers = true

[publish]
repo = "chenjing002/hy-ebook-translator"
include_inputs = false       # source ebooks stay local by default
```

### Choosing weights

| Repository | Size | Resident memory | Notes |
|---|---:|---:|---|
| `mlx-community/Hy-MT2-7B` | 15 GB | ~16 GB | bf16 reference; needs 32 GB to be comfortable |
| `mlx-community/Hy-MT2-7B-8bit` | 8 GB | ~8.5 GB | **default** — fits a 16 GB Mac with room |
| `mlx-community/Hy-MT2-7B-4bit` | 4.5 GB | ~5 GB | fastest, slight loss on long technical prose |
| `mlx-community/Hy-MT2-1.8B-4bit` | 1 GB | ~1.5 GB | useful for a quick smoke test |

`./hy status` tells you what is downloaded and whether it fits.

Weights land in `models/` (git-ignored) and are fetched with resumable `curl`
rather than a Python downloader, because a multi-gigabyte download over a slow
link needs resume above all else: interrupt `./hy setup` at any point and
re-running it continues from the byte it stopped on.

The same MLX conversions are published on HuggingFace and mirrored on
`hf-mirror.com` and ModelScope. Throughput between them varies by an order of
magnitude depending on time and location, so `./hy setup` measures all three on
a real weight file and uses whichever is fastest:

```
→ measuring download endpoints …
    https://hf-mirror.com              527 KB/s
    https://huggingface.co             471 KB/s
    https://www.modelscope.cn          292 KB/s
  using https://hf-mirror.com — roughly 253 min at this rate
```

Pin one with `./hy setup --endpoint https://www.modelscope.cn`.

### Other language pairs

The direction is configuration, not code:

```bash
./hy run --target-lang Japanese
./hy run --source-lang Chinese --target-lang English
```

Hy-MT2 is multilingual; `hymt/engine.py` carries the Chinese and English
instruction templates and the language-name table used to fill them.

## Layout

```
inputs/          English Markdown you provide          (not published by default)
outputs/         bilingual Markdown, same filenames    (published)
models/          downloaded weights                    (git-ignored)
.state/          resume checkpoints and segment cache   (git-ignored)
index.html       the library page
reader.html      the document reader
assets/          stylesheet and front-end script
library.json     generated shelf manifest
samples/         a document to try the pipeline on
hymt/
  cli.py         command line interface
  config.py      configuration loading
  download.py    resumable weight downloader
  engine.py      MLX model wrapper and Hy-MT2 prompt templates
  mdparse.py     Markdown block parser and rebuilder
  protect.py     inline sentinel protection
  translate.py   batching, retries, fallback ladder
  assemble.py    bilingual layout
  validate.py    structural checks
  state.py       content-addressed resume state
  site.py        library manifest
  publish.py     git and GitHub Pages
  pipeline.py    the workflow
tests/           43 tests, no weights required
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite runs without model weights: a fake engine speaks the same protocol as
Hy-MT2 (segment markers in, segment markers out), which is what allows the
parser, the sentinel protection, the batching fallbacks, the assembler, the
validator and the resume logic to be tested end to end. It includes a
block-by-block round-trip proof that the parser can rebuild
`tests/fixtures/complex.md` exactly.

## Requirements

- macOS on Apple Silicon (MLX is Metal-only)
- Python 3.10+
- `curl` for downloads; `git` to publish; `gh` only for `--pages`

## Licence

Pipeline code: MIT. The Hy-MT2 weights follow the upstream Tencent Hunyuan
licence; translated documents inherit the licence of their source material.
