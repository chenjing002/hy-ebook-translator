# Markdown boundary sample review

On 2026-10-14, 45 unresolved line boundaries were sampled across the six
`inputs/` books. The sample was stratified by sentence-ending punctuation,
colon, short lines, and long lines, using random seed `20261014`. An AI review
classified each boundary using only the two adjacent line excerpts; no source
words were rewritten.

| Source | Sampled | Split | Join | Keep |
|---|---:|---:|---:|---:|
| Financial Statement Analysis | 8 | 8 | 0 | 0 |
| How to Lie with Statistics | 8 | 7 | 1 | 0 |
| Security Analysis | 8 | 8 | 0 | 0 |
| Show Me the Money | 8 | 8 | 0 | 0 |
| Special Situation Investing Classes | 8 | 8 | 0 | 0 |
| Huishan short report | 5 | 0 | 0 | 5 |
| **Total** | **45** | **39** | **1** | **5** |

The split decisions clustered around table-of-contents and index entries,
glossary terms, short standalone labels, and quotations introduced by a colon.
The join decision was a sentence wrapped immediately after an em dash. The five
kept boundaries were fragments of line-oriented bilingual project records in
the Huishan report, where local context was insufficient to infer paragraphs.

The rules now cover the repeated, high-confidence patterns. Boundaries that
remain uncertain are unchanged unless an explicitly configured local AI
classifier selects an action. The sample is a targeted review, not a claim that
every remaining boundary is correct.

After the rule update, a second deterministic sample of five remaining
boundaries per book was reviewed. Of these 30 boundaries, 21 were marked
`split`, 3 `join`, and 6 `keep`. Their source hashes, boundary signatures, and
actions are stored in `docs/preprocess-reviewed-decisions.json`; no book text
is stored there. They can be applied reproducibly with:

```bash
python3 scripts/preprocess_markdown.py inputs/ \
  --review-file docs/preprocess-reviewed-decisions.json \
  --output-dir preprocessed/ --overwrite
```
