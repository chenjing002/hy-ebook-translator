---
title: Structure Torture Test
author: Pipeline
---

# Attention and Memory

A paragraph with `inline_code()`, inline math $E = mc^2$, a [link](https://example.com/a_b),
an ![image](imgs/figure-1.png "Figure 1"), a reference [link][spec], a footnote[^note],
**bold**, *italic*, an autolink <https://mlx.dev>, an entity &mdash; and a raw <kbd>Tab</kbd> tag.

## Scaled Dot-Product Attention

The scaled dot-product attention is defined as follows, where $d_k$ is the key dimension:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
$$

\begin{align}
a &= b + c \\
d &= e - f
\end{align}

### Properties

| Property | Complexity | Notes |
|----------|-----------:|-------|
| Time | $O(n^2 d)$ | quadratic in sequence length |
| Memory | $O(n^2)$ | dominated by the attention matrix |
| Parallelism | $O(1)$ | all positions computed at once |

Here is a numbered procedure:

1. Project the inputs into queries, keys and values.
2. Compute pairwise scores and scale them.
   - Divide by $\sqrt{d_k}$ to keep variance stable.
   - Apply the causal mask where needed.
3. Normalise with a softmax and take the weighted sum.

And a loose list with nested content:

- First item, which continues
  onto a second source line.
- Second item with its own paragraph:

  This is that paragraph, indented under the item.

  ```python
  def attention(q, k, v):
      scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
      return mx.softmax(scores, axis=-1) @ v
  ```

- Third item.

> Attention is a mechanism for routing information.
> It replaces recurrence with direct access.
>
> - Even a list inside a quotation must survive.

Setext Heading Style
--------------------

Some closing prose that mentions MLX, Apple Silicon, and the number 42. It also
has a price of $5 and $6, which must not be read as mathematics.

```
plain fenced block with no language
  indentation preserved
```

~~~js
// a tilde fence
const x = {a: 1, b: 2};
~~~

<div align="center">
  <strong>Raw HTML block</strong>
</div>

---

[^note]: Footnotes are translated but keep their label.

[spec]: https://example.com/spec "The Specification"
