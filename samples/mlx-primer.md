# A Short Primer on Apple MLX

MLX is an array framework for machine learning on Apple silicon. It borrows the
feel of NumPy and the composability of PyTorch, but it is designed around one
architectural fact: on an Apple system the CPU and the GPU share the same
memory.

## Unified Memory

On most accelerators an array lives in either host memory or device memory, and
moving it between them is an explicit, expensive step. MLX arrays live in
unified memory instead. An operation is told *where to run*, not *where the data
is*:

```python
import mlx.core as mx

a = mx.random.normal((1024, 1024))
b = mx.random.normal((1024, 1024))

c = mx.matmul(a, b, stream=mx.gpu)   # runs on the GPU
d = mx.add(a, b, stream=mx.cpu)      # same arrays, no copy
```

This is why a 7-billion-parameter model can be loaded once and used from either
device without a transfer: the weights are simply *there*.

## Lazy Evaluation

Nothing is computed until a result is actually needed. Building an expression
only records it:

| Stage | What happens | Cost |
|-------|--------------|-----:|
| Build | The graph is recorded | ~0 |
| `mx.eval()` | The graph is compiled and run | full |
| Access | Implicit evaluation, then read | full |

Laziness lets MLX fuse neighbouring operations. The attention score
computation, for instance, becomes a single kernel rather than three:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
$$

where $d_k$ is the dimension of the key vectors and the scaling by
$1/\sqrt{d_k}$ keeps the variance of the scores stable as $d_k$ grows.

## Quantisation

Weights can be stored at lower precision to fit a larger model into the same
memory:

1. **bfloat16** — the reference format. Two bytes per parameter.
2. **8-bit** — roughly half the memory, with quality differences that are hard
   to detect in translation work.
3. **4-bit** — a quarter of the memory, noticeably faster, with some loss on
   long or technical passages.

For a 7B model that is about 15 GB, 8 GB and 4.5 GB respectively. The practical
consequence is simple: a 16 GB Mac can run a 7B model comfortably at 8-bit
precision, and cannot run one at bfloat16.

> Choose the largest model that fits in memory with room to spare, then the
> highest precision that still fits. Swapping is far more expensive than
> quantisation.

## Further Reading

See the [MLX documentation](https://ml-explore.github.io/mlx/) and the
[model gallery](https://huggingface.co/mlx-community) for converted weights.[^1]

[^1]: Most community conversions ship bf16, 8-bit and 4-bit variants of the
    same checkpoint.
