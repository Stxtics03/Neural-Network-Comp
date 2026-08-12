# Neural Network Compression

Compressing LeNet-5 for MNIST digit classification, with a focus on
getting the accounting right rather than reporting headline numbers:
storage is converted to an actual format before any compression ratio is
claimed, and the two results that looked strongest were each re-run to
check they weren't artifacts.

Four methods, in the order they were built: magnitude pruning, INT8
quantization, unit clustering (deterministic annealing vs. k-means), and
L0-regularized structured sparsity.

## Headline result

**L0 gates at lambda=0.01: 99.00% +/- 0.03% accuracy at 6.96x +/- 0.33
compression**, against a 99.04% dense baseline — verified across 5 seeds.
The compression is structural (whole filters and neurons removed), so it
is real parameter removal rather than sparse-format bookkeeping.

## Results

Baseline: **99.04% test accuracy**, 241 KB (dense fp32 weights).

### Pruning and quantization (single seed)

| Method | Setting | Accuracy | Size | Compression |
|---|---|---|---|---|
| Baseline | — | 99.04% | 241.0 KB | 1.00x |
| Magnitude pruning | 30% sparsity | 99.06% | 252.9 KB | **0.95x** |
| Magnitude pruning | 50% sparsity | 99.02% | 180.9 KB | 1.33x |
| Magnitude pruning | 70% sparsity | 98.30% | 108.9 KB | 2.21x |
| Magnitude pruning | 90% sparsity | 63.66% | 36.9 KB | 6.54x |
| INT8 quantization | per-tensor symmetric | 99.05% | 61.0 KB | 3.95x |

### L0 structured sparsity (5 seeds)

| lambda | Accuracy | Compression |
|---|---|---|
| 0.005 | 99.01% +/- 0.09pp | 4.26x +/- 0.15 |
| 0.01 | 99.00% +/- 0.03pp | 6.96x +/- 0.33 |
| 0.02 | 98.78% +/- 0.06pp | 10.93x +/- 0.67 |

### Unit clustering (5 seeds, DA vs. k-means)

| k | DA accuracy | kmeans++ accuracy | Compression |
|---|---|---|---|
| 4 | 14.21% | 11.12% | 18.09x |
| 8 | 47.07% | 32.73% | 9.53x |
| 16 | 85.22% | 68.98% | 5.14x |
| 32 | 98.04% | 93.97% | 3.04x |
| 64 | 98.84% | 98.88% | 1.67x |

The DA column looks like a clear win. It isn't — see below.

## Findings

**Sparse storage has a break-even point, and it's easy to miss.** At 30%
sparsity the "compressed" model is actually *larger* than dense storage
(0.95x). Each stored nonzero weight costs 4 bytes (value) + 2 bytes
(index) = 6 bytes, vs. 4 bytes for a dense fp32 weight — so a naive
sparse format only wins once sparsity exceeds `4/6 ~= 33%`. Below that,
you're paying index overhead for too few actual zeros to offset it. This
is a common gotcha in pruning papers that report sparsity percentages
without converting them to an actual storage format.

**There's a sharp accuracy cliff between 70% and 90% sparsity.** Accuracy
is essentially flat from 0%->70% (99.04%->98.30%, a 0.74pp drop for more
than 2x compression), then collapses to 63.66% by 90% — a 34.6pp drop in
one step. The exact knee between 70-90% hasn't been located yet.

**Structured L0 sparsity beats everything else here, and it holds up.**
At lambda=0.01 it reaches 6.96x for a 0.04pp accuracy cost, against INT8's
3.95x and magnitude pruning's 2.21x at comparable accuracy. Across 5
seeds the accuracy spread is +/-0.03 to +/-0.09pp while the gaps between
lambda settings are an order of magnitude larger, and the compression
gaps are 4-10 standard deviations apart — the curve's shape is a
property of lambda, not of the seed.

**Deterministic annealing's advantage over k-means was a compute
artifact.** DA appeared to beat kmeans++ by a wide margin (+14.3pp at
k=8, +16.2pp at k=16). But DA runs far more inner-loop iterations per
fit, so `experiments/compute_parity_check.py` re-ran the comparison
giving kmeans++ a matched budget via multi-restart. Under per-layer
matched compute the gap collapses to +3.8pp at k=8 and **-2.4pp at
k=16** — kmeans++ wins at k=16 — and neither is significant on a paired
t-test at n=5 (t=1.07 and t=-0.56; |t|>2.776 needed). The sign isn't even
consistent across k. The "DA is more robust to initialization" claim does
not survive: DA just searches longer.

**Matching compute has to be done per layer.** The first version of the
parity check matched globally, summing DA's iterations across layers but
averaging kmeans' across layers, then applying that single ratio as
`n_init` everywhere. Those quantities aren't commensurable, and the
result (n_init=1043 at k=8) handed kmeans++ 2-8x more than parity on
every layer. The correct per-layer ratios range from 135x (fc1) to 497x
(conv1), because DA's iteration count is set by its temperature schedule
and barely varies with layer size (444-838 iterations across layers
spanning 6 to 120 units), while kmeans' tracks how fast that layer's
points settle (1-6 iterations).

**Lower inertia does not mean better accuracy.** At k=16, giving kmeans++
the over-generous global budget (n_init=1218) produced *worse* accuracy
than the smaller per-layer-matched budget (80.98% vs. 87.64%). More
restarts selected by inertia means a lower within-cluster SSE, so
minimizing SSE on weight vectors is actively working against preserving
network function. Any clustering-based compression that picks its
solution by inertia is optimizing the wrong objective.

**Unit clustering is the weakest method here.** Its best usable point
(k=32: 98.04% at 3.04x) is dominated by INT8 quantization (99.05% at
3.95x) — better accuracy *and* better compression — and thoroughly beaten
by L0 gates (99.00% at 6.96x). Clustering whole units is simply too
coarse for a network this small.

## Scope and limitations

- **Pruning and quantization numbers are single-seed.** Only the L0 and
  clustering results have been repeated across seeds. The pruning cliff
  and the INT8 number should be treated as one draw each.
- **Single-shot pruning, no fine-tuning.** Magnitude-pruned weights are
  zeroed once and evaluated directly, with no retraining to recover
  accuracy — a deliberate choice to keep it comparable to quantization
  (also single-shot). L0 is different by nature: it fine-tunes, because
  the gates have to be learned.
- **Quantization is simulated (fake quant), not real INT8 inference.**
  Weights are quantized then dequantized back to fp32 for the forward
  pass, which correctly measures the effect on accuracy but does not
  benchmark INT8 inference speed.
- **L0 size accounting is architecture-specific.** The cascading
  calculation in `compressed_size_bytes_l0` assumes this exact LeNet-5
  (padding=2 conv1, two 2x2 pools, 5x5 map into fc1) and would need
  rewriting for another network.
- **n=5 seeds is low.** Non-significant results here mean "not detectable
  at n=5", not "no effect exists".

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

## Usage

Train the baseline model (downloads MNIST automatically on first run):
```
python experiments/train_baseline.py --epochs 10
```

Everything below needs that checkpoint. Each writes a CSV to `results/`.

```
python experiments/eval_baselines.py                                    # pruning + INT8
python experiments/eval_clustering.py                                   # DA vs k-means
python experiments/compute_parity_check.py --k 8 16 --n-seeds 5         # is DA's edge real?
python experiments/train_l0.py --lambdas 0.001 0.005 0.01 0.02 0.05     # L0 sweep
python experiments/l0_seed_verify.py --lambdas 0.005 0.01 0.02 --n-seeds 5
```

Note that `train_l0.py` does not set a seed, so its sweep is one
uncontrolled draw; `l0_seed_verify.py` is the seeded version and is what
the numbers above come from.

## Project structure

```
models/         LeNet-5 architecture
data/           MNIST loading
compression/    Magnitude pruning, INT8 quantization, clustering (DA + k-means), L0 gates
eval/           Shared accuracy/size/compression-ratio metrics
experiments/    Training, evaluation, and verification entry points
results/        Checkpoints and output CSVs (checkpoints gitignored)
```

## Next steps

- Locate the pruning accuracy knee precisely (finer sweep at 75/80/85%
  sparsity).
- Push L0 past lambda=0.02 to find where structured sparsity finally breaks —
  10.93x still only costs 0.26pp, so the cliff hasn't been reached.
- Multi-seed the pruning and INT8 baselines so every number in the first
  table has the same standing as the L0 ones.
- Combine L0 with INT8: they compress along independent axes (fewer
  units vs. fewer bits per weight), so ~7x and ~4x may partly multiply.
