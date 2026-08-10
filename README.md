# Neural Network Compression

Compressing LeNet-5 for MNIST digit classification using magnitude pruning
and post-training INT8 quantization, with a focus on getting the storage
accounting right rather than just reporting raw sparsity or bit-width.

## Results

Baseline: **99.04% test accuracy**, 241 KB (dense fp32 weights).

| Method | Setting | Accuracy | Size | Compression |
|---|---|---|---|---|
| Baseline | — | 99.04% | 241.0 KB | 1.00x |
| Magnitude pruning | 30% sparsity | 99.06% | 252.9 KB | **0.95x** |
| Magnitude pruning | 50% sparsity | 99.02% | 180.9 KB | 1.33x |
| Magnitude pruning | 70% sparsity | 98.30% | 108.9 KB | 2.21x |
| Magnitude pruning | 90% sparsity | 63.66% | 36.9 KB | 6.54x |
| INT8 quantization | per-tensor symmetric | 99.05% | 61.0 KB | 3.95x |

### Findings

**Sparse storage has a break-even point, and it's easy to miss.** At 30%
sparsity the "compressed" model is actually *larger* than dense storage
(0.95x). Each stored nonzero weight costs 4 bytes (value) + 2 bytes (index)
= 6 bytes, vs. 4 bytes for a dense fp32 weight — so a naive sparse format
only wins once sparsity exceeds `4/6 ≈ 33%`. Below that, you're paying
index overhead for too few actual zeros to offset it. This is a common
gotcha in pruning papers that report sparsity percentages without
converting them to an actual storage format.

**There's a sharp accuracy cliff between 70% and 90% sparsity.** Accuracy
is essentially flat from 0%→70% (99.04%→98.30%, a 0.74pp drop for more
than 2x compression), then collapses to 63.66% by 90% — a 34.6pp drop in
one step. The exact knee between 70-90% hasn't been located yet (see
Next Steps).

**INT8 quantization dominates pruning in this regime.** 3.95x compression
for a ~0pp accuracy change (99.05% vs. 99.04% baseline, within run-to-run
noise), a better ratio than pruning achieves even at 70% sparsity (2.21x,
-0.74pp) — and pruning needs to reach 90% sparsity before it beats
quantization's compression ratio at all, at which point it costs 34pp of
accuracy that quantization doesn't.

## Scope and limitations

- **Single-shot pruning, no fine-tuning.** Weights are zeroed once and
  evaluated directly — no retraining afterward to recover accuracy. This
  is a deliberate choice to keep the pruning number directly comparable
  to quantization (also single-shot), rather than conflating "pruning" with
  "pruning + extra training."
- **Quantization is simulated (fake quant), not real INT8 inference.**
  Weights are quantized then immediately dequantized back to fp32 for the
  forward pass, which correctly measures quantization's effect on
  accuracy but doesn't benchmark actual INT8 inference speed (no INT8
  GEMM kernels are used here).
- **Single run, no seed sweep yet.** These numbers are from one trained
  model and one evaluation pass per setting — not yet verified for
  run-to-run variance.

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

Evaluate pruning and quantization against the trained checkpoint:
```
python experiments/eval_baselines.py
```
Writes results to `results/baselines.csv`.

## Project structure

```
models/         LeNet-5 architecture
data/           MNIST loading
compression/    Magnitude pruning, INT8 quantization
eval/           Shared accuracy/size/compression-ratio metrics
experiments/    Training and evaluation entry points
results/        Checkpoints and output CSVs (checkpoints gitignored)
```

## Next steps

- Locate the pruning accuracy knee precisely (finer sweep at 75/80/85%
  sparsity).
- Deterministic-annealing-based weight clustering, compared against
  k-means, as a third compression method.
- Multi-seed reproduction of the strongest findings above before treating
  them as more than single-run results.
