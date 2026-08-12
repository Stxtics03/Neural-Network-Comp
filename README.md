# Neural Network Compression

Compressing LeNet-5 on MNIST with five techniques — magnitude pruning,
INT8/low-bit quantization, unit clustering, L0-regularized structured
sparsity, and quantization-aware training — with a bias toward checking
results rather than reporting them. Every headline number here has been
either repeated across seeds, cross-validated against an independent
implementation, or materialized as a real artifact on disk.

Two of the results are negative. They're kept because they're the
useful part.

## Headline results

| Result | Number | Evidence |
|---|---|---|
| Best compression within 1pp | **147.82x** at 98.49% (0.54pp loss) | L0 + INT4 QAT, 2 seeds |
| Best near-free config | **81.51x** at 98.93% (0.11pp loss) | L0 + INT4 QAT, 2 seeds |
| Best without QAT | **26.20x** at 98.97% (0.07pp loss) | L0 + INT8, 3 seeds |
| Highest sparsity at ~1% error | **95%** weights zero, 1.03% error | iterative prune + fine-tune, 3 seeds |
| Inference speedup from 10.4x fewer params | **none** (1.01x / 0.90x) | measured on materialized model |
| Deterministic annealing's edge over k-means | **none** under matched compute | paired t-test, n=5 |

### Résumé-accurate phrasings

Wording that the data here actually supports:

- Compressed LeNet-5 **147x** (241 KB → 1.6 KB) at **0.54pp** accuracy
  loss (98.49% vs. 99.04% dense) by stacking learned L0 structured
  sparsity with INT4 quantization-aware training; storage accounting
  cross-validated against an independent implementation and against a
  materialized on-disk model.
- Reached **95% weight sparsity at 1.03% +/- 0.06pp MNIST test error**
  (3 seeds) via iterative magnitude pruning with fine-tuning — recovering
  **+23pp** over the same pruning without retraining.
- Disproved a **+16pp** apparent advantage of deterministic-annealing
  clustering over k-means by matching per-layer optimization budgets; the
  gap fell to **-2.4pp** and was not statistically significant (paired
  t-test, n=5).
- Showed **10.4x fewer parameters produced no CPU inference speedup**
  (1.01x at batch 1, 0.90x at batch 64), demonstrating that FLOP
  reduction does not imply wall-clock gains at small model scale.

Note the distinction between **size** and **parameter count**: 147.82x is
storage compression. The parameter count itself falls about 20x (that's
the structural L0 part); INT4 shrinks bits-per-parameter, not the number
of parameters. The two are not interchangeable.

## What this project does NOT show

Stated explicitly, because these are the claims a compression project is
most often assumed to support:

- **No inference speedup.** Measured, and it isn't there (see above).
  Unstructured sparsity additionally yields no speedup on standard
  hardware without specialized sparse kernels, so the 95%-sparsity result
  is not a latency result either.
- **No real INT8/INT4 inference.** Quantization is simulated
  (quantize-then-dequantize). It measures accuracy impact correctly; it
  does not benchmark integer kernels.
- **No novel algorithm.** Deterministic annealing is Rose (1998),
  hard-concrete L0 gates are Louizos et al. (2017), iterative pruning is
  Han et al. (2015). This is careful reimplementation and evaluation.
- **Nothing multimodal, no graph models, no Markov chains.** MNIST
  images only.
- **One architecture, one dataset.** Every conclusion is about LeNet-5 on
  MNIST. The size accounting is literally hardcoded to this network's
  shape.

## Results

Baseline: **99.04%** test accuracy, 241.0 KB dense fp32.

### Pruning and quantization, single-shot (single seed)

| Method | Setting | Accuracy | Size | Compression |
|---|---|---|---|---|
| Baseline | — | 99.04% | 241.0 KB | 1.00x |
| Magnitude pruning | 30% sparsity | 99.06% | 252.9 KB | **0.95x** |
| Magnitude pruning | 50% sparsity | 99.02% | 180.9 KB | 1.33x |
| Magnitude pruning | 70% sparsity | 98.30% | 108.9 KB | 2.21x |
| Magnitude pruning | 90% sparsity | 63.66% | 36.9 KB | 6.54x |
| INT8 quantization | per-tensor symmetric | 99.05% | 61.0 KB | 3.95x |

### Iterative pruning WITH fine-tuning (3 seeds)

A different protocol from the table above — retraining between pruning
steps. Not comparable to the single-shot numbers and never to be quoted
as one series.

| Sparsity | Before fine-tune | After fine-tune | Error | Recovered | Size | Compression |
|---|---|---|---|---|---|---|
| 50% | 99.02% | 99.18% +/- 0.06 | 0.82% | +0.16pp | 180.9 KB | 1.33x |
| 70% | 98.31% | 99.18% +/- 0.14 | 0.82% | +0.88pp | 108.9 KB | 2.21x |
| 80% | 97.72% | 99.17% +/- 0.04 | 0.83% | +1.45pp | 72.9 KB | 3.31x |
| 90% | 73.68% | 99.08% +/- 0.05 | 0.92% | +25.40pp | 36.9 KB | 6.54x |
| 93% | 83.70% | 99.02% +/- 0.08 | 0.98% | +15.31pp | 26.1 KB | 9.25x |
| **95%** | 75.81% | **98.97% +/- 0.06** | **1.03%** | +23.16pp | 18.9 KB | 12.77x |

Seed 0 alone reached 0.95% error at 95% sparsity; the 3-seed mean is
1.03%. Reporting the single best seed would have overstated the result by
0.08pp — small in absolute terms, but it is the difference between
"beats 1% error" and "lands just above it".

### L0 structured sparsity (5 seeds)

| lambda | Accuracy | Compression |
|---|---|---|
| 0.005 | 99.01% +/- 0.09pp | 4.26x +/- 0.15 |
| 0.01 | 99.00% +/- 0.03pp | 6.96x +/- 0.33 |
| 0.02 | 98.78% +/- 0.06pp | 10.93x +/- 0.67 |

### L0 + INT8 (3 seeds)

| lambda | L0 alone | L0 + INT8 | Size | Quantization cost |
|---|---|---|---|---|
| 0.005 | 99.00% @ 4.25x | 99.02% @ 16.69x | 14.4 KB | -0.02pp |
| 0.01 | 98.98% @ 6.70x | 98.97% @ 26.20x | 9.2 KB | +0.00pp |
| 0.02 | 98.74% @ 11.06x | 98.76% @ 42.91x | 5.6 KB | -0.02pp |

### Pushing the stack to its limit (2 seeds, post-training quantization)

Pass mark is 1 percentage point of accuracy loss, i.e. >= 98.04%.

| lambda | fp32 | INT8 | INT4 | INT3 | INT2 |
|---|---|---|---|---|---|
| 0.02 | 98.76% @ 10.63x | 98.79% @ 41.27x | **98.22% @ 79.55x** | 94.84% @ 103.47x | 9.95% @ 148.27x |
| 0.05 | 98.28% @ 20.46x | **98.30% @ 78.18x** | 97.70% @ 147.82x | 91.66% @ 189.83x | 10.03% @ 266.05x |
| 0.1 | 60.17% @ 214.44x | 59.96% @ 670.40x | 48.19% @ 1058x | 30.37% @ 1230x | 10.30% @ 1489x |

Bold entries are the best passing configs. INT2 destroys the network
outright (~10% = random guessing over 10 classes), and lambda=0.1 is past
the structural cliff before any quantization is applied.

### Quantization-aware training (2 seeds)

| Config | PTQ accuracy | QAT accuracy | Recovered | Compression | Verdict |
|---|---|---|---|---|---|
| lambda=0.02, INT4 | 98.22% (-0.81pp) | **98.93% (-0.11pp)** | +0.71pp | **81.51x** | pass |
| lambda=0.05, INT4 | 97.70% (-1.34pp) | **98.49% (-0.54pp)** | +0.80pp | **147.82x** | pass |
| lambda=0.05, INT3 | 91.66% (-7.38pp) | 97.77% (-1.27pp) | **+6.11pp** | 189.83x | fail by 0.27pp |

The lambda=0.02 ratio is 81.51x rather than the 79.55x its PTQ counterpart
scored because INT4 rounding deleted whole units during fine-tuning,
making the real model smaller than the pre-QAT report claimed; the run
re-prices from the counts actually observed.

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
sparsity the "compressed" model is *larger* than dense (0.95x). A stored
nonzero costs 4 bytes (value) + 2 bytes (index) = 6, vs. 4 for a dense
fp32 weight, so a naive sparse format only wins past `4/6 ~= 33%`
sparsity. Reporting sparsity percentages without converting to a storage
format hides this entirely.

**Fine-tuning is doing most of the work at high sparsity, and it should
be labeled as such.** Single-shot pruning collapses to 63.66% at 90%
sparsity; with retraining between steps the same sparsity reaches 99.08%.
Averaged over 3 seeds the rescue is +25.40pp at 90% and +23.16pp at 95%,
and it is what the extra training bought, not what pruning achieved.
Quoting a fine-tuned sparsity figure next to single-shot numbers from the
same repo would be misleading, which is why they are in separate tables
here.

**Structured L0 sparsity is the strongest single method, and it holds
up.** At lambda=0.01 it gives 6.96x for 0.04pp, against INT8's 3.95x and
magnitude pruning's 2.21x at comparable accuracy. Across 5 seeds the
accuracy spread is +/-0.03 to +/-0.09pp while gaps between lambda
settings are an order of magnitude larger, and compression gaps are 4-10
standard deviations apart — the curve's shape is a property of lambda,
not the seed.

**L0 and quantization stack almost perfectly, and the residual is the
accounting being honest.** INT8 on top of L0 costs -0.02/+0.00/-0.02pp
across lambda=0.005/0.01/0.02 — measured paired, same model before and
after. The plausible worry was that a network stripped of half its units
has less redundancy to absorb rounding error, so quantization should hurt
*more* at high sparsity. It doesn't. Meanwhile INT8's multiplier decays
3.93x -> 3.91x -> 3.88x against 3.95x standalone, because biases stay
fp32 under both schemes: a layer's bias count falls linearly with
surviving units while its weight count falls roughly quadratically, so
the fixed bias cost becomes a larger share of an ever-smaller model.
Perfect multiplication would have been evidence of a double-counting bug.

**Quantization-aware training buys back most of INT4's loss.** Post-hoc
INT4 at lambda=0.05 costs 1.34pp — enough to miss a 1pp budget at
147.82x. Fine-tuning *through* a straight-through-estimator quantizer
recovers +0.80pp, landing at 0.54pp. The weights adapt to the grid they
will be stored on rather than being rounded onto it blind.

**INT3's failure is a rounding problem, not a representational one.**
Under post-training quantization INT3 looks hopeless — 4.20pp at
lambda=0.02, 7.38pp at lambda=0.05 — which reads like a hard floor
between INT4 (7 levels per sign) and INT3 (3 levels). It isn't. QAT
recovers **+6.11pp** of that 7.38pp gap, landing INT3 at 1.27pp and
189.83x, missing a 1pp budget by 0.27pp. Three levels per sign *can*
encode this network; the weights just have to be trained knowing it.
INT2 (1 level) is genuinely dead at ~10% accuracy, so the real floor is
one bit lower than PTQ suggests.

**QAT collapses seed variance at INT4 — and inflates it at INT3.** At
INT4 the two seeds' spread falls from 0.81pp to 0.02pp (lambda=0.02) and
0.14pp to 0.03pp (lambda=0.05): rounding a finished model onto a coarse
grid depends on exactly where its weights happened to land, and training
into the grid removes that dependence. At INT3 the pattern reverses,
0.26pp before to 0.76pp after. The plausible reading is that INT3 sits at
the edge of what the network can represent, so the outcome depends on
which solution the fine-tuning trajectory happens to find rather than on
the rounding — which is also a reason to treat the INT3 number as the
least trustworthy in this table, at n=2.

**Coarse quantization can delete whole units by itself.** During INT4
QAT, a surviving unit whose weights are all small relative to its
layer's per-tensor maximum rounded entirely to zero, changing the
surviving-unit counts underneath the size accounting. This is a mild
argument for per-channel scales, which would give each unit its own
range. The experiment detects it and re-prices from observed counts;
it treats a *revived* unit as a hard error, since that direction
overstates compression, while a newly dead unit only understates it.

**FLOP reduction did not produce a speedup.** The materialized narrow
model — (6,16,120,84) -> (6,11,14,10), 61,706 -> 5,941 parameters, 10.39x
fewer — ran at 1.01x at batch 1 and **0.90x (slower) at batch 64**. At
LeNet-5's scale per-op framework overhead dominates arithmetic, and
narrow layers fall off SIMD-friendly alignment, so smaller matrices can
run worse. Any "Nx smaller therefore Nx faster" claim needs a stopwatch,
not a parameter count.

**Deterministic annealing's advantage over k-means was a compute
artifact.** DA appeared to beat kmeans++ by +14.3pp at k=8 and +16.2pp at
k=16. But DA runs far more inner-loop iterations per fit. Given a
per-layer matched restart budget, the gap collapses to +3.8pp at k=8 and
**-2.4pp at k=16** — kmeans++ wins there — and neither is significant on
a paired t-test at n=5 (t=1.07, t=-0.56; |t|>2.776 needed). The sign
isn't even consistent across k. DA just searches longer.

**Matching compute has to be done per layer.** The first version of that
check summed DA's iterations across layers but averaged kmeans' across
layers, then applied the single ratio as `n_init` everywhere — handing
kmeans++ 2-8x more than parity. Correct per-layer ratios span 135x (fc1)
to 497x (conv1), because DA's iteration count is set by its temperature
schedule and barely varies with layer size (444-838 iterations across
layers of 6 to 120 units) while kmeans' tracks how fast that layer's
points settle (1-6 iterations).

**Lower inertia does not mean better accuracy.** At k=16, the
over-generous global budget (n_init=1218) scored *worse* than the smaller
per-layer-matched one (80.98% vs. 87.64%). More restarts selected by
inertia means lower within-cluster SSE — which is working against
preserving network function. Clustering-based compression that selects by
inertia is optimizing the wrong objective.

**Unit clustering is the weakest method here.** Its best usable point
(k=32: 98.04% at 3.04x) is dominated by INT8 (99.05% at 3.95x) on both
axes, and buried by L0+INT8 (98.97% at 26.20x). Clustering whole units is
too coarse for a network this small.

## Verification

Numbers that could silently drift are pinned by checks that fail loudly:

- **Size accounting identity.** At 0% sparsity `compressed_size_bytes_l0`
  returns 246,824 bytes — exactly the dense model — and priced as INT8
  returns 3.953x, reproducing the 3.95x that `eval_baselines.py` computes
  through a separate code path.
- **Gate folding.** Hard-concrete gates are not binary at eval:
  `z = clamp(sigmoid(log_alpha)*1.2 - 0.1, 0, 1)` equals 1.0 only when
  `log_alpha >= 2.4`. Since `z*(Wx+b) == (zW)x + (zb)`, gates are folded
  into weights, which is what lets the accounting store nothing per unit.
  Every run asserts folding is accuracy-neutral to 1e-4.
- **Materialized model.** `materialize_pruned.py` builds the genuinely
  narrower network, transplants surviving weights through the cascade
  (each surviving conv2 channel owns 25 consecutive fc1 columns, not
  one), and asserts identical accuracy — drift was exactly 0.00e+00. A
  wrong index mapping breaks accuracy rather than quietly reporting a
  smaller number.
- **Structural sparsity under QAT.** Surviving-unit counts are re-derived
  after fine-tuning and compared; revival is a hard error.
- **Selection on means, not rows.** Best-config selection aggregates per
  configuration across seeds. Picking the best individual seed row would
  report whichever run got lucky — at lambda=0.02/INT4 that would have
  turned a 79.55x @ 98.22% result into an 84.67x @ 98.63% one.

## Scope and limitations

- **Seed counts are low**: 5 for L0 and clustering, 3 for L0+INT8 and
  iterative pruning, 2 for the stack sweeps and QAT. Non-significant
  results mean "not detectable at this n", not "no effect". The INT3 QAT
  number is the shakiest, at n=2 with a 0.76pp spread between seeds.
- **Quantization is simulated**, not real integer inference.
- **L0 size accounting is architecture-specific** — the cascade in
  `compressed_size_bytes_l0` hardcodes this LeNet-5 (padding=2 conv1,
  two 2x2 pools, 5x5 map into fc1).
- **The iterative-pruning schedule was chosen after a shorter run missed
  the target.** The shorter run (2 epochs/step, no 93% step) reached 95%
  sparsity at 1.41% error; the reported schedule uses 4 epochs/step and
  an added 93% step. Both are in `results/`. The schedule was not
  re-tuned after seeing the 3-seed result.
- **Latency was measured on a CPU** under a Python/PyTorch runtime. A
  different runtime, or a model large enough to be compute-bound, could
  give a different answer.

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

## Usage

Train the baseline (downloads MNIST on first run):
```
python experiments/train_baseline.py --epochs 10
```

Everything below needs that checkpoint; each writes a CSV to `results/`.

```
python experiments/eval_baselines.py                                   # pruning + INT8
python experiments/eval_clustering.py                                  # DA vs k-means
python experiments/compute_parity_check.py --k 8 16 --n-seeds 5        # is DA's edge real?
python experiments/train_l0.py --lambdas 0.001 0.005 0.01 0.02 0.05    # L0 sweep
python experiments/l0_seed_verify.py --lambdas 0.005 0.01 0.02 --n-seeds 5
python experiments/l0_int8_combined.py --lambdas 0.005 0.01 0.02 --n-seeds 3
python experiments/aggressive_stack.py --lambdas 0.02 0.05 0.1 --bits 32 8 4 3 2
python experiments/qat_stack.py --configs 0.05:4 0.02:4 --n-seeds 2
python experiments/iterative_prune.py --schedule 0.5 0.7 0.8 0.9 0.93 0.95 --epochs-per-step 4
python experiments/materialize_pruned.py --lam 0.02        # real narrow model + latency
```

`train_l0.py` does not set a seed, so its sweep is one uncontrolled draw;
`l0_seed_verify.py` is the seeded version and is the source of the L0
numbers above.

## Project structure

```
models/         LeNet-5 (width-parameterizable, for materializing pruned nets)
data/           MNIST loading
compression/    Pruning, quantization (any bit width), clustering, L0 gates
eval/           Shared accuracy/size/compression-ratio metrics
experiments/    Training, evaluation, and verification entry points
results/        Output CSVs and checkpoints (checkpoints gitignored)
```

## Next steps

- **Per-channel quantization scales.** Would likely push INT3 from
  unusable toward usable, and would stop coarse quantization from
  deleting whole units.
- **Multi-seed the INT3 QAT result**, currently n=2 with a 0.76pp spread
  — the weakest evidence behind any number here, and the one closest to
  changing a verdict (it misses the 1pp bar by 0.27pp).
- **Combine unstructured pruning with the L0+QAT stack.** They are
  currently separate branches; 95% weight sparsity inside an already
  structurally-pruned INT4 model is untested.
- **Benchmark on a compute-bound model** to find where FLOP reduction
  does start converting into wall-clock speedup.
