"""
Tests whether DA's accuracy advantage over kmeans++ (see eval_clustering.py)
is a genuine algorithmic property, or just DA getting far more total
optimization steps per run.

Compute is matched PER LAYER. An earlier version of this check matched
globally -- it summed DA's iterations across layers but averaged
kmeans' across layers, then applied that single ratio as n_init
everywhere. Those are not commensurable, and the resulting n_init
(1043 at k=8) handed kmeans++ 2-8x more than parity on every layer.
Per-layer ratios are what's fair, and they vary a lot (497x on conv1
down to 135x on fc1) because DA's iteration count is set by its
temperature schedule and barely depends on layer size, while kmeans'
depends on how quickly that layer's points settle.

Three kmeans++ arms are reported:
  - n_init=1        the naive baseline
  - per-layer       true parity, the number this check exists to produce
  - global (legacy) the old over-generous budget, kept because "kmeans++
                    still loses at several times parity" is worth showing

If DA's edge shrinks to ~0 once kmeans++ gets equivalent compute, the
"DA is more robust to initialization" story doesn't hold up here -- it
would just be "DA searches longer." If the edge survives, that's real
support for the robustness claim.

Usage:
    python experiments/compute_parity_check.py --k 8 16 --n-seeds 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.apply_clustering import apply_clustering
from compression.deterministic_annealing import DeterministicAnnealingClusterer
from compression.kmeans import KMeansClusterer
from compression.unit_vectors import extract_unit_vectors
from data.loaders import get_loaders
from eval.metrics import evaluate_accuracy
from models.lenet5 import build_model

CKPT_PATH = Path("./results/checkpoints/lenet5_best.pt")


def load_baseline(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}. Run experiments/train_baseline.py first.")
    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model


def clusterable_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            yield name, module


def measure_per_layer_budget(model, k: int, seed: int) -> dict[str, dict]:
    """For each layer: DA's real iteration count, kmeans' iterations for
    one restart, and the n_init that equalizes them."""
    budget = {}
    for name, module in clusterable_layers(model):
        vectors = extract_unit_vectors(module)

        da = DeterministicAnnealingClusterer(k=k, seed=seed)
        da.fit(vectors)

        km = KMeansClusterer(k=k, init="kmeans++", seed=seed, n_init=1)
        km.fit(vectors)

        budget[name] = {
            "da_iters": da.total_iters_,
            "km_iters_per_restart": km.total_iters_,
            "matched_n_init": max(1, round(da.total_iters_ / max(km.total_iters_, 1))),
            "n_units": vectors.shape[0],
        }
    return budget


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-statistic for a-b. Both arms share a seed per run, so
    pairing removes the between-seed variance an unpaired test leaves in.
    Returns (mean_difference, t). df = n-1; |t| > 2.776 is p<0.05 at n=5."""
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), float("inf") if d.mean() != 0 else 0.0
    return float(d.mean()), float(d.mean() / (sd / np.sqrt(n)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", default="./results/parity_check.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    _, test_loader = get_loaders()
    model = load_baseline(device)

    rows = []
    for k in args.k:
        budget = measure_per_layer_budget(model, k, seed=0)
        matched = {name: {"n_init": b["matched_n_init"]} for name, b in budget.items()}

        da_total = sum(b["da_iters"] for b in budget.values())
        km_mean = np.mean([b["km_iters_per_restart"] for b in budget.values()])
        legacy_n_init = max(1, round(da_total / max(km_mean, 1)))

        print(f"\n=== k={k} ===")
        print(f"{'layer':8} {'units':>6} {'DA iters':>9} {'KM iters':>9} {'matched n_init':>15}")
        for name, b in budget.items():
            print(f"{name:8} {b['n_units']:6} {b['da_iters']:9} "
                  f"{b['km_iters_per_restart']:9} {b['matched_n_init']:15}")
        print(f"legacy global n_init (over-generous, kept for comparison): {legacy_n_init}")

        arms = {
            "da": lambda s: apply_clustering(model, k=k, method="da", seed=s),
            "kmeans++_n_init=1": lambda s: apply_clustering(model, k=k, method="kmeans++", seed=s),
            "kmeans++_matched_per_layer": lambda s: apply_clustering(
                model, k=k, method="kmeans++", seed=s, per_layer_kwargs=matched),
            "kmeans++_matched_global_legacy": lambda s: apply_clustering(
                model, k=k, method="kmeans++", seed=s, n_init=legacy_n_init),
        }

        accs = {name: [] for name in arms}
        for seed in range(args.n_seeds):
            for arm_name, build in arms.items():
                clustered, _ = build(seed)
                acc = evaluate_accuracy(clustered.to(device), test_loader, device)
                accs[arm_name].append(acc)
                rows.append({"k": k, "arm": arm_name, "seed": seed, "accuracy": round(acc, 4)})

        da_arr = np.array(accs["da"])
        print()
        print(f"DA: {da_arr.mean():.4f} +/- {da_arr.std():.4f}")
        for arm_name in list(arms)[1:]:
            arr = np.array(accs[arm_name])
            mean_diff, t = paired_t(da_arr, arr)
            sig = "significant" if abs(t) > 2.776 else "NOT significant"
            print(f"{arm_name}: {arr.mean():.4f} +/- {arr.std():.4f}   "
                  f"paired gap vs DA: {mean_diff:+.4f}  t={t:+.2f} ({sig} at n=5)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "arm", "seed", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[compute_parity_check] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
