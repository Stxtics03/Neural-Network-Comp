"""
Tests whether DA's accuracy advantage over kmeans++ (see eval_clustering.py)
is a genuine algorithmic property, or just DA getting far more total
optimization steps per run. Measures DA's real total_iters_ on each
layer, then gives kmeans++ a matched compute budget via multi-restart
(n_init = DA's total iters / kmeans++'s average iters per restart,
picking the best of all restarts by inertia -- standard practice,
what scikit-learn's n_init does) and re-compares accuracy.

If DA's edge shrinks to ~0 once kmeans++ gets equivalent compute, the
"DA is more robust to initialization" story doesn't hold up here --
it would just be "DA searches longer." If the edge survives, that's
real support for the robustness claim.

Usage:
    python experiments/compute_parity_check.py --k 8 16 --n-seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.apply_clustering import apply_clustering, compressed_size_bytes_clustered
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


def measure_da_iters(model, k: int, seed: int) -> int:
    """Sum DA's total_iters_ across every prunable layer for one seed --
    this is DA's real per-run compute budget for this k."""
    total = 0
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        vectors = extract_unit_vectors(module)
        da = DeterministicAnnealingClusterer(k=k, seed=seed)
        da.fit(vectors)
        total += da.total_iters_
    return total


def measure_kmeans_iters_per_restart(model, k: int, seed: int) -> float:
    """Average iters-per-restart for a single kmeans++ run, across layers --
    used to convert DA's iteration budget into an equivalent n_init."""
    iters, count = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        vectors = extract_unit_vectors(module)
        km = KMeansClusterer(k=k, init="kmeans++", seed=seed, n_init=1)
        km.fit(vectors)
        iters += km.total_iters_
        count += 1
    return iters / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cpu")
    _, test_loader = get_loaders()
    model = load_baseline(device)

    for k in args.k:
        da_iters = measure_da_iters(model, k, seed=0)
        km_iters_per_restart = measure_kmeans_iters_per_restart(model, k, seed=0)
        matched_n_init = max(1, round(da_iters / max(km_iters_per_restart, 1)))
        print(f"\n=== k={k} ===")
        print(f"DA total inner-loop iters (summed across layers): {da_iters}")
        print(f"kmeans++ avg iters per single restart: {km_iters_per_restart:.1f}")
        print(f"-> matched n_init for kmeans++: {matched_n_init}")

        da_accs, kpp_default_accs, kpp_matched_accs = [], [], []
        for seed in range(args.n_seeds):
            da_model, _ = apply_clustering(model, k=k, method="da", seed=seed)
            da_accs.append(evaluate_accuracy(da_model.to(device), test_loader, device))

            kpp_default, _ = apply_clustering(model, k=k, method="kmeans++", seed=seed)
            kpp_default_accs.append(evaluate_accuracy(kpp_default.to(device), test_loader, device))

            kpp_matched, _ = apply_clustering(
                model, k=k, method="kmeans++", seed=seed, n_init=matched_n_init,
            )
            kpp_matched_accs.append(evaluate_accuracy(kpp_matched.to(device), test_loader, device))

        da_m, da_s = np.mean(da_accs), np.std(da_accs)
        kd_m, kd_s = np.mean(kpp_default_accs), np.std(kpp_default_accs)
        km_m, km_s = np.mean(kpp_matched_accs), np.std(kpp_matched_accs)

        print(f"DA:                        {da_m:.4f} ± {da_s:.4f}")
        print(f"kmeans++ (n_init=1):       {kd_m:.4f} ± {kd_s:.4f}   gap vs DA: {da_m - kd_m:+.4f}")
        print(f"kmeans++ (n_init={matched_n_init}, matched):  {km_m:.4f} ± {km_s:.4f}   gap vs DA: {da_m - km_m:+.4f}")


if __name__ == "__main__":
    main()
