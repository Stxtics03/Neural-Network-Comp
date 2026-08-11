"""
Compares three clustering methods for weight-unit compression:
  - da: deterministic annealing (soft assignment, cools to hard)
  - kmeans++: k-means with smart (spread-out) initialization
  - kmeans_random: k-means with naive random initialization

kmeans_random is the fairer test of DA's actual selling point --
robustness to bad initialization. kmeans++ already partly solves that
problem itself, so it's a harder baseline to beat; DA beating
kmeans_random more decisively than it beats kmeans++ would be the
expected pattern if the theory (DA's soft assignment smooths out the
bad-init problem) actually holds here.

Multiple seeds per (method, k) from the start -- a single-seed DA vs.
k-means comparison can't distinguish "DA is better" from "that k-means
run got unlucky," and re-running with seeds later (after already
believing a single-run number) is a bad habit worth not forming.

Usage:
    python experiments/eval_clustering.py --n-seeds 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.apply_clustering import apply_clustering, compressed_size_bytes_clustered
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from models.lenet5 import build_model

CKPT_PATH = Path("./results/checkpoints/lenet5_best.pt")
K_VALUES = [4, 8, 16, 32, 64]
METHODS = ["da", "kmeans++", "kmeans_random"]


def load_baseline(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}. Run experiments/train_baseline.py first.")
    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out", default="./results/clustering.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_clustering] device={device}, n_seeds={args.n_seeds}")

    _, test_loader = get_loaders()
    model = load_baseline(device)
    orig_bytes = dense_model_size_bytes(model)
    baseline_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[eval_clustering] baseline accuracy={baseline_acc:.4f}\n")

    rows = []
    for k in K_VALUES:
        accs_by_method = {m: [] for m in METHODS}
        ratio_by_method = {m: [] for m in METHODS}

        for method in METHODS:
            for seed in range(args.n_seeds):
                clustered, info = apply_clustering(model, k=k, method=method, seed=seed)
                clustered = clustered.to(device)
                acc = evaluate_accuracy(clustered, test_loader, device)
                size_bytes = compressed_size_bytes_clustered(info, clustered)
                ratio = compression_ratio(orig_bytes, size_bytes)

                accs_by_method[method].append(acc)
                ratio_by_method[method].append(ratio)
                rows.append({
                    "method": method, "k": k, "seed": seed,
                    "accuracy": round(acc, 4), "compression_ratio": round(ratio, 4),
                })

        line = f"k={k:3d}  "
        for method in METHODS:
            accs = torch.tensor(accs_by_method[method])
            ratio_mean = sum(ratio_by_method[method]) / len(ratio_by_method[method])
            line += f"{method}: {accs.mean():.4f}±{accs.std():.4f} ({ratio_mean:.2f}x)   "
        da_mean = torch.tensor(accs_by_method["da"]).mean()
        krand_mean = torch.tensor(accs_by_method["kmeans_random"]).mean()
        line += f" gap(DA - kmeans_random): {(da_mean - krand_mean):+.4f}"
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "k", "seed", "accuracy", "compression_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[eval_clustering] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
