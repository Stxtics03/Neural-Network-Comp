"""
Verifies the L0 sweep's strongest/most citable points across multiple
seeds -- same reasoning as the earlier clustering compute-parity check:
a single-seed number (especially one likely to end up as a headline
resume figure) needs to survive re-running before it's trustworthy.

Reuses train_l0.py's train_one_lambda() and compressed_size_bytes_l0()
directly rather than reimplementing them, so this is testing the exact
same training procedure, just repeated across seeds.

Usage:
    python experiments/l0_seed_verify.py --lambdas 0.005 0.01 0.02 --n-seeds 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.train_l0 import compressed_size_bytes_l0, load_baseline_state, train_one_lambda
from models.lenet5 import build_model

CKPT_PATH = Path("./results/checkpoints/lenet5_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr-mult", type=float, default=10.0)
    parser.add_argument("--out", default="./results/l0_seed_verify.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[l0_seed_verify] device={device}, n_seeds={args.n_seeds}")

    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    dense_base = build_model().to(device)
    dense_base.load_state_dict(baseline_state)
    orig_bytes = dense_model_size_bytes(dense_base)
    baseline_acc = evaluate_accuracy(dense_base, test_loader, device)
    print(f"[l0_seed_verify] baseline accuracy={baseline_acc:.4f}\n")

    rows = []
    for lam in args.lambdas:
        accs, ratios, sparsities = [], [], []
        for seed in range(args.n_seeds):
            torch.manual_seed(seed)  # controls both DataLoader shuffling and gate sampling
            acc, report = train_one_lambda(
                lam, baseline_state, train_loader, test_loader, device,
                epochs=args.epochs, base_lr=args.base_lr, gate_lr_mult=args.gate_lr_mult,
            )
            size_bytes = compressed_size_bytes_l0(report)
            ratio = compression_ratio(orig_bytes, size_bytes)
            avg_sparsity = sum(r["sparsity"] for r in report.values()) / len(report)

            accs.append(acc)
            ratios.append(ratio)
            sparsities.append(avg_sparsity)
            rows.append({
                "lambda": lam, "seed": seed, "accuracy": round(acc, 4),
                "avg_sparsity": round(avg_sparsity, 4), "compression_ratio": round(ratio, 3),
            })

        acc_arr, ratio_arr = np.array(accs), np.array(ratios)
        print(f"lambda={lam:.4f}  accuracy={acc_arr.mean():.4f}±{acc_arr.std():.4f}  "
              f"ratio={ratio_arr.mean():.2f}x±{ratio_arr.std():.2f}  "
              f"(range: {acc_arr.min():.4f}-{acc_arr.max():.4f} acc, "
              f"{ratio_arr.min():.2f}x-{ratio_arr.max():.2f}x ratio)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "seed", "accuracy", "avg_sparsity", "compression_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[l0_seed_verify] wrote {len(rows)} rows to {out_path}")
    print("[l0_seed_verify] check the ± spread above -- if it's small relative to the gaps "
          "between lambda values, the sweep's shape is real, not seed noise.")


if __name__ == "__main__":
    main()
