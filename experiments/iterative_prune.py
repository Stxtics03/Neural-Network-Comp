"""
Iterative magnitude pruning WITH fine-tuning, in the style of Han et al.
(2015): prune to a moderate sparsity, retrain to recover, prune harder,
retrain again, and so on up to the target.

This is deliberately a DIFFERENT PROTOCOL from the pruning numbers in
results/baselines.csv, which are single-shot with no retraining. Those
exist to be comparable against the other single-shot methods
(quantization, clustering); these exist to answer a separate question:
how far can unstructured sparsity go when retraining is allowed? The
two are not interchangeable and should never be quoted as one series --
single-shot pruning collapses to 63.66% at 90% sparsity, and if
fine-tuning rescues that, the rescue is what the retraining bought, not
what pruning achieved.

The target this was built to test: 95% sparsity at about 1.0% test
error (i.e. ~99.0% accuracy).

Pruned weights are held at exactly zero for the rest of training by
re-applying the mask after every optimizer step -- without that, Adam's
momentum immediately drags them off zero and the sparsity evaporates.

Usage:
    python experiments/iterative_prune.py --schedule 0.5 0.7 0.8 0.9 0.95 --epochs-per-step 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.magnitude_pruning import actual_sparsity, compressed_size_bytes_pruned, prune_model
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.train_l0 import load_baseline_state
from models.lenet5 import build_model


def current_masks(model: nn.Module) -> dict:
    return {m: (m.weight.data != 0).float()
            for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))}


def finetune(model: nn.Module, masks: dict, train_loader, device, epochs: int, lr: float):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            with torch.no_grad():
                for module, mask in masks.items():
                    module.weight.data *= mask  # keep pruned weights pinned at zero
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--epochs-per-step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="./results/iterative_prune.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    model = build_model().to(device)
    model.load_state_dict(baseline_state)
    orig_bytes = dense_model_size_bytes(model)
    baseline_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[iterative] device={device}  baseline={baseline_acc:.4f} "
          f"({(1-baseline_acc)*100:.2f}% error)\n")

    rows = []
    for target in args.schedule:
        model = prune_model(model, sparsity=target)
        acc_before = evaluate_accuracy(model, test_loader, device)

        masks = current_masks(model)
        model = finetune(model, masks, train_loader, device, args.epochs_per_step, args.lr)
        acc_after = evaluate_accuracy(model, test_loader, device)

        sparsity = actual_sparsity(model)
        size = compressed_size_bytes_pruned(model)
        ratio = compression_ratio(orig_bytes, size)

        print(f"sparsity={sparsity:.1%}  before_finetune={acc_before:.4f}  "
              f"after={acc_after:.4f} ({(1-acc_after)*100:.2f}% error)  "
              f"recovered={(acc_after-acc_before)*100:+.2f}pp  "
              f"size={size/1024:.1f} KB ({ratio:.2f}x)")

        rows.append({
            "target_sparsity": target, "actual_sparsity": round(sparsity, 4),
            "acc_before_finetune": round(acc_before, 4),
            "acc_after_finetune": round(acc_after, 4),
            "error_pct_after": round((1 - acc_after) * 100, 2),
            "size_kb": round(size / 1024, 2), "compression_ratio": round(ratio, 2),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[iterative] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
