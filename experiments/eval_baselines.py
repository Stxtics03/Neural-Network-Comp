"""
Loads your trained baseline checkpoint and evaluates:
  1. Magnitude pruning at several sparsity levels
  2. INT8 quantization
against it, reporting accuracy and compression ratio for each --
your first real, comparable compression numbers.

Usage:
    python experiments/eval_baselines.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.magnitude_pruning import (
    actual_sparsity, compressed_size_bytes_pruned, prune_model,
)
from compression.quantization import compressed_size_bytes_int8, quantize_model_int8
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from models.lenet5 import build_model

CKPT_PATH = Path("./results/checkpoints/lenet5_best.pt")
SPARSITY_LEVELS = [0.3, 0.5, 0.7, 0.9]


def load_baseline(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Run experiments/train_baseline.py first."
        )
    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_baselines] device={device}")

    _, test_loader = get_loaders()
    model = load_baseline(device)

    orig_bytes = dense_model_size_bytes(model)
    baseline_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[eval_baselines] baseline: accuracy={baseline_acc:.4f}  size={orig_bytes/1024:.1f} KB (dense fp32)\n")

    rows = [{
        "method": "baseline", "setting": "none",
        "accuracy": round(baseline_acc, 4), "size_kb": round(orig_bytes / 1024, 2),
        "compression_ratio": 1.0,
    }]

    print("--- Magnitude pruning ---")
    for sparsity in SPARSITY_LEVELS:
        pruned = prune_model(model, sparsity).to(device)
        acc = evaluate_accuracy(pruned, test_loader, device)
        real_sparsity = actual_sparsity(pruned)
        size_bytes = compressed_size_bytes_pruned(pruned)
        ratio = compression_ratio(orig_bytes, size_bytes)
        print(f"  target_sparsity={sparsity:.1f}  actual_sparsity={real_sparsity:.3f}  "
              f"accuracy={acc:.4f}  size={size_bytes/1024:.1f} KB  ratio={ratio:.2f}x")
        rows.append({
            "method": "magnitude_pruning", "setting": f"sparsity={sparsity}",
            "accuracy": round(acc, 4), "size_kb": round(size_bytes / 1024, 2),
            "compression_ratio": round(ratio, 2),
        })

    print("\n--- INT8 quantization ---")
    quantized = quantize_model_int8(model).to(device)
    acc = evaluate_accuracy(quantized, test_loader, device)
    size_bytes = compressed_size_bytes_int8(quantized)
    ratio = compression_ratio(orig_bytes, size_bytes)
    print(f"  accuracy={acc:.4f}  size={size_bytes/1024:.1f} KB  ratio={ratio:.2f}x")
    rows.append({
        "method": "int8_quantization", "setting": "per-tensor symmetric",
        "accuracy": round(acc, 4), "size_kb": round(size_bytes / 1024, 2),
        "compression_ratio": round(ratio, 2),
    })

    out_path = Path("./results/baselines.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "setting", "accuracy", "size_kb", "compression_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[eval_baselines] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
