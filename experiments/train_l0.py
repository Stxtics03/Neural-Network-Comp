"""
Fine-tunes the trained baseline with L0-regularized gates on each
hidden layer's units, sweeping lambda (the sparsity penalty weight) to
trace out an accuracy-vs-compression curve. Starts from the trained
baseline rather than from scratch -- the gates' job is to identify
which already-useful units can be removed, not to relearn the task
from nothing.

Gate parameters use 10x the base learning rate -- verified necessary
via a controlled test on synthetic data (at 1x LR, sparsity stayed at
0.000 regardless of lambda; at 10x, sparsity clearly tracked lambda).
Without this, Adam's adaptive per-parameter step size normalizes away
most of the difference between lambda settings.

Compressed-size accounting is STRUCTURAL (whole units removed), unlike
magnitude pruning's per-weight sparse format -- a gated-to-zero filter
or neuron is genuinely absent, which also shrinks the next layer's
input dimension. This cascades through the network (see
compressed_size_bytes_l0) and is architecture-specific to this LeNet-5
(assumes the padding=2/pool-twice conv stack in models/lenet5.py,
which leaves a 5x5 spatial map into fc1).

Usage:
    python experiments/train_l0.py --lambdas 0.0001 0.001 0.005 0.01 0.02 0.05
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.l0_wrapper import L0GatedModel
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from models.lenet5 import build_model

CKPT_PATH = Path("./results/checkpoints/lenet5_best.pt")
EXCLUDE_LAYERS = ["fc3"]  # never gate the classification head -- see l0_wrapper.py


def compressed_size_bytes_l0(sparsity_report: dict, bits_per_weight: int = 32,
                              scale_bytes_per_layer: int = 0) -> int:
    """LeNet-5-specific cascading size accounting: each gated layer's
    surviving-unit count shrinks both its own weight matrix AND the
    next layer's input dimension. Assumes the exact architecture in
    models/lenet5.py (padding=2 on conv1, two 2x2 max-pools -> 5x5
    spatial map feeding fc1).

    bits_per_weight defaults to 32 (fp32). Pass 8 (or 4, 2...) with
    scale_bytes_per_layer=4 to price the same structure stored at that
    bit width with a per-tensor scale -- see l0_int8_combined.py and
    aggressive_stack.py. Sub-byte widths are packed, so a layer's weight
    block is ceil(n_weights * bits / 8) bytes rather than one byte per
    weight. Biases stay fp32 throughout, matching the quantization
    baseline."""
    surviving = {name: info["num_surviving"] for name, info in sparsity_report.items()}

    def wbytes(n_weights: int) -> int:
        return math.ceil(n_weights * bits_per_weight / 8)

    out1 = surviving.get("conv1", 6)
    out2 = surviving.get("conv2", 16)
    out_fc1 = surviving.get("fc1", 120)
    out_fc2 = surviving.get("fc2", 84)

    total = 0
    total += wbytes(out1 * 1 * 5 * 5) + out1 * 4              # conv1: in_channels=1 fixed
    total += wbytes(out2 * out1 * 5 * 5) + out2 * 4           # conv2: in = surviving conv1 channels
    total += wbytes(out_fc1 * (out2 * 5 * 5)) + out_fc1 * 4   # fc1: in = surviving conv2 channels * 5x5
    total += wbytes(out_fc2 * out_fc1) + out_fc2 * 4          # fc2: in = surviving fc1 units
    total += wbytes(10 * out_fc2) + 10 * 4                    # fc3: in = surviving fc2 units, out=10 fixed
    total += 5 * scale_bytes_per_layer                        # one fp32 scale per layer, if quantized
    return total


def load_baseline_state(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}. Run experiments/train_baseline.py first.")
    return torch.load(CKPT_PATH, map_location=device)


def train_one_lambda(lam: float, baseline_state, train_loader, test_loader, device,
                      epochs: int, base_lr: float, gate_lr_mult: float,
                      return_model: bool = False):
    base = build_model().to(device)
    base.load_state_dict(copy.deepcopy(baseline_state))
    model = L0GatedModel(base, exclude_layer_names=EXCLUDE_LAYERS).to(device)

    gate_params = [p for n, p in model.named_parameters() if "log_alpha" in n]
    other_params = [p for n, p in model.named_parameters() if "log_alpha" not in n]
    optimizer = optim.Adam([
        {"params": other_params, "lr": base_lr},
        {"params": gate_params, "lr": base_lr * gate_lr_mult},
    ])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y) + lam * model.total_expected_l0()
            loss.backward()
            optimizer.step()

    model.eval()
    acc = evaluate_accuracy(model, test_loader, device)
    report = model.sparsity_report()
    if return_model:
        return acc, report, model
    return acc, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+",
                         default=[0.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr-mult", type=float, default=10.0)
    parser.add_argument("--out", default="./results/l0_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_l0] device={device}")

    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    dense_base = build_model().to(device)
    dense_base.load_state_dict(baseline_state)
    orig_bytes = dense_model_size_bytes(dense_base)
    baseline_acc = evaluate_accuracy(dense_base, test_loader, device)
    print(f"[train_l0] baseline: accuracy={baseline_acc:.4f}  size={orig_bytes/1024:.1f} KB\n")

    rows = []
    for lam in args.lambdas:
        acc, report = train_one_lambda(
            lam, baseline_state, train_loader, test_loader, device,
            epochs=args.epochs, base_lr=args.base_lr, gate_lr_mult=args.gate_lr_mult,
        )
        size_bytes = compressed_size_bytes_l0(report)
        ratio = compression_ratio(orig_bytes, size_bytes)
        avg_sparsity = sum(r["sparsity"] for r in report.values()) / len(report)

        print(f"lambda={lam:.4f}  accuracy={acc:.4f}  avg_sparsity={avg_sparsity:.3f}  "
              f"size={size_bytes/1024:.1f} KB  ratio={ratio:.2f}x")
        for layer, info in report.items():
            print(f"    {layer}: {info['num_surviving']}/{info['num_units']} units surviving "
                  f"({info['sparsity']:.1%} sparse)")

        rows.append({
            "lambda": lam, "accuracy": round(acc, 4), "avg_sparsity": round(avg_sparsity, 4),
            "size_kb": round(size_bytes / 1024, 2), "compression_ratio": round(ratio, 3),
            **{f"{layer}_surviving": info["num_surviving"] for layer, info in report.items()},
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[train_l0] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
