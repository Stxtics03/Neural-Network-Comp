"""
Pushes the L0 + quantization stack as far as it goes, to find where it
finally breaks. l0_int8_combined.py showed INT8-on-top-of-L0 costs
nothing at up to 42.91x; this sweeps higher lambda AND lower bit widths
to locate the actual frontier.

The target this was built to test: is 100x compression reachable at
under 1 percentage point of accuracy loss? Baseline is 99.04%, so the
bar is 98.04%.

Each (lambda, seed) is trained ONCE and the same folded model is then
quantized at every bit width, so bit-width comparisons are paired --
differences between bit widths can't be seed noise.

Usage:
    python experiments/aggressive_stack.py --lambdas 0.02 0.05 0.1 --bits 32 8 4 3 2 --n-seeds 2
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.quantization import _quantize_dequantize_tensor
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.l0_int8_combined import fold_gates_into_weights
from experiments.train_l0 import compressed_size_bytes_l0, load_baseline_state, train_one_lambda
from models.lenet5 import build_model

LOSS_BUDGET_PP = 1.0  # "under 1% accuracy loss", read as 1 percentage point absolute


def quantized_copy(model, bits: int):
    q = copy.deepcopy(model)
    if bits >= 32:
        return q
    for m in q.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            m.weight.data = _quantize_dequantize_tensor(m.weight.data, bits=bits)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--bits", type=int, nargs="+", default=[32, 8, 4, 3, 2])
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="./results/aggressive_stack.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    dense_base = build_model().to(device)
    dense_base.load_state_dict(baseline_state)
    orig_bytes = dense_model_size_bytes(dense_base)
    baseline_acc = evaluate_accuracy(dense_base, test_loader, device)
    floor = baseline_acc - LOSS_BUDGET_PP / 100.0
    print(f"[aggressive] device={device}  baseline={baseline_acc:.4f}  "
          f"pass mark (<= {LOSS_BUDGET_PP}pp loss) = {floor:.4f}\n")

    rows = []
    for lam in args.lambdas:
        per_bits = {b: {"acc": [], "ratio": []} for b in args.bits}

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            _, report, gated = train_one_lambda(
                lam, baseline_state, train_loader, test_loader, device,
                epochs=args.epochs, base_lr=1e-3, gate_lr_mult=10.0, return_model=True,
            )
            folded = fold_gates_into_weights(gated)

            for bits in args.bits:
                q = quantized_copy(folded, bits)
                acc = evaluate_accuracy(q, test_loader, device)
                size = compressed_size_bytes_l0(
                    report, bits_per_weight=bits,
                    scale_bytes_per_layer=0 if bits >= 32 else 4,
                )
                ratio = compression_ratio(orig_bytes, size)
                per_bits[bits]["acc"].append(acc)
                per_bits[bits]["ratio"].append(ratio)
                rows.append({
                    "lambda": lam, "bits": bits, "seed": seed,
                    "accuracy": round(acc, 4), "compression_ratio": round(ratio, 2),
                    "size_kb": round(size / 1024, 2),
                })

        print(f"lambda={lam}")
        for bits in args.bits:
            a = np.array(per_bits[bits]["acc"])
            r = np.array(per_bits[bits]["ratio"])
            loss_pp = (baseline_acc - a.mean()) * 100
            verdict = "PASS" if a.mean() >= floor else "fail"
            label = "fp32" if bits >= 32 else f"INT{bits}"
            print(f"    {label:5} acc={a.mean():.4f}  ratio={r.mean():7.2f}x  "
                  f"loss={loss_pp:+.2f}pp  [{verdict}]")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[aggressive] wrote {len(rows)} rows to {out_path}")

    # Select on per-config MEANS, never on individual seed rows. Picking
    # the best row would report whichever seed got lucky, which is the
    # single easiest way to manufacture a number that won't reproduce.
    configs = {}
    for r in rows:
        configs.setdefault((r["lambda"], r["bits"]), []).append(r)
    summary = [
        {"lambda": lam, "bits": bits,
         "acc": float(np.mean([r["accuracy"] for r in rs])),
         "ratio": float(np.mean([r["compression_ratio"] for r in rs]))}
        for (lam, bits), rs in configs.items()
    ]
    passing = [c for c in summary if c["acc"] >= floor]
    if passing:
        best = max(passing, key=lambda c: c["ratio"])
        print(f"[aggressive] best config within {LOSS_BUDGET_PP}pp (seed means, n={args.n_seeds}): "
              f"lambda={best['lambda']} bits={best['bits']} -> "
              f"{best['ratio']:.2f}x at {best['acc']:.4f}")
    else:
        print(f"[aggressive] NO config stayed within {LOSS_BUDGET_PP}pp of baseline")


if __name__ == "__main__":
    main()
