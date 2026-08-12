"""
Stacks L0 structured sparsity and INT8 quantization, which compress
along independent axes: L0 removes whole units (fewer weights), INT8
shrinks each surviving weight (fewer bytes per weight). If they don't
interact, the ratios should roughly multiply.

Two details this script exists to get right:

1. GATES MUST BE FOLDED INTO THE WEIGHTS FIRST. A hard-concrete gate is
   not binary at eval time -- z = clamp(sigmoid(log_alpha)*1.2 - 0.1,
   0, 1), so a surviving unit's gate is only exactly 1.0 when
   log_alpha >= 2.4, and is otherwise a fractional scaling. Since
   z*(Wx+b) == (zW)x + (zb), folding z into the weights and biases
   gives a plain model that is functionally identical and needs no
   extra storage for the gate values. The existing size accounting in
   train_l0.py implicitly assumes this folding has happened; without
   it you would have to store a float per surviving unit. The fold is
   verified numerically before anything else runs.

2. FOLDING ALSO FIXES THE QUANTIZATION SCALE. INT8 here is per-tensor
   symmetric, so the scale is set by max|w| over the whole tensor. A
   dead unit's weights are still sitting in that tensor and can carry
   a large magnitude, which would inflate the scale and spend
   precision on units that contribute nothing. Folding zeroes them
   exactly, so the scale is set only by weights that survive.

Usage:
    python experiments/l0_int8_combined.py --lambdas 0.005 0.01 0.02 --n-seeds 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.quantization import _quantize_dequantize_tensor
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.train_l0 import compressed_size_bytes_l0, load_baseline_state, train_one_lambda
from models.lenet5 import build_model

FOLD_TOLERANCE = 1e-4  # max acceptable accuracy drift from folding; it should be ~0


def fold_gates_into_weights(gated_model) -> nn.Module:
    """Returns the underlying base model with each gate's eval-time value
    multiplied into its layer's weights and bias. Dead units land at
    exactly zero. The returned model needs no gates and no hooks."""
    base = gated_model.base_model
    name_to_module = dict(base.named_modules())

    with torch.no_grad():
        for gate_name, gate in gated_model.gates.items():
            if gate_name not in name_to_module:
                # L0GatedModel builds gate names as name.replace(".", "_"),
                # which is identity for LeNet-5's flat layer names. A
                # nested model would need the inverse mapping instead of
                # silently skipping, so fail loudly rather than quietly
                # folding nothing.
                raise KeyError(f"gate {gate_name!r} has no matching module; nested layer names need remapping")
            module = name_to_module[gate_name]
            gate.eval()
            z = gate()
            module.weight.data *= z.view([-1] + [1] * (module.weight.dim() - 1))
            if module.bias is not None:
                module.bias.data *= z

    for handle in gated_model._hook_handles:
        handle.remove()  # the folded model must not re-apply the gates
    return base


def quantize_in_place(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data = _quantize_dequantize_tensor(m.weight.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr-mult", type=float, default=10.0)
    parser.add_argument("--out", default="./results/l0_int8_combined.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[l0_int8] device={device}, n_seeds={args.n_seeds}")

    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    dense_base = build_model().to(device)
    dense_base.load_state_dict(baseline_state)
    orig_bytes = dense_model_size_bytes(dense_base)
    baseline_acc = evaluate_accuracy(dense_base, test_loader, device)
    print(f"[l0_int8] baseline: accuracy={baseline_acc:.4f}  size={orig_bytes/1024:.1f} KB\n")

    rows = []
    for lam in args.lambdas:
        l0_accs, combo_accs, l0_ratios, combo_ratios = [], [], [], []

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            acc_l0, report, gated = train_one_lambda(
                lam, baseline_state, train_loader, test_loader, device,
                epochs=args.epochs, base_lr=args.base_lr,
                gate_lr_mult=args.gate_lr_mult, return_model=True,
            )

            folded = fold_gates_into_weights(gated)
            acc_folded = evaluate_accuracy(folded, test_loader, device)
            drift = abs(acc_folded - acc_l0)
            if drift > FOLD_TOLERANCE:
                raise RuntimeError(
                    f"folding changed accuracy by {drift:.5f} (gated={acc_l0:.4f}, "
                    f"folded={acc_folded:.4f}) -- the folded model is not the model "
                    f"that was measured, so the combined number would be meaningless"
                )

            quantize_in_place(folded)
            acc_combo = evaluate_accuracy(folded, test_loader, device)

            size_l0 = compressed_size_bytes_l0(report)
            size_combo = compressed_size_bytes_l0(report, bits_per_weight=8, scale_bytes_per_layer=4)
            ratio_l0 = compression_ratio(orig_bytes, size_l0)
            ratio_combo = compression_ratio(orig_bytes, size_combo)

            l0_accs.append(acc_l0)
            combo_accs.append(acc_combo)
            l0_ratios.append(ratio_l0)
            combo_ratios.append(ratio_combo)
            rows.append({
                "lambda": lam, "seed": seed,
                "acc_l0_fp32": round(acc_l0, 4), "acc_l0_int8": round(acc_combo, 4),
                "ratio_l0_fp32": round(ratio_l0, 3), "ratio_l0_int8": round(ratio_combo, 3),
                "size_kb_l0_fp32": round(size_l0 / 1024, 2),
                "size_kb_l0_int8": round(size_combo / 1024, 2),
            })

        l0_a, cm_a = np.array(l0_accs), np.array(combo_accs)
        l0_r, cm_r = np.array(l0_ratios), np.array(combo_ratios)
        quant_cost = (l0_a - cm_a).mean()
        print(f"lambda={lam:.4f}")
        print(f"    L0 only    : acc={l0_a.mean():.4f}+/-{l0_a.std():.4f}  ratio={l0_r.mean():.2f}x+/-{l0_r.std():.2f}")
        print(f"    L0 + INT8  : acc={cm_a.mean():.4f}+/-{cm_a.std():.4f}  ratio={cm_r.mean():.2f}x+/-{cm_r.std():.2f}")
        print(f"    quantization cost on top of L0: {quant_cost*100:+.2f}pp   "
              f"extra compression: {cm_r.mean()/l0_r.mean():.2f}x\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[l0_int8] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
