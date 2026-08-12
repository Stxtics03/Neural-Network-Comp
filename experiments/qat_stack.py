"""
Quantization-aware fine-tuning on top of the L0-pruned model.

aggressive_stack.py applies quantization post-hoc to a finished model
(PTQ): the weights were never told they would be rounded. At INT8 that
costs nothing, but at INT4 it cost 1.34pp at lambda=0.05 -- enough to
miss a 1pp budget at 147.82x. QAT puts the rounding inside the training
loop so the weights can adapt to the grid they will be stored on.

Mechanics:
  - The quantizer is registered as a weight PARAMETRIZATION, so an fp32
    master weight is kept and quantized on every access, which is what
    lets gradients accumulate at full precision between steps. Storing
    the rounded value back into the parameter instead would lose every
    update smaller than one quantization step.
  - round() has zero gradient almost everywhere, so it uses a
    straight-through estimator: quantize on the forward pass, pass the
    gradient through unchanged on the backward.

The structural sparsity has to be defended. L0's dead units sit at
exactly zero, but nothing in the QAT loss keeps them there -- one Adam
step with any nonzero gradient revives them, the surviving-unit counts
change, and every size number computed from the pre-QAT report becomes
a lie. So dead units are re-pinned to zero after every step, and the
sparsity report is re-derived afterward and compared against the
original. Mismatch is a hard failure, not a warning.

Usage:
    python experiments/qat_stack.py --configs 0.05:4 0.02:4 0.05:3 --n-seeds 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.l0_int8_combined import fold_gates_into_weights
from experiments.train_l0 import compressed_size_bytes_l0, load_baseline_state, train_one_lambda
from models.lenet5 import build_model


class _RoundSTE(torch.autograd.Function):
    """round() on the forward, identity on the backward."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class FakeQuant(nn.Module):
    """Symmetric per-tensor fake quantization, matching
    compression/quantization.py's scheme exactly so QAT and PTQ numbers
    are comparable."""

    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        max_abs = w.abs().max()
        if max_abs == 0:
            return w
        scale = max_abs / self.qmax
        return torch.clamp(_RoundSTE.apply(w / scale), -self.qmax, self.qmax) * scale


def quantizable_modules(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def dead_unit_masks(model) -> dict:
    """A unit is dead if L0 zeroed its entire weight row. Shaped to
    broadcast over the weight tensor."""
    masks = {}
    for m in quantizable_modules(model):
        flat = m.weight.data.reshape(m.weight.shape[0], -1)
        alive = (flat.abs().sum(dim=1) != 0).float()
        masks[m] = alive
    return masks


def apply_masks(model, masks):
    with torch.no_grad():
        for m, alive in masks.items():
            shape = [-1] + [1] * (m.weight.dim() - 1)
            if parametrize.is_parametrized(m, "weight"):
                m.parametrizations.weight.original *= alive.view(shape)
            else:
                m.weight.data *= alive.view(shape)
            if m.bias is not None:
                m.bias.data *= alive


def surviving_counts(model) -> dict:
    counts = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            flat = m.weight.data.reshape(m.weight.shape[0], -1)
            counts[name] = int((flat.abs().sum(dim=1) != 0).sum().item())
    return counts


def qat_finetune(model, masks, train_loader, device, epochs: int, lr: float):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
            apply_masks(model, masks)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["0.05:4", "0.02:4", "0.05:3"],
                        help="lambda:bits pairs")
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--l0-epochs", type=int, default=5)
    parser.add_argument("--qat-epochs", type=int, default=3)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    parser.add_argument("--out", default="./results/qat_stack.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    dense = build_model().to(device)
    dense.load_state_dict(baseline_state)
    orig_bytes = dense_model_size_bytes(dense)
    baseline_acc = evaluate_accuracy(dense, test_loader, device)
    floor = baseline_acc - 0.01
    print(f"[qat] baseline={baseline_acc:.4f}  pass mark (<=1pp) = {floor:.4f}\n")

    rows = []
    for cfg in args.configs:
        lam_s, bits_s = cfg.split(":")
        lam, bits = float(lam_s), int(bits_s)
        ptq_accs, qat_accs, ratios = [], [], []

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            _, report, gated = train_one_lambda(
                lam, baseline_state, train_loader, test_loader, device,
                epochs=args.l0_epochs, base_lr=1e-3, gate_lr_mult=10.0, return_model=True,
            )
            folded = fold_gates_into_weights(gated)
            counts_before = surviving_counts(folded)
            masks = dead_unit_masks(folded)

            # PTQ reference on this exact model, for a paired comparison.
            ptq = FakeQuant(bits)
            ptq_model = build_model().to(device)
            ptq_model.load_state_dict(folded.state_dict())
            with torch.no_grad():
                for m in quantizable_modules(ptq_model):
                    m.weight.data = ptq(m.weight.data)
            acc_ptq = evaluate_accuracy(ptq_model, test_loader, device)

            # QAT.
            for m in quantizable_modules(folded):
                parametrize.register_parametrization(m, "weight", FakeQuant(bits))
            qat_finetune(folded, masks, train_loader, device, args.qat_epochs, args.qat_lr)
            for m in quantizable_modules(folded):
                parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)

            acc_qat = evaluate_accuracy(folded, test_loader, device)
            counts_after = surviving_counts(folded)

            # Two very different failure directions, only one of which is
            # dishonest. A REVIVED unit (count up) means the model is
            # bigger than the report claims -- that would overstate
            # compression, so it is a hard error. A unit that DIES during
            # QAT (count down) is safe: at INT4 the grid is only 7 levels
            # per sign and the scale is per-tensor, so a surviving unit
            # whose weights are all small relative to its layer's max can
            # round entirely to zero. The model is then smaller than the
            # pre-QAT report says, so pricing it from that report would
            # understate compression. Re-price from the counts actually
            # observed instead.
            revived = {k: (counts_before[k], counts_after[k])
                       for k in counts_before if counts_after[k] > counts_before[k]}
            if revived:
                raise RuntimeError(
                    f"QAT revived dead units {revived} (before -> after). The model is "
                    f"larger than the report claims, so the compression number would "
                    f"be overstated. Masking failed."
                )
            killed = {k: (counts_before[k], counts_after[k])
                      for k in counts_before if counts_after[k] < counts_before[k]}
            if killed:
                print(f"    note: INT{bits} rounding removed whole units {killed} "
                      f"(before -> after); re-pricing from actual counts")

            actual_report = {name: {"num_surviving": n} for name, n in counts_after.items()}
            size = compressed_size_bytes_l0(actual_report, bits_per_weight=bits,
                                            scale_bytes_per_layer=4)
            ratio = compression_ratio(orig_bytes, size)
            ptq_accs.append(acc_ptq)
            qat_accs.append(acc_qat)
            ratios.append(ratio)
            rows.append({
                "lambda": lam, "bits": bits, "seed": seed,
                "acc_ptq": round(acc_ptq, 4), "acc_qat": round(acc_qat, 4),
                "compression_ratio": round(ratio, 2), "size_kb": round(size / 1024, 2),
            })

        p, q, r = np.array(ptq_accs), np.array(qat_accs), np.array(ratios)
        verdict = "PASS" if q.mean() >= floor else "fail"
        print(f"lambda={lam} INT{bits}  ratio={r.mean():.2f}x")
        print(f"    PTQ acc={p.mean():.4f}  loss={(baseline_acc-p.mean())*100:+.2f}pp")
        print(f"    QAT acc={q.mean():.4f}  loss={(baseline_acc-q.mean())*100:+.2f}pp  "
              f"recovered={(q.mean()-p.mean())*100:+.2f}pp  [{verdict}]\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[qat] wrote {len(rows)} rows to {out_path}")

    # Select on per-config means, not on individual seed rows -- see the
    # same note in aggressive_stack.py.
    configs = {}
    for r in rows:
        configs.setdefault((r["lambda"], r["bits"]), []).append(r)
    summary = [
        {"lambda": lam, "bits": bits,
         "acc": float(np.mean([r["acc_qat"] for r in rs])),
         "ratio": float(np.mean([r["compression_ratio"] for r in rs]))}
        for (lam, bits), rs in configs.items()
    ]
    passing = [c for c in summary if c["acc"] >= floor]
    if passing:
        best = max(passing, key=lambda c: c["ratio"])
        print(f"[qat] best within 1pp (seed means, n={args.n_seeds}): "
              f"lambda={best['lambda']} INT{best['bits']} -> "
              f"{best['ratio']:.2f}x at {best['acc']:.4f}")
    else:
        print("[qat] no config stayed within 1pp")


if __name__ == "__main__":
    main()
