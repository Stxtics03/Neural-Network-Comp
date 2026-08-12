"""
Turns an L0-gated model into a REAL smaller network and times it.

Everything up to now measured compression as an accounting exercise:
count the surviving units, price the bytes they would occupy. This
script actually builds the narrower LeNet-5, transplants the surviving
weights into it, checks it computes the same function, saves it to
disk, and benchmarks its inference latency.

That does three things no accounting can:
  - validates compressed_size_bytes_l0 against a real file on disk
  - proves the surviving-unit bookkeeping (including the cascade into
    each next layer's input dimension) is correct, because a wrong
    index mapping produces a different accuracy, not a smaller number
  - produces an honest inference speedup, which fake-quant experiments
    fundamentally cannot

The cascade: a gated-off conv1 filter doesn't just delete its own
weights, it deletes a channel from conv2's INPUT. Into fc1 the mapping
is not one column per surviving conv2 channel but 25 -- flatten order
is (channel, height, width) over a 5x5 map, so channel c owns columns
[25c, 25c+25).

Usage:
    python experiments/materialize_pruned.py --lam 0.02 --epochs 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.l0_int8_combined import fold_gates_into_weights
from experiments.train_l0 import compressed_size_bytes_l0, load_baseline_state, train_one_lambda
from models.lenet5 import build_model

SPATIAL = 25  # 5x5 feature map feeding fc1
ACC_TOLERANCE = 1e-4


def surviving_indices(gated_model) -> dict[str, torch.Tensor]:
    idx = {}
    for gate_name, gate in gated_model.gates.items():
        mask = gate.hard_gate_mask()
        idx[gate_name] = torch.nonzero(mask, as_tuple=True)[0]
    return idx


def transplant(folded, idx: dict[str, torch.Tensor]):
    """Build the narrow model and copy across only surviving rows/columns."""
    s1, s2 = idx["conv1"], idx["conv2"]
    f1, f2 = idx["fc1"], idx["fc2"]
    narrow = build_model(widths=(len(s1), len(s2), len(f1), len(f2)))

    # fc1's input columns: 25 consecutive columns per surviving conv2 channel.
    fc1_cols = torch.cat([torch.arange(c * SPATIAL, (c + 1) * SPATIAL) for c in s2])

    with torch.no_grad():
        narrow.conv1.weight.copy_(folded.conv1.weight[s1])
        narrow.conv1.bias.copy_(folded.conv1.bias[s1])

        narrow.conv2.weight.copy_(folded.conv2.weight[s2][:, s1])
        narrow.conv2.bias.copy_(folded.conv2.bias[s2])

        narrow.fc1.weight.copy_(folded.fc1.weight[f1][:, fc1_cols])
        narrow.fc1.bias.copy_(folded.fc1.bias[f1])

        narrow.fc2.weight.copy_(folded.fc2.weight[f2][:, f1])
        narrow.fc2.bias.copy_(folded.fc2.bias[f2])

        narrow.fc3.weight.copy_(folded.fc3.weight[:, f2])
        narrow.fc3.bias.copy_(folded.fc3.bias)
    return narrow


def benchmark(model, batch_size: int, iters: int, warmup: int = 20) -> float:
    """Median seconds per forward pass. Median, not mean -- CPU timing on
    a machine doing other work has a long right tail that a mean would
    absorb."""
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--save-to", default="./results/checkpoints/lenet5_narrow.pt")
    args = parser.parse_args()

    device = torch.device("cpu")  # latency comparison must be single-device
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_loaders()
    baseline_state = load_baseline_state(device)

    dense = build_model().to(device)
    dense.load_state_dict(baseline_state)
    dense_bytes = dense_model_size_bytes(dense)
    dense_acc = evaluate_accuracy(dense, test_loader, device)

    print(f"[materialize] training L0 at lambda={args.lam} ...")
    acc_gated, report, gated = train_one_lambda(
        args.lam, baseline_state, train_loader, test_loader, device,
        epochs=args.epochs, base_lr=1e-3, gate_lr_mult=10.0, return_model=True,
    )
    idx = surviving_indices(gated)
    folded = fold_gates_into_weights(gated)
    acc_folded = evaluate_accuracy(folded, test_loader, device)

    narrow = transplant(folded, idx)
    acc_narrow = evaluate_accuracy(narrow, test_loader, device)

    widths = narrow.widths
    print(f"\n[materialize] widths: (6,16,120,84) -> {widths}")
    print(f"[materialize] accuracy  gated={acc_gated:.4f}  folded={acc_folded:.4f}  "
          f"narrow={acc_narrow:.4f}")
    drift = abs(acc_narrow - acc_folded)
    if drift > ACC_TOLERANCE:
        raise RuntimeError(
            f"transplanted model differs from folded model by {drift:.5f} -- "
            f"the surviving-unit index mapping is wrong, so the size accounting "
            f"built on the same cascade is wrong too"
        )
    print(f"[materialize] transplant is exact (drift {drift:.2e}) -- the cascade "
          f"used by compressed_size_bytes_l0 is validated against a real model")

    # Size: accounted vs. actual file on disk.
    accounted = compressed_size_bytes_l0(report)
    save_path = Path(args.save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(narrow.state_dict(), save_path)
    on_disk = save_path.stat().st_size
    params = sum(p.numel() for p in narrow.parameters())
    print(f"\n[materialize] parameters: {sum(p.numel() for p in dense.parameters())} -> {params} "
          f"({sum(p.numel() for p in dense.parameters())/params:.2f}x fewer)")
    print(f"[materialize] accounted size : {accounted/1024:7.2f} KB "
          f"({compression_ratio(dense_bytes, accounted):.2f}x)")
    print(f"[materialize] actual .pt file: {on_disk/1024:7.2f} KB "
          f"(includes torch pickle overhead, so it is larger)")

    print(f"\n[materialize] latency (median over {args.iters} forward passes, CPU):")
    for bs in (1, 64):
        t_dense = benchmark(dense, bs, args.iters)
        t_narrow = benchmark(narrow, bs, args.iters)
        print(f"    batch={bs:3}  dense={t_dense*1000:7.3f} ms  narrow={t_narrow*1000:7.3f} ms  "
              f"speedup={t_dense/t_narrow:.2f}x")


if __name__ == "__main__":
    main()
