"""
Puts the pruning and INT8 baselines on the same evidentiary footing as
everything else in the project.

results/baselines.csv is a single draw: one trained checkpoint, one
evaluation per setting. Every other headline number here is a multi-seed
mean, which made that first table the weakest evidence in the repo.

The variance being measured is NOT in the compression methods --
magnitude pruning and per-tensor INT8 are both deterministic given a
fixed checkpoint, so re-running them on one model returns identical
numbers forever. The variance lives in the trained model itself. So each
seed needs its own baseline trained from scratch, which is why this is
slower than the other seed-verification scripts.

Seed 0 reuses the existing checkpoint (the one every other result in the
repo is built on); additional seeds are trained into their own
directories. The fixed checkpoint filename means a shared directory
would silently overwrite, so each seed gets its own.

Usage:
    python experiments/baseline_seed_verify.py --extra-seeds 1 2 --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.magnitude_pruning import compressed_size_bytes_pruned, prune_model
from compression.quantization import compressed_size_bytes_int8, quantize_model_int8
from data.loaders import get_loaders
from eval.metrics import compression_ratio, dense_model_size_bytes, evaluate_accuracy
from experiments.train_baseline import train_baseline
from models.lenet5 import build_model

BASE_CKPT = Path("./results/checkpoints/lenet5_best.pt")
SPARSITY_LEVELS = [0.3, 0.5, 0.7, 0.9]


def evaluate_one_checkpoint(ckpt_path: Path, test_loader, device) -> list[dict]:
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    orig_bytes = dense_model_size_bytes(model)
    out = [{
        "method": "baseline", "setting": "none",
        "accuracy": evaluate_accuracy(model, test_loader, device),
        "compression_ratio": 1.0,
    }]

    for sparsity in SPARSITY_LEVELS:
        pruned = prune_model(model, sparsity).to(device)
        size = compressed_size_bytes_pruned(pruned)
        out.append({
            "method": "magnitude_pruning", "setting": f"sparsity={sparsity}",
            "accuracy": evaluate_accuracy(pruned, test_loader, device),
            "compression_ratio": compression_ratio(orig_bytes, size),
        })

    quantized = quantize_model_int8(model).to(device)
    size = compressed_size_bytes_int8(quantized)
    out.append({
        "method": "int8_quantization", "setting": "per-tensor symmetric",
        "accuracy": evaluate_accuracy(quantized, test_loader, device),
        "compression_ratio": compression_ratio(orig_bytes, size),
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="./results/baselines_seeds.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders()

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"No checkpoint at {BASE_CKPT}. Run train_baseline.py first.")

    checkpoints = {0: BASE_CKPT}
    for seed in args.extra_seeds:
        ckpt_dir = Path(f"./results/checkpoints/seed{seed}")
        ckpt_path = ckpt_dir / "lenet5_best.pt"
        if ckpt_path.exists():
            print(f"[baseline_seeds] seed {seed}: reusing {ckpt_path}")
        else:
            print(f"[baseline_seeds] seed {seed}: training {args.epochs} epochs -> {ckpt_dir}")
            acc, ckpt_path = train_baseline(seed, args.epochs, 64, 1e-3, ckpt_dir, device, quiet=True)
            print(f"[baseline_seeds] seed {seed}: best acc={acc:.4f}")
        checkpoints[seed] = ckpt_path

    rows = []
    for seed, path in sorted(checkpoints.items()):
        for r in evaluate_one_checkpoint(path, test_loader, device):
            rows.append({"seed": seed, **r,
                         "accuracy": round(r["accuracy"], 4),
                         "compression_ratio": round(r["compression_ratio"], 2)})

    print(f"\n{'method':<20}{'setting':<22}{'accuracy (mean +/- sd)':<26}{'ratio'}")
    settings = []
    for r in rows:
        key = (r["method"], r["setting"])
        if key not in settings:
            settings.append(key)
    for method, setting in settings:
        accs = [r["accuracy"] for r in rows if (r["method"], r["setting"]) == (method, setting)]
        rats = [r["compression_ratio"] for r in rows if (r["method"], r["setting"]) == (method, setting)]
        sd = st.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{method:<20}{setting:<22}{st.mean(accs)*100:6.2f}% +/- {sd*100:4.2f}pp"
              f"          {st.mean(rats):.2f}x")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "method", "setting", "accuracy", "compression_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[baseline_seeds] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
