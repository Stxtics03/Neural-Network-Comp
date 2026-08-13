"""
Trains LeNet-5 on MNIST from scratch and saves the best checkpoint.
This is the foundation everything else (pruning, quantization,
clustering, L0) gets compared against -- so it needs a real, reported
accuracy number, not an assumed one.

Usage:
    python experiments/train_baseline.py --epochs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import get_loaders
from models.lenet5 import build_model


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_baseline(seed: int, epochs: int, batch_size: int, lr: float,
                    ckpt_dir: Path, device, quiet: bool = False):
    """Trains one LeNet-5 from scratch and saves the best-by-test-accuracy
    checkpoint into ckpt_dir. Returns (best_accuracy, checkpoint_path).

    Factored out so baseline_seed_verify.py can train additional seeds
    through the exact same procedure rather than reimplementing it. Note
    that ckpt_dir must differ per seed -- the filename is fixed, so
    sharing a directory silently overwrites the previous seed's model.
    """
    torch.manual_seed(seed)
    train_loader, test_loader = get_loaders(batch_size=batch_size)
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "lenet5_best.pt"
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        if not quiet:
            print(f"[train_baseline] epoch {epoch:2d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), ckpt_path)

    return best_acc, ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-dir", default="./results/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_baseline] device={device}")

    best_acc, ckpt_path = train_baseline(
        args.seed, args.epochs, args.batch_size, args.lr, Path(args.ckpt_dir), device,
    )
    print(f"\n[train_baseline] best test accuracy: {best_acc:.4f}")
    print(f"[train_baseline] checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
