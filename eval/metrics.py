"""
Shared metrics used by every compression method so numbers are
comparable across techniques -- if pruning and quantization each
computed "size" their own way, a reported "3x compression" wouldn't
mean the same thing between them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def evaluate_accuracy(model: nn.Module, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def dense_model_size_bytes(model: nn.Module) -> int:
    """Naive baseline size: every parameter stored as fp32, no compression."""
    return sum(p.numel() * 4 for p in model.parameters())


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / compressed_bytes
