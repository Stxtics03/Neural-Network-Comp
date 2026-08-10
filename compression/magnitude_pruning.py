"""
Global unstructured magnitude pruning: zero out the smallest-magnitude
weights across ALL prunable layers combined (not per-layer), up to a
target sparsity. Global (not per-layer) is the standard choice --
per-layer pruning at a fixed sparsity forces every layer to lose the
same fraction even if some layers have much more redundancy than
others; global pruning lets the threshold naturally protect
important layers and hit harder ones that don't matter as much.

No fine-tuning after pruning (single-shot) -- that's a deliberate
scope choice for this project: fine-tuning would recover more
accuracy at the same sparsity, but it also means "pruning" and
"further training" are doing the work together, which muddies a
clean comparison against other single-shot compression methods
(quantization, DA clustering) later. Worth noting as a limitation,
not hiding it.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


def prune_model(model: nn.Module, sparsity: float) -> nn.Module:
    """Returns a NEW model (deep copy) with the smallest-magnitude
    weights in every Conv2d/Linear layer zeroed out, globally ranked
    by |weight| across all such layers combined."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    pruned = copy.deepcopy(model)
    prunable_modules = [m for m in pruned.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    all_weights = torch.cat([m.weight.data.abs().flatten() for m in prunable_modules])
    if sparsity == 0.0:
        return pruned
    k = int(sparsity * all_weights.numel())
    threshold = torch.kthvalue(all_weights, k).values.item()

    for m in prunable_modules:
        mask = m.weight.data.abs() > threshold
        m.weight.data.mul_(mask)

    return pruned


def actual_sparsity(model: nn.Module) -> float:
    """Sanity-check helper: what fraction of weights are exactly zero,
    post-pruning. Should closely match the requested sparsity (won't be
    exact due to ties at the threshold)."""
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            zeros += (m.weight.data == 0).sum().item()
    return zeros / total


def compressed_size_bytes_pruned(model: nn.Module) -> int:
    """
    Sparse storage accounting: for each Conv2d/Linear layer, store only
    the nonzero weights, each as (value: 4 bytes fp32) + (index: enough
    bits to address any position in that layer's flattened weight
    tensor, rounded up to whole bytes -- e.g. a layer with 2400 weights
    needs ceil(log2(2400))=12 bits -> 2 bytes per index, same idea as
    COO sparse format).

    Biases are NOT pruned (there are few of them, and zeroing them adds
    accuracy risk for negligible size savings) -- always stored dense,
    at fp32.
    """
    total_bytes = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n_weights = m.weight.numel()
            n_nonzero = (m.weight.data != 0).sum().item()
            index_bytes = max(1, math.ceil(math.ceil(math.log2(max(n_weights, 2))) / 8))
            total_bytes += n_nonzero * (4 + index_bytes)  # value + index per nonzero
            if m.bias is not None:
                total_bytes += m.bias.numel() * 4
    return total_bytes
