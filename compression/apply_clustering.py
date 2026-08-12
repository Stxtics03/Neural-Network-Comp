"""
Applies unit-clustering compression to a model: for each Conv2d/Linear
layer, cluster its output units (filters/neurons) by weight-vector
similarity, then replace every unit's weights with its cluster's
centroid. Units that land in the same cluster become bit-for-bit
identical -- which is exactly where the compression comes from (see
compressed_size_bytes_clustered).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from compression.deterministic_annealing import DeterministicAnnealingClusterer
from compression.kmeans import KMeansClusterer
from compression.unit_vectors import extract_unit_vectors


def _cluster_layer(module: nn.Module, clusterer) -> tuple[torch.Tensor, list[int], int]:
    vectors = extract_unit_vectors(module)
    clusterer.fit(vectors)
    clustered_vectors = clusterer.centroids_[clusterer.labels_]  # (num_units, unit_dim)
    new_weight = torch.tensor(
        clustered_vectors, dtype=module.weight.dtype
    ).reshape(module.weight.shape)
    return new_weight, clusterer.labels_.tolist(), clusterer.centroids_.shape[0]


def apply_clustering(
    model: nn.Module, k: int, method: str = "da", seed: int = 0,
    per_layer_kwargs: dict[str, dict] | None = None, **method_kwargs,
) -> tuple[nn.Module, dict]:
    """
    method: "da" (deterministic annealing), "kmeans++" or "kmeans_random".

    per_layer_kwargs optionally overrides method_kwargs for individual
    layers, keyed by layer name. Needed for compute-parity comparisons:
    a fair kmeans restart budget is per-layer, because DA's cost per
    layer and kmeans' cost per layer scale differently (see
    experiments/compute_parity_check.py).

    Returns (new_model, info) where info maps layer_name -> (labels, num_clusters, unit_dim).
    """
    clustered = copy.deepcopy(model)
    info = {}
    per_layer_kwargs = per_layer_kwargs or {}

    for name, module in clustered.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        kwargs = {**method_kwargs, **per_layer_kwargs.get(name, {})}
        if method == "da":
            clusterer = DeterministicAnnealingClusterer(k=k, seed=seed, **kwargs)
        elif method == "kmeans++":
            clusterer = KMeansClusterer(k=k, init="kmeans++", seed=seed, **kwargs)
        elif method == "kmeans_random":
            clusterer = KMeansClusterer(k=k, init="random", seed=seed, **kwargs)
        else:
            raise ValueError(f"unknown method {method}")

        new_weight, labels, num_clusters = _cluster_layer(module, clusterer)
        module.weight.data = new_weight
        unit_dim = new_weight.numel() // new_weight.shape[0]
        info[name] = {"labels": labels, "num_clusters": num_clusters, "unit_dim": unit_dim}

    return clustered, info


def compressed_size_bytes_clustered(info: dict, model: nn.Module) -> int:
    """
    Storage accounting: for each clustered layer, store the centroids
    (num_clusters x unit_dim, fp32) once, plus one small index per unit
    saying which centroid it uses (ceil(log2(num_clusters)) bits,
    rounded up to whole bytes). This is where the compression comes
    from -- e.g. if 16 filters collapse to 4 clusters, we store 4
    centroids instead of 16 full filters, plus 16 tiny indices.

    Biases are never clustered -- stored dense at fp32, same reasoning
    as the pruning/quantization baselines (too few of them to matter,
    too much accuracy risk to bother).
    """
    total_bytes = 0
    for name, module in model.named_modules():
        if name not in info:
            continue
        layer_info = info[name]
        num_units = len(layer_info["labels"])
        num_clusters = layer_info["num_clusters"]
        unit_dim = layer_info["unit_dim"]

        centroid_bytes = num_clusters * unit_dim * 4  # fp32 centroids
        index_bits = max(1, math.ceil(math.log2(max(num_clusters, 2))))
        index_bytes_per_unit = math.ceil(index_bits / 8)
        total_bytes += centroid_bytes + num_units * index_bytes_per_unit

        if hasattr(module, "bias") and module.bias is not None:
            total_bytes += module.bias.numel() * 4

    return total_bytes
