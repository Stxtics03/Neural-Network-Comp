"""
Turns a layer's weight tensor into one vector per output unit -- the
thing we'll cluster. The idea: if two filters (or two neurons) have
very similar weight vectors, they're doing almost the same job, so
replacing both with their average loses little accuracy but saves
storing them separately.

Conv2d.weight shape: (out_channels, in_channels, kh, kw) -- each
output channel/filter is one unit, flattened to a 1D vector.
Linear.weight shape: (out_features, in_features) -- each output
neuron is already a row, i.e. already a vector.
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn


def extract_unit_vectors(module: nn.Module) -> np.ndarray:
    if isinstance(module, nn.Conv2d):
        w = module.weight.data.cpu().numpy()
        return w.reshape(w.shape[0], -1)  # (out_channels, in_channels*kh*kw)
    elif isinstance(module, nn.Linear):
        return module.weight.data.cpu().numpy()  # (out_features, in_features), already per-unit
    else:
        raise TypeError(f"extract_unit_vectors only supports Conv2d/Linear, got {type(module)}")
