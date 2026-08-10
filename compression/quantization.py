"""
Post-training INT8 quantization, per-tensor, symmetric, min-max range
(the simplest standard scheme -- affine/per-channel schemes recover
more accuracy but add complexity that isn't the point of this
baseline).

We simulate quantization's effect on accuracy by quantizing then
immediately dequantizing back to float32 for the forward pass
("fake quantization"). This is standard practice for measuring
quantization's accuracy impact without needing real INT8 GEMM
kernels (which PyTorch's default CPU build often doesn't have wired
up for arbitrary custom layers) -- we're asking "how much does 8-bit
precision hurt accuracy," not benchmarking inference speed here.
Size accounting, separately, assumes the real 1-byte-per-weight
storage this scheme implies.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _quantize_dequantize_tensor(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT8: map [-max|w|, +max|w|] -> [-127, 127],
    round to nearest integer, then map back to float. The rounding is
    where information is lost -- that loss is what we're measuring."""
    max_abs = w.abs().max()
    if max_abs == 0:
        return w.clone()
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127)
    return q * scale


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Returns a NEW model (deep copy) with every Conv2d/Linear weight
    tensor passed through fake INT8 quantization. Biases are left at
    fp32 (standard practice -- bias tensors are tiny, quantizing them
    saves almost no size but adds real accuracy risk since bias errors
    don't average out across a layer's output the way weight errors do)."""
    quantized = copy.deepcopy(model)
    for m in quantized.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data = _quantize_dequantize_tensor(m.weight.data)
    return quantized


def compressed_size_bytes_int8(model: nn.Module) -> int:
    """1 byte per weight (the quantized int8 value) + one fp32 scale
    factor per layer (negligible, but included for honesty). Biases
    stay fp32 (see quantize_model_int8 docstring)."""
    total_bytes = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total_bytes += m.weight.numel() * 1  # int8
            total_bytes += 4  # one fp32 scale factor for this layer
            if m.bias is not None:
                total_bytes += m.bias.numel() * 4
    return total_bytes
