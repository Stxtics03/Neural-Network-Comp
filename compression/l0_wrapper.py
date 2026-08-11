"""
Wraps a model, attaching one HardConcreteGate per Conv2d/Linear layer's
output units, and multiplies that layer's output by the gate values via
a forward hook -- so gates are learned jointly with the base model's
weights, without needing to modify the base model's forward() method
at all.

Conv2d output shape (B, C, H, W): gate multiplies per-channel, so it
needs reshaping to (1, C, 1, 1) to broadcast correctly.
Linear output shape (B, F): gate multiplies per-feature, reshaping to
(1, F).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from compression.l0_gates import HardConcreteGate


class L0GatedModel(nn.Module):
    def __init__(self, base_model: nn.Module, exclude_layer_names: list[str] | None = None):
        """
        exclude_layer_names: layers to skip gating entirely -- MUST
        include the final classification layer (e.g. "fc3" for
        LeNet-5). Gating a layer means its output can be multiplied by
        0, and a gated-to-0 output-layer unit would mean "this class
        can never be predicted" -- a correctness bug, not a pruning
        decision. L0 is meant to prune hidden-layer capacity, not
        remove entire output classes.
        """
        super().__init__()
        self.base_model = base_model
        self.exclude_layer_names = set(exclude_layer_names or [])
        self.gates = nn.ModuleDict()
        self._hook_handles = []

        for name, module in base_model.named_modules():
            if name in self.exclude_layer_names:
                continue
            if isinstance(module, nn.Conv2d):
                num_units = module.out_channels
            elif isinstance(module, nn.Linear):
                num_units = module.out_features
            else:
                continue
            gate_name = name.replace(".", "_")
            self.gates[gate_name] = HardConcreteGate(num_units)
            handle = module.register_forward_hook(self._make_hook(gate_name, is_conv=isinstance(module, nn.Conv2d)))
            self._hook_handles.append(handle)

    def _make_hook(self, gate_name: str, is_conv: bool):
        def hook(module, inputs, output):
            z = self.gates[gate_name]()
            if is_conv:
                z = z.view(1, -1, 1, 1)
            else:
                z = z.view(1, -1)
            return output * z
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def total_expected_l0(self) -> torch.Tensor:
        return sum(gate.expected_l0() for gate in self.gates.values())

    def gate_param_names(self) -> list[str]:
        return [f"gates.{name}.log_alpha" for name in self.gates.keys()]

    def sparsity_report(self) -> dict[str, dict]:
        """Per-layer count of surviving (nonzero-gate) units at eval time."""
        report = {}
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for gate_name, gate in self.gates.items():
                mask = gate.hard_gate_mask()
                report[gate_name] = {
                    "num_units": mask.numel(),
                    "num_surviving": int(mask.sum().item()),
                    "sparsity": 1.0 - mask.sum().item() / mask.numel(),
                }
        self.train(was_training)
        return report
