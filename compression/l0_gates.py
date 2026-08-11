"""
Hard-concrete gates (Louizos, Welling & Kingma, 2017, "Learning Sparse
Neural Networks through L0 Regularization").

The problem this solves: we want to penalize the L0 norm (count of
nonzero units) directly -- "how many filters/neurons are actually
in use" -- but L0 is discrete and non-differentiable, so it can't be
optimized with gradient descent directly. The hard-concrete
distribution is a continuous relaxation: each unit gets a learnable
gate z in [0,1], sampled from a distribution controlled by one
parameter (log_alpha) per unit. During training, z is a smooth random
variable (so gradients flow), but the distribution is constructed so
z lands EXACTLY at 0 or 1 with nonzero probability -- unlike a plain
sigmoid gate, which only approaches 0/1 in the limit. That "exactly
zero" property is what makes true pruning possible: a unit with
z=0 contributes NOTHING to the forward pass, so it's genuinely removable,
not just "small."

At eval time, sampling is replaced with a deterministic gate value
(no randomness) -- see forward()'s train/eval branching.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# Stretch interval: hard-concrete samples are drawn on (0,1) then
# stretched to (GAMMA, ZETA) before being clipped back to [0,1]. Using
# an interval slightly wider than [0,1] (rather than sampling directly
# on [0,1]) is what gives the distribution nonzero probability mass
# exactly AT 0 and 1 after clipping -- these specific values are the
# ones used in the original paper and are not sensitive to small changes.
GAMMA = -0.1
ZETA = 1.1
BETA = 2.0 / 3.0  # temperature; controls how "hard" (bimodal) the distribution is


class HardConcreteGate(nn.Module):
    def __init__(self, num_units: int, init_log_alpha: float = 0.0):
        super().__init__()
        # log_alpha controls each unit's tendency to stay open (higher
        # -> more likely z stays near 1) vs. close (lower -> more likely
        # z gets clipped to 0). init_log_alpha=0 starts every gate at a
        # neutral 50/50 tendency, undecided until training pushes it
        # one way or the other.
        self.log_alpha = nn.Parameter(torch.full((num_units,), init_log_alpha))

    def forward(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / BETA)
        else:
            # Deterministic at eval time: no sampling, just the
            # distribution's "center" -- this is what makes eval
            # reproducible instead of different every forward pass.
            s = torch.sigmoid(self.log_alpha)
        s_stretched = s * (ZETA - GAMMA) + GAMMA
        z = torch.clamp(s_stretched, 0.0, 1.0)
        return z

    def expected_l0(self) -> torch.Tensor:
        """Expected number of gates that are nonzero, in closed form
        (Louizos et al. eq. 12) -- this is the differentiable stand-in
        for the true L0 count, used directly as the regularization loss."""
        return torch.sigmoid(self.log_alpha - BETA * math.log(-GAMMA / ZETA)).sum()

    def hard_gate_mask(self) -> torch.Tensor:
        """Deterministic 0/1 mask for reporting actual sparsity after
        training -- 'is this unit's eval-time gate exactly zero.'"""
        with torch.no_grad():
            z = self.forward()
            return (z > 0).float()
