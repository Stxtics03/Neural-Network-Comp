"""
LeNet-5, the classic 1998 CNN architecture -- small enough to train
fast on CPU, standard enough that its baseline accuracy on MNIST
(~99%) is well documented, so we have a real number to sanity-check
against once we've trained our own.

Architecture (LeCun et al. 1998, lightly modernized: ReLU instead of
tanh, max-pool instead of average-pool -- these are standard swaps
that don't change the spirit of the architecture, just make it easier
to train with modern optimizers):

  input (1x28x28)
    -> conv1 (6 filters, 5x5)   -> ReLU -> maxpool 2x2
    -> conv2 (16 filters, 5x5)  -> ReLU -> maxpool 2x2
    -> flatten
    -> fc1 (120 units) -> ReLU
    -> fc2 (84 units)  -> ReLU
    -> fc3 (10 units, one per digit class)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # padding=2 on conv1 turns 28x28 input into an effective 32x32
        # (LeNet-5's original input size) so the spatial dims work out
        # to exactly 5x5 by the time we reach fc1 -- no fiddly cropping.
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14x14 -> 5x5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no softmax here -- CrossEntropyLoss applies it internally
        return x

    def prunable_layers(self) -> list[tuple[str, nn.Module]]:
        """Conv/Linear layers whose output units we'll later cluster or
        prune. Named so later phases can log which layer they're touching."""
        return [
            ("conv1", self.conv1), ("conv2", self.conv2),
            ("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3),
        ]


def build_model(num_classes: int = 10) -> LeNet5:
    return LeNet5(num_classes=num_classes)
