"""
MNIST loaders. Standard normalization constants (0.1307, 0.3081) are
MNIST's actual per-pixel mean/std over the training set -- not
arbitrary, they're what make the input roughly zero-mean/unit-variance,
which helps optimization. First run downloads to ./data_cache/
(needs internet).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_loaders(batch_size: int = 64, data_dir: str = "./data_cache") -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader
