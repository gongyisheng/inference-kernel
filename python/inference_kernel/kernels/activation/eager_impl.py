"""Torch eager reference for activation kernels.

These are the correctness oracles for every other backend (including the
faster torch impl in `torch_impl.py`). Keep them the simplest possible
correct implementations; do not optimize.
"""

import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU: y = max(x, 0)"""
    return torch.clamp(x, min=0)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU: y = x * sigmoid(x)"""
    return x * torch.sigmoid(x)
