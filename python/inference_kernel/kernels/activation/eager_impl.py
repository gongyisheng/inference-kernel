"""Torch eager reference for activation kernels.

These are the correctness oracles for every other backend (including the
faster torch impl in `torch_impl.py`). Keep them the simplest possible
correct implementations; do not optimize.
"""

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU: y = x * sigmoid(x). Element-wise; preserves shape and dtype."""
    return x * torch.sigmoid(x)
