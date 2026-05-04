"""Torch eager references for activation kernels.

These are the correctness oracles for the triton and cuda backends.
Keep them the simplest possible correct implementations; do not optimize.
"""
from __future__ import annotations

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU: y = x * sigmoid(x). Element-wise; preserves shape and dtype."""
    return x * torch.sigmoid(x)
