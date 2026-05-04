"""Torch eager reference for SiLU (a.k.a. swish).

This is the correctness oracle for the triton and cuda backends.
Keep it the simplest possible correct implementation; do not optimize.
"""
from __future__ import annotations

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU: y = x * sigmoid(x). Element-wise; preserves shape and dtype."""
    return x * torch.sigmoid(x)
