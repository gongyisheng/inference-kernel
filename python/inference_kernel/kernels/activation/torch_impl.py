"""Fast torch backend for activation kernels.

Uses fused PyTorch ops where available. Correctness is verified against
`eager_impl.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)
