"""Fast torch backend for norm kernels.

Uses fused PyTorch ops where available. Correctness is verified against
`eager_impl.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, weight.shape, weight, eps)
