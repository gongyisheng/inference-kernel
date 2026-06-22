"""Torch reference for norm kernels.

The correctness oracle for every other backend. Uses fused PyTorch ops
where available.
"""

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, weight.shape, weight, eps)
