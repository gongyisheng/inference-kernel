"""Torch reference for activation kernels.

The correctness oracle for every other backend. Uses fused PyTorch ops
where available.
"""

import torch
import torch.nn.functional as F


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)
