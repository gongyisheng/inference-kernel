"""Fast torch backend for activation kernels.

Uses fused PyTorch ops where available. Correctness is verified against
`eager_impl.py`.
"""

import torch
import torch.nn.functional as F


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)
