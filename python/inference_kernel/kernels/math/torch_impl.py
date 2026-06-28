"""Torch reference for math kernels.

The correctness oracle for every other backend. Reductions accumulate in
fp32 and cast back to the input dtype.
"""

import torch


def max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Max reduction along `dim`."""
    return x.amax(dim=dim)


def min(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Min reduction along `dim`."""
    return x.amin(dim=dim)


def sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum reduction along `dim`, accumulated in fp32."""
    return x.float().sum(dim=dim).to(x.dtype)


def avg(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Mean reduction along `dim`, accumulated in fp32."""
    return x.float().mean(dim=dim).to(x.dtype)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along `dim`, computed in fp32."""
    return torch.softmax(x.float(), dim=dim).to(x.dtype)
