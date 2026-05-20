"""Torch eager reference for norm kernels.

These are the correctness oracles for every other backend (including the
faster torch impl in `torch_impl.py`). Keep them the simplest possible
correct implementations; do not optimize.
"""

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight, last-dim reduction."""
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(var + eps)
    norm = x_fp32 * rms * weight.to(torch.float32)
    return norm.to(x.dtype)
