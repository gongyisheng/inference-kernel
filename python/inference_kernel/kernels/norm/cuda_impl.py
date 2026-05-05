"""CUDA backends for norm kernels.

One compiled extension per category, loaded once at module import time
via the shared loader in inference_kernel._build.jit (AOT first, JIT
fallback). All norm entry points dispatch into _ext.
"""
from __future__ import annotations

import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_contiguous

_ext = load_kernel(
    package="inference_kernel.kernels.norm",
    sources=["rmsnorm.cu", "binding.cpp"],
)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm via custom CUDA kernel. Requires CUDA + contiguous inputs."""
    if not x.is_cuda:
        raise ValueError("cuda rmsnorm requires a CUDA tensor")
    assert_contiguous(x)
    assert_contiguous(weight)
    return _ext.rmsnorm_forward(x, weight, eps)
