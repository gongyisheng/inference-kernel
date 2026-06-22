"""CUDA backends for gemm kernels.

One compiled extension per category registers both gemm ops. `gemm`
dispatches to the register-blocked tensor-core kernel for aligned fp16/bf16
and falls back to the naive WMMA kernel otherwise; `gemm_naive` always uses
the naive kernel.
"""

import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import (
    assert_contiguous,
    assert_is_cuda,
    assert_same_device,
    assert_same_dtype,
)

# Import for its registration side effect; ops are called via torch.ops below.
load_kernel(
    package="inference_kernel.kernels.gemm",
    sources=["gemm.cu", "gemm_opt.cu", "binding.cpp"],
)

_WK = 16  # WMMA fragment depth; tensor-core path requires K aligned to it.


def _validate(a: torch.Tensor, b: torch.Tensor):
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"gemm expects 2D tensors, got a.dim={a.dim()} b.dim={b.dim()}")
    if a.size(1) != b.size(0):
        raise ValueError(f"inner dims mismatch: {a.size(1)} vs {b.size(0)}")
    return device, dtype


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Register-blocked tensor-core GEMM. fp16/bf16 with K % 16 == 0 take the
    opt kernel; fp32 or K-unaligned shapes fall back to the naive kernel."""
    device, dtype = _validate(a, b)
    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16) and a.size(1) % _WK == 0:
        torch.ops.inference_kernel.gemm_opt(out, a, b)
    else:
        torch.ops.inference_kernel.gemm(out, a, b)
    return out


def gemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Naive WMMA GEMM (no register blocking)."""
    device, dtype = _validate(a, b)
    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    torch.ops.inference_kernel.gemm(out, a, b)
    return out
