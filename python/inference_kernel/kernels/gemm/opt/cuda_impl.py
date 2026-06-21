import torch

# Importing the naive impl triggers the single category extension build
# (one .so registers both `gemm` and `gemm_opt`); reuse its registration here.
from inference_kernel.kernels.gemm.naive import cuda_impl as _naive  # noqa: F401
from inference_kernel._common.utils import assert_is_cuda, assert_contiguous, assert_same_device, assert_same_dtype

_WK = 16  # WMMA fragment depth; tensor-core path requires K aligned to it.


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Register-blocked tensor-core GEMM. fp16/bf16 with K % 16 == 0 take the
    opt kernel; fp32 or K-unaligned shapes fall back to the naive tier."""
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"gemm expects 2D tensors, got a.dim={a.dim()} b.dim={b.dim()}")
    if a.size(1) != b.size(0):
        raise ValueError(f"inner dims mismatch: {a.size(1)} vs {b.size(0)}")

    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    if dtype in (torch.float16, torch.bfloat16) and a.size(1) % _WK == 0:
        torch.ops.inference_kernel.gemm_opt(out, a, b)
    else:
        torch.ops.inference_kernel.gemm(out, a, b)
    return out
