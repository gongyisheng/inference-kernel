"""CUDA norm kernels."""

import torch

from . import _C  # noqa: F401  (import side effect: registers torch.ops.aot_kernel.*)
from .utils import assert_contiguous, assert_same_device, assert_same_dtype


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm via custom CUDA kernel. Requires CUDA + contiguous inputs."""
    if not x.is_cuda:
        raise ValueError("cuda rmsnorm requires a CUDA tensor")
    assert_contiguous(x, weight)
    assert_same_device(x, weight)
    assert_same_dtype(x, weight)
    out = torch.empty_like(x)
    torch.ops.aot_kernel.rmsnorm_forward(out, x, weight, eps)
    return out
