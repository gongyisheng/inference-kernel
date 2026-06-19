import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_contiguous, assert_same_device, assert_same_dtype

_ext = load_kernel(
    package="inference_kernel.kernels.norm",
    sources=["naive/rmsnorm.cu", "binding.cpp"],
)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm via custom CUDA kernel. Requires CUDA + contiguous inputs."""
    if not x.is_cuda:
        raise ValueError("cuda rmsnorm requires a CUDA tensor")
    assert_contiguous(x, weight)
    assert_same_device(x, weight)
    assert_same_dtype(x, weight)
    return _ext.rmsnorm_forward(x, weight, eps)
