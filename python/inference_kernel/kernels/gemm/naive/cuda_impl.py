import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_is_cuda, assert_contiguous, assert_same_device, assert_same_dtype

# Import for its registration side effect; ops are called via torch.ops below.
load_kernel(
    package="inference_kernel.kernels.gemm",
    sources=["naive/gemm.cu", "binding.cpp"],
)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM via custom CUDA kernel. Requires CUDA + contiguous inputs."""
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)
    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    torch.ops.inference_kernel.gemm(out, a, b)
    return out