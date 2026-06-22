import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_contiguous, assert_same_device, assert_same_dtype

# Import for its registration side effect; ops are called via torch.ops below.
load_kernel(
    package="inference_kernel.kernels.norm",
    sources=["rmsnorm.cu", "binding.cpp"],
)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm via custom CUDA kernel. Requires CUDA + contiguous inputs."""
    if not x.is_cuda:
        raise ValueError("cuda rmsnorm requires a CUDA tensor")
    assert_contiguous(x, weight)
    assert_same_device(x, weight)
    assert_same_dtype(x, weight)
    out = torch.empty_like(x)
    torch.ops.inference_kernel.rmsnorm_forward(out, x, weight, eps)
    return out
