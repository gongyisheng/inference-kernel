"""CUDA backends for activation kernels.

One compiled extension per category, loaded once at module import time
via the shared loader in inference_kernel._build.jit (AOT first, JIT
fallback). All activation entry points dispatch into _ext.
"""

import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_contiguous

_ext = load_kernel(
    package="inference_kernel.kernels.activation",
    sources=["silu.cu", "binding.cpp"],
)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU via custom CUDA kernel. Requires CUDA + contiguous input."""
    if not x.is_cuda:
        raise ValueError("cuda silu requires a CUDA tensor")
    assert_contiguous(x)
    return _ext.silu_forward(x)
