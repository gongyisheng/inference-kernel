"""CUDA backends for activation kernels.

One compiled extension per category, loaded once at module import time
via the shared loader in inference_kernel._build.jit (AOT first, JIT
fallback). All activation entry points dispatch into _ext.
"""

import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_contiguous, assert_is_cuda

_ext = load_kernel(
    package="inference_kernel.kernels.activation",
    sources=["naive/silu.cu", "naive/relu.cu", "binding.cpp"],
)


def relu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    return _ext.relu_forward(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    return _ext.silu_forward(x)
