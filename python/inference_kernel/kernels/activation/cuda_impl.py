"""CUDA backends for activation kernels.

One compiled extension per category. Importing it (via the shared loader in
inference_kernel._build.jit, AOT first then JIT) runs the TORCH_LIBRARY_FRAGMENT
static initializers, registering the ops under the `inference_kernel`
namespace. Entry points then dispatch through torch.ops.inference_kernel.
"""

import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import assert_contiguous, assert_is_cuda

# Import for its registration side effect; ops are called via torch.ops below.
load_kernel(
    package="inference_kernel.kernels.activation",
    sources=["naive/silu.cu", "naive/relu.cu", "binding.cpp"],
)


def relu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    out = torch.empty_like(x)
    torch.ops.inference_kernel.relu_forward(out, x)
    return out


def silu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    out = torch.empty_like(x)
    torch.ops.inference_kernel.silu_forward(out, x)
    return out
