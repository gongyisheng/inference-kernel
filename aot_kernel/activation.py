"""CUDA activation kernels. Ops are registered under the `aot_kernel` torch
namespace by importing the compiled `_C` extension."""

import torch

from . import _C  # noqa: F401  (import side effect: registers torch.ops.aot_kernel.*)
from ._utils import assert_contiguous, assert_is_cuda


def relu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    out = torch.empty_like(x)
    torch.ops.aot_kernel.relu_forward(out, x)
    return out


def silu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    out = torch.empty_like(x)
    torch.ops.aot_kernel.silu_forward(out, x)
    return out
