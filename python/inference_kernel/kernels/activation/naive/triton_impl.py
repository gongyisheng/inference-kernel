"""Triton implementations of activation kernels.

Each entry point ships with its own @triton.jit kernel; element-wise
activations use a 1-D flat grid. Compute happens in fp32 for predictable
half/bf16 precision, then casts back on store.
"""

import torch
import triton
import triton.language as tl

from inference_kernel._common.utils import assert_contiguous, assert_is_cuda


@triton.jit
def _relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _relu_kernel[grid](x, y, n, BLOCK_SIZE)
    return y


@triton.jit
def _silu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(x)
    tl.store(y_ptr + offsets, y, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    assert_is_cuda(x)
    assert_contiguous(x)
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _silu_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
    return y
