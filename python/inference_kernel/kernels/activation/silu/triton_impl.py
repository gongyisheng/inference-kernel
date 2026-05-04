"""Triton implementation of SiLU.

One-dimensional element-wise kernel: each program handles BLOCK_SIZE
elements of the flattened input. Computes silu in fp32 to keep
half/bf16 precision predictable, then casts back.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from inference_kernel._common.utils import assert_contiguous


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
    """SiLU via Triton. Requires CUDA + contiguous input."""
    if not x.is_cuda:
        raise ValueError("triton silu requires a CUDA tensor")
    assert_contiguous(x)
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _silu_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
    return y
