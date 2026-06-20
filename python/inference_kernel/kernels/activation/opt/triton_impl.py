"""Optimized Triton activation kernels.

Same element-wise body as the naive tier, but ``@triton.autotune`` picks
``BLOCK_SIZE`` and ``num_warps`` per problem size (keyed on ``n_elements``),
and ``relu`` can write in place via ``out`` (pass ``out=x``).
"""

import torch
import triton
import triton.language as tl

from inference_kernel._common.utils import assert_contiguous, assert_is_cuda

_RELU_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
    for bs in (256, 512, 1024, 2048, 4096)
    for w in (2, 4, 8)
]


@triton.autotune(configs=_RELU_CONFIGS, key=["n_elements"])
@triton.jit
def _relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)


def relu(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """ReLU. Writes to ``out`` if given (``out=x`` for in-place), else allocates."""
    assert_is_cuda(x)
    assert_contiguous(x)
    if out is None:
        out = torch.empty_like(x)
    else:
        assert_is_cuda(out)
        assert_contiguous(out)
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out must match x in shape and dtype")
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _relu_kernel[grid](x, out, n)
    return out
