import torch
import triton
import triton.language as tl

from ._utils import assert_contiguous, assert_is_cuda

_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
    for bs in (256, 512, 1024, 2048, 4096)
    for w in (2, 4, 8)
    for s in (2, 3, 4)
]


@triton.autotune(configs=_CONFIGS, key=["n_elements"])
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
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _relu_kernel[grid](x, y, n)
    return y


@triton.autotune(configs=_CONFIGS, key=["n_elements"])
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
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _silu_kernel[grid](x, y, n)
    return y
