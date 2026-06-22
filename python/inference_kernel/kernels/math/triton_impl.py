import torch
import triton
import triton.language as tl

from inference_kernel._common.utils import assert_is_cuda, assert_contiguous

@triton.jit
def _max_kernel(
    a_ptr,
    b_ptr,
    reduce_size,
    a_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    row_start = a_ptr + row * a_row_stride
    acc = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)

    for off in range(0, reduce_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < reduce_size
        frag = tl.load(row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        acc = tl.maximum(acc, frag)
    
    tl.store(b_ptr + row, tl.max(acc, axis=0))


def max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert_is_cuda(x)

    x = x.movedim(dim, -1).contiguous()
    reduce_size = x.shape[-1]
    out_shape = x.shape[:-1]

    x2d = x.reshape(-1, reduce_size)
    out = torch.empty(x2d.shape[0], device=x.device, dtype=x.dtype)

    grid = (x2d.shape[0],)
    _max_kernel[grid](
        x2d, out,
        reduce_size,
        x2d.stride(0),
        BLOCK_SIZE=1024
    )
    return out.reshape(out_shape)


@triton.jit
def _min_kernel(
    a_ptr,
    b_ptr,
    reduce_size,
    a_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    row_start = a_ptr + row * a_row_stride
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)

    for off in range(0, reduce_size, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask_a = col < reduce_size
        frag_a = tl.load(row_start + col, mask=mask_a, other=float("inf")).to(tl.float32)
        acc = tl.minimum(acc, frag_a)
    
    tl.store(b_ptr + row, tl.min(acc, axis=0))


def min(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert_is_cuda(x)

    x = x.movedim(dim, -1).contiguous()
    reduce_size = x.shape[-1]
    out_shape = x.shape[:-1]

    x2d = x.reshape(-1, reduce_size)
    out = torch.empty(x2d.shape[0], device=x.device, dtype=x.dtype)
    
    grid = (x2d.shape[0],)
    _min_kernel[grid](
        x2d, out,
        reduce_size,
        x2d.stride(0),
        BLOCK_SIZE=1024
    )
    return out.reshape(out_shape)