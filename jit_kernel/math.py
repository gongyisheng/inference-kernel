import torch
import triton
import triton.language as tl

from ._utils import assert_is_cuda


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
    
    tl.store(b_ptr + row, tl.max(acc, axis=0).to(b_ptr.dtype.element_ty))


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
    
    tl.store(b_ptr + row, tl.min(acc, axis=0).to(b_ptr.dtype.element_ty))


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


@triton.jit
def _sum_kernel(
    a_ptr,
    b_ptr,
    reduce_size,
    a_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    a_start = a_ptr + a_row_stride * row
    acc = tl.full((BLOCK_SIZE,), 0, dtype=tl.float32)

    for off in range(0, reduce_size, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask = col < reduce_size
        frag = tl.load(a_start + col, mask = mask, other=0.0).to(tl.float32)
        acc = tl.add(acc, frag)
    
    tl.store(b_ptr + row, tl.sum(acc, axis=0).to(b_ptr.dtype.element_ty))        


def sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert_is_cuda(x)

    x = x.movedim(dim, -1).contiguous()
    reduce_size = x.shape[-1]
    out_shape = x.shape[:-1]

    x2d = x.reshape(-1, reduce_size)
    out = torch.empty(x2d.shape[0], dtype=x.dtype, device=x.device)
    grid = (x2d.shape[0],)
    _sum_kernel[grid](
        x2d, out,
        reduce_size,
        x2d.stride(0),
        BLOCK_SIZE=1024
    )
    return out.reshape(out_shape)


@triton.jit
def _avg_kernel(
    a_ptr, 
    b_ptr,
    reduce_size,
    a_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    a_start = a_ptr + a_row_stride * row
    acc = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    for off in range(0, reduce_size, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask = col < reduce_size
        frag_a = tl.load(a_start + col, mask=mask, other=0.0).to(tl.float32)
        acc = tl.add(acc, frag_a)

    tl.store(b_ptr + row, (tl.sum(acc, axis=0) / reduce_size).to(b_ptr.dtype.element_ty))


def avg(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert_is_cuda(x)

    x = x.movedim(dim, -1).contiguous()
    reduce_size = x.shape[-1]
    out_shape = x.shape[:-1]

    x2d = x.reshape(-1, reduce_size)
    out = torch.empty(x2d.shape[0], dtype=x.dtype, device=x.device)
    grid = (x2d.shape[0],)

    _avg_kernel[grid](
        x2d, out,
        reduce_size,
        x2d.stride(0),
        BLOCK_SIZE=1024
    )
    return out.reshape(out_shape)


@triton.jit
def _softmax_kernel(
    a_ptr,
    b_ptr,
    reduce_size,
    a_row_stride,
    b_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(axis=0)
    a_start = a_ptr + row * a_row_stride
    b_start = b_ptr + row * b_row_stride
    
    m = float("-inf")
    denom = 0.0

    for off in range(0, reduce_size, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask = col < reduce_size
        frag = tl.load(a_start + col, mask=mask, other=float("-inf")).to(tl.float32)
        _m = tl.max(frag)
        if _m > m:
            denom = denom * tl.exp(m - _m)
            m = _m
        denom += tl.sum(tl.exp(frag - m))
    
    for off in range(0, reduce_size, BLOCK_SIZE):
        col = off + tl.arange(0, BLOCK_SIZE)
        mask = col < reduce_size
        frag = tl.load(a_start + col, mask=mask, other=float("-inf")).to(tl.float32)
        out = tl.exp(frag - m) / denom
        tl.store(b_start + col, out.to(b_ptr.dtype.element_ty), mask=mask)
            

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert_is_cuda(x)

    x = x.movedim(dim, -1).contiguous()
    moved_shape = x.shape
    
    x2d = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x2d, dtype=x.dtype, device=x.device)
    reduce_size = x2d.shape[-1]
    
    grid = (x2d.shape[0],)
    _softmax_kernel[grid](
        x2d, out,
        reduce_size,
        x2d.stride(0),
        out.stride(0),
        BLOCK_SIZE=1024
    )
    return out.reshape(moved_shape).movedim(-1, dim)