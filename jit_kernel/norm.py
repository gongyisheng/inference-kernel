
import torch
import triton
import triton.language as tl

from ._utils import assert_contiguous, assert_is_cuda, assert_same_device, assert_same_dtype


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    x_base_ptr = x_ptr + row_idx * stride
    y_base_ptr = y_ptr + row_idx * stride
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    x = tl.load(x_base_ptr + offset, mask, other=0.0)
    w = tl.load(w_ptr + offset, mask, other=0.0)
    dtype = x.dtype
    x = x.to(tl.float32)
    w = w.to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd * w).to(dtype)

    tl.store(y_base_ptr + offset, y, mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert_is_cuda(x, weight)
    assert_contiguous(x, weight)
    device = assert_same_device(x, weight)
    dtype = assert_same_dtype(x, weight)
    y = torch.empty_like(x, device=device, dtype=dtype)

    N = x.shape[-1] 
    M = x.numel() // N
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    _rmsnorm_kernel[grid](x, weight, y, N, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y
