import torch
import triton
import triton.language as tl

from inference_kernel._common.utils import assert_contiguous, assert_same_device, assert_same_dtype


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    a_base_ptr = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_base_ptr = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
    c_base_ptr = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offset_m[:, None]<M) & ((offset_k[None, :]+k)<K)
        mask_b = ((offset_k[:, None]+k)<K) & (offset_n[None, :]<N)
        a = tl.load(a_base_ptr, mask=mask_a, other=0.0)
        b = tl.load(b_base_ptr, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_base_ptr += BLOCK_K * stride_ak
        b_base_ptr += BLOCK_K * stride_bk

    c_base_ptr = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    c_mask = (offset_m[:, None]<M) & (offset_n[None, :]<N)
    tl.store(c_base_ptr, acc.to(c_ptr.dtype.element_ty), c_mask)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton gemm requires CUDA tensors")
    assert_contiguous(a)
    assert_contiguous(b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)

    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "gemm shape mismatch"
    K = K1
    c = torch.empty((M, N), device=device, dtype=dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c