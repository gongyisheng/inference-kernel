import torch
import triton
import triton.language as tl

from ._utils import assert_contiguous, assert_is_cuda, assert_same_device, assert_same_dtype


def _gemm_configs():
    shapes = [
        (64, 64, 32), (64, 128, 32), (128, 64, 32),
        (128, 128, 32), (128, 128, 64),
        (128, 256, 64), (256, 128, 64),
    ]
    configs = []
    for block_m, block_n, block_k in shapes:
        for num_stages in (3, 4):
            num_warps = 8 if block_m * block_n >= 128 * 128 else 4
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                    num_stages=num_stages,
                    num_warps=num_warps,
                )
            )
    return configs


@triton.autotune(configs=_gemm_configs(), key=["M", "N", "K"])
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offset_m[:, None] < M) & (offset_k[None, :] + k < K)
        mask_b = (offset_k[:, None] + k < K) & (offset_n[None, :] < N)
        frag_a = tl.load(a_base_ptr, mask=mask_a, other=0.0)
        frag_b = tl.load(b_base_ptr, mask=mask_b, other=0.0)
        acc += tl.dot(frag_a, frag_b, input_precision="ieee")
        a_base_ptr += BLOCK_K * stride_ak
        b_base_ptr += BLOCK_K * stride_bk

    c_base_ptr = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_base_ptr, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, "gemm expects 2D inputs"
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "gemm shape mismatch"
    K = K1

    c = torch.empty((M, N), device=device, dtype=dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c