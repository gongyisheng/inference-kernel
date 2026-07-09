import torch
import triton
import triton.language as tl

from .utils import assert_contiguous, assert_is_cuda, assert_same_device, assert_same_dtype


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


@triton.autotune(configs=_gemm_configs(), key=["M", "N", "K"])
@triton.jit
def _scaled_gemm_kernel(
    a_ptr, b_ptr, scale_a_ptr, scale_b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scale_a, stride_scale_b,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    a_base_ptr = a_ptr + stride_am * offset_m[:, None] + stride_ak * offset_k[None, :] # [BLOCK_M, BLOCK_K]
    b_base_ptr = b_ptr + stride_bk * offset_k[:, None] + stride_bn * offset_n[None, :] # [BLOCK_K, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offset_m[:, None] < M) & (offset_k[None, :] + k < K)
        mask_b = (offset_k[:, None] + k < K) & (offset_n[None, :] < N)
        frag_a = tl.load(a_base_ptr, mask=mask_a, other=0.0)
        frag_b = tl.load(b_base_ptr, mask=mask_b, other=0.0)
        acc += tl.dot(frag_a, frag_b)
        a_base_ptr += BLOCK_K * stride_ak
        b_base_ptr += BLOCK_K * stride_bk
    
    frag_scale_a = tl.load(scale_a_ptr + stride_scale_a * offset_m, mask=(offset_m<M), other=0.0)
    frag_scale_b = tl.load(scale_b_ptr + stride_scale_b * offset_n, mask=(offset_n<N), other=0.0)
    acc = acc * frag_scale_a[:, None] * frag_scale_b[None, :]

    c_base_ptr = c_ptr + stride_cm * offset_m[:, None] + stride_cn * offset_n[None, :]
    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_base_ptr, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


def _scaled_gemm_opt_configs():
    # num_warps=4 consistently beats 8 for the fp8/int8 mma.sync path on sm_120
    # (measured: it keeps register pressure low enough for occupancy to hide the
    # ldmatrix->QMMA latency). BLOCK_K up to 128 favours int8; 64 favours fp8.
    configs = []
    for block_m, block_n in [(64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]:
        for block_k in (64, 128):
            for num_stages in (3, 4):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "GROUP_M": 8},
                        num_stages=num_stages,
                        num_warps=4,
                    )
                )
    return configs


@triton.autotune(configs=_scaled_gemm_opt_configs(), key=["M", "N", "K"])
@triton.jit
def _scaled_gemm_kernel_optimized(
    a_ptr, b_ptr, scale_a_ptr, scale_b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scale_a, stride_scale_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        frag_a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        frag_b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(frag_a, frag_b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    frag_scale_a = tl.load(scale_a_ptr + stride_scale_a * offset_m, mask=offset_m < M, other=0.0)
    frag_scale_b = tl.load(scale_b_ptr + stride_scale_b * offset_n, mask=offset_n < N, other=0.0)
    acc = acc * frag_scale_a[:, None] * frag_scale_b[None, :]

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    optimized: bool = True,
):
    assert a.dim() == 2 and b.dim() == 2, "gemm expects 2D inputs"
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    assert_same_dtype(a, b)
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "gemm shape mismatch"
    K = K1
    assert_is_cuda(scale_a, scale_b)
    assert scale_a.dtype == torch.float32
    assert scale_b.dtype == torch.float32
    sa_n, sb_n = scale_a.numel(), scale_b.numel()
    if sa_n == 1 and sb_n == 1:
        scale_mode = "tensor"
    elif sa_n == M and sb_n == N:
        scale_mode = "row"
    else:
        raise ValueError(
            "expect tensorwise (scalar scale_a and scale_b)"
            "or rowwise (scale_a.numel()==M, scale_b.numel()==N)"
        )

    stride_sa = 0 if scale_mode == "tensor" else scale_a.stride(0)
    stride_sb = 0 if scale_mode == "tensor" else scale_b.stride(0)
    c = torch.empty(M, N, dtype=out_dtype, device=device)
    args = (
        a, b, scale_a, scale_b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        stride_sa, stride_sb,
    )
    if optimized:
        # 1D grid: one program per output tile, remapped to (m, n) inside the kernel.
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        _scaled_gemm_kernel_optimized[grid](*args)
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _scaled_gemm_kernel[grid](*args)
    return c
    