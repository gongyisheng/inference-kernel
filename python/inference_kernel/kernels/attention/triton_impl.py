import torch

import triton
import triton.language as tl

from inference_kernel._common.utils import assert_is_cuda, assert_same_device, assert_same_dtype


@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, scale,
    qb_stride, qh_stride, qm_stride, qd_stride,
    kb_stride, kh_stride, km_stride, kd_stride,
    vb_stride, vh_stride, vm_stride, vd_stride,
    ob_stride, oh_stride, om_stride, od_stride,
    H, M, N,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # This program owns one Q-tile (BLOCK_M rows) for one (batch, head).
    tile_m = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)
    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    # Base pointer for this (batch, head). offset = sum of index * stride.
    q_base = q_ptr + batch_id * qb_stride + head_id * qh_stride
    k_base = k_ptr + batch_id * kb_stride + head_id * kh_stride
    v_base = v_ptr + batch_id * vb_stride + head_id * vh_stride
    o_base = o_ptr + batch_id * ob_stride + head_id * oh_stride

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)   # this tile's query rows
    offs_d = tl.arange(0, HEAD_DIM)                     # whole head dim, no tiling

    # Load the Q-tile once: [BLOCK_M, HEAD_DIM]. Stays in registers/SRAM the whole loop.
    q_ptrs = q_base + offs_m[:, None] * qm_stride + offs_d[None, :] * qd_stride
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    # Online-softmax state, one entry per query row.
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal: a row at position offs_m only attends to keys <= offs_m, so we can
    # stop the KV loop at the diagonal tile instead of scanning all of N.
    n_end = (tile_m + 1) * BLOCK_M if IS_CAUSAL else N

    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)         # this block's key/value rows

        # K loaded as [HEAD_DIM, BLOCK_N] so tl.dot(q, k) contracts over HEAD_DIM.
        k_ptrs = k_base + offs_n[None, :] * km_stride + offs_d[:, None] * kd_stride
        v_ptrs = v_base + offs_n[:, None] * vm_stride + offs_d[None, :] * vd_stride
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

        # GEMM #1: scores. input_precision="ieee" disables TF32 so fp32 stays exact.
        s = tl.dot(q, k, input_precision="ieee") * scale  # [BLOCK_M, BLOCK_N]

        # Mask out-of-range keys (padding) and, if causal, the upper triangle.
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(s, axis=1))        # running max
        p = tl.exp(s - m_new[:, None])                    # softmax numerator
        alpha = tl.exp(m_i - m_new)                        # rescale old state to new max
        l_i = alpha * l_i + tl.sum(p, axis=1)             # running denominator
        # GEMM #2: rescale the running output, then add this block's contribution.
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]                               # finalize the softmax divide

    o_ptrs = o_base + offs_m[:, None] * om_stride + offs_d[None, :] * od_stride
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < M)


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    assert_is_cuda(Q, K, V)
    dtype = assert_same_dtype(Q, K, V)
    device = assert_same_device(Q, K, V)
    assert attn_mask is None, "triton flash attention only supports is_causal, not attn_mask"

    B, H, M, D = Q.shape
    N = K.shape[2]
    scale = scale or 1.0 / (D ** 0.5)
    O = torch.empty_like(Q, dtype=dtype, device=device)

    # Large head dim needs smaller seq tiles to fit the tiles + accumulator in SRAM.
    BLOCK_M = BLOCK_N = 32 if D >= 128 else 64
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _flash_attn_kernel[grid](
        Q, K, V, O, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        H, M, N,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
    )
    return O  # B, H, M, D
