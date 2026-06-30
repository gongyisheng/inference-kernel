import torch
import triton
import triton.language as tl

from ._utils import assert_is_cuda, assert_same_device, assert_same_dtype


def _attn_configs() -> list[triton.Config]:
    # Curated set: large tiles for small head dims, smaller spill-safe tiles
    # for large head dims / mask. Configs that exceed SRAM are pruned by triton.
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    ]


@triton.autotune(configs=_attn_configs(), key=["M", "N", "HEAD_DIM"])
@triton.jit
def _flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, attn_mask_ptr, scale,
    qb_stride, qh_stride, qm_stride, qd_stride,
    kb_stride, kh_stride, kn_stride, kd_stride,
    vb_stride, vh_stride, vn_stride, vd_stride,
    ob_stride, oh_stride, om_stride, od_stride,
    H, M, N,
    IS_CAUSAL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Q: (block_m, D)
    # K: (D, block_n)
    # S: (block_m, block_n)
    # V: (block_n, D)
    # O: (block_m, D)
    # attn_mask: (M, N)

    m_id = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    # exp2 maps to a hardware instruction; fold ln(2) in via log2(e). 
    # s stays in natural units so the -inf masking above is unaffected.
    LOG2E: tl.constexpr = 1.4426950408889634

    q_base_ptr = q_ptr + qb_stride * batch_id + qh_stride * head_id
    k_base_ptr = k_ptr + kb_stride * batch_id + kh_stride * head_id
    v_base_ptr = v_ptr + vb_stride * batch_id + vh_stride * head_id
    o_base_ptr = o_ptr + ob_stride * batch_id + oh_stride * head_id

    offset_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_d = tl.arange(0, HEAD_DIM)

    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    mask_q = offset_m[:, None] < M
    q = tl.load(q_base_ptr + qm_stride * offset_m[:, None] + qd_stride * offset_d[None, :], mask=mask_q, other=0.0)

    n_end = tl.minimum((m_id + 1) * BLOCK_M, N) if IS_CAUSAL else N

    for n in range(0, n_end, BLOCK_N):
        offset_n = n + tl.arange(0, BLOCK_N)
        mask_k = offset_n[None, :] < N
        k = tl.load(k_base_ptr + kd_stride * offset_d[:, None] + kn_stride * offset_n[None, :], mask=mask_k, other=0.0)
        s = tl.dot(q, k, input_precision="ieee") * scale
        s = tl.where(offset_n[None, :] < N, s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offset_m[:, None] >= offset_n[None, :], s, float("-inf"))
        elif HAS_ATTN_MASK:
            _mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
            attn_mask = tl.load(attn_mask_ptr + N * offset_m[:, None] + offset_n[None, :], mask=_mask, other=float("-inf"))
            s = s + attn_mask
        _m = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp2((s - _m[:, None]) * LOG2E)
        alpha = tl.exp2((m - _m) * LOG2E)
        _l = tl.sum(p, axis=1)
        l = l * alpha + _l

        mask_v = offset_n[:, None] < N
        v = tl.load(v_base_ptr + vn_stride * offset_n[:, None] + vd_stride * offset_d[None, :], mask=mask_v, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m = _m

    acc = acc / l[:, None]
    mask_o = offset_m[:, None] < M
    tl.store(o_base_ptr + om_stride * offset_m[:, None] + od_stride * offset_d[None, :], acc, mask=mask_o)


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
    assert not (is_causal and attn_mask is not None), "pass only one of is_causal / attn_mask"

    QB, QH, QM, QD = Q.shape
    KB, KH, KN, KD = K.shape
    VB, VH, VN, VD = V.shape

    assert QB == KB == VB
    assert QH == KH == VH
    assert QD == KD == VD
    assert KN == VN
    if attn_mask is not None:
        mask_M, mask_N = attn_mask.shape
        assert mask_M == QM
        assert mask_N == KN

    B = QB
    H = QH
    M = QM
    N = KN
    D = QD

    scale = scale or 1.0 / (D ** 0.5)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            _mask = torch.zeros_like(attn_mask, dtype=torch.float32)
            attn_mask = _mask.masked_fill(~attn_mask, float("-inf"))
        elif attn_mask.dtype in (torch.float16, torch.bfloat16, torch.float32):
            attn_mask = attn_mask.to(torch.float32)
        else:
            raise ValueError(f"attn mask dtype only support float and bool, get{attn_mask.dtype}")

    O = torch.empty_like(Q, dtype=dtype, device=device)

    # Block sizes / warps / stages are autotuned; configs that spill SRAM
    # (large head dim or a mask tile) are pruned by triton automatically.
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), B * H)
    _flash_attn_kernel[grid](
        Q, K, V, O, attn_mask, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        H, M, N,
        IS_CAUSAL=is_causal,
        HAS_ATTN_MASK=attn_mask is not None,
        HEAD_DIM=D,
    )
    return O  # B, H, M, D
