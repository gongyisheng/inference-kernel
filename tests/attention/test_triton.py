"""Triton flash attention correctness vs torch reference."""

import pytest
import torch

from jit_kernel.attention import attention as attn_triton
from ref.attention import attention as attn_ref
from tests.conftest import assert_close_for_attention


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (2, 4, 128, 64), (1, 2, 200, 128), (2, 8, 100, 64)], ids=str)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_attention_triton_matches_ref(shape, causal, dtype, device) -> None:
    B, H, M, D = shape
    torch.manual_seed(0)
    Q = torch.randn(B, H, M, D, dtype=dtype, device=device)
    K = torch.randn(B, H, M, D, dtype=dtype, device=device)
    V = torch.randn(B, H, M, D, dtype=dtype, device=device)
    got = attn_triton(Q, K, V, is_causal=causal)
    expected = attn_ref(Q, K, V, is_causal=causal)
    assert_close_for_attention(got, expected, dtype)


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (2, 4, 128, 64), (1, 2, 200, 128), (2, 8, 100, 64)], ids=str)
@pytest.mark.parametrize("mask_type", ["float", "bool"], ids=["float_mask", "bool_mask"])
def test_attention_triton_attn_mask_matches_ref(shape, mask_type, dtype, device) -> None:
    B, H, M, D = shape
    N = M
    torch.manual_seed(0)
    Q = torch.randn(B, H, M, D, dtype=dtype, device=device)
    K = torch.randn(B, H, N, D, dtype=dtype, device=device)
    V = torch.randn(B, H, N, D, dtype=dtype, device=device)
    keep = torch.rand(M, N, device=device) > 0.3
    keep[:, 0] = True  # guarantee each query attends to at least one key
    if mask_type == "bool":
        attn_mask = keep
    else:
        attn_mask = torch.zeros(M, N, dtype=dtype, device=device).masked_fill(~keep, float("-inf"))
    got = attn_triton(Q, K, V, attn_mask=attn_mask)
    expected = attn_ref(Q, K, V, attn_mask=attn_mask)
    assert_close_for_attention(got, expected, dtype)


@pytest.mark.triton
def test_attention_triton_preserves_shape_and_dtype(dtype, device) -> None:
    Q = torch.randn(2, 4, 64, 64, dtype=dtype, device=device)
    out = attn_triton(Q, Q, Q)
    assert out.shape == Q.shape
    assert out.dtype == Q.dtype
