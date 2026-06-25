"""Triton flash attention correctness vs torch reference."""

import pytest
import torch
from inference_kernel.kernels.attention.torch_impl import attention as attn_ref
from inference_kernel.kernels.attention.triton_impl import attention as attn_triton

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
def test_attention_triton_preserves_shape_and_dtype(dtype, device) -> None:
    Q = torch.randn(2, 4, 64, 64, dtype=dtype, device=device)
    out = attn_triton(Q, Q, Q)
    assert out.shape == Q.shape
    assert out.dtype == Q.dtype
