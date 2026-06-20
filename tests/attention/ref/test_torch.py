"""Torch (ref tier) attention correctness vs eager reference."""

import pytest
import torch
from inference_kernel.kernels.attention.ref.eager_impl import attention as attention_ref
from inference_kernel.kernels.attention.ref.torch_impl import attention as attention_torch

from tests.conftest import assert_close_for_attention


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 8, 8), (2, 4, 16, 32), (1, 8, 33, 64), (4, 8, 128, 64)],
    ids=str,
)
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_attention_torch_matches_ref(
    shape: tuple[int, int, int, int], is_causal: bool, dtype: torch.dtype
) -> None:
    B, H, N, D = shape
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, dtype=dtype)
    K = torch.randn(B, H, N, D, dtype=dtype)
    V = torch.randn(B, H, N, D, dtype=dtype)
    got = attention_torch(Q, K, V, is_causal=is_causal)
    expected = attention_ref(Q, K, V, is_causal=is_causal)
    assert_close_for_attention(got, expected, dtype)


def test_attention_torch_custom_scale() -> None:
    torch.manual_seed(0)
    Q = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    K = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    V = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    got = attention_torch(Q, K, V, scale=0.1)
    expected = attention_ref(Q, K, V, scale=0.1)
    assert_close_for_attention(got, expected, torch.float32)


def test_attention_torch_preserves_shape_and_dtype() -> None:
    Q = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    K = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    V = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    out = attention_torch(Q, K, V)
    assert out.shape == (2, 4, 16, 32)
    assert out.dtype == Q.dtype
