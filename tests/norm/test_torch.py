"""Fast torch rmsnorm correctness vs torch reference."""
from __future__ import annotations

import pytest
import torch

from inference_kernel.kernels.norm.torch_impl import rmsnorm as rmsnorm_torch
from inference_kernel.kernels.norm.eager_impl import rmsnorm as rmsnorm_ref

from tests.conftest import assert_close_for_dtype


@pytest.mark.parametrize("shape", [(8, 16), (32, 64), (4, 16, 128), (2, 3, 1023)], ids=str)
def test_rmsnorm_torch_matches_ref(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    got = rmsnorm_torch(x, weight)
    expected = rmsnorm_ref(x, weight)
    assert_close_for_dtype(got, expected, dtype)


def test_rmsnorm_torch_preserves_shape_and_dtype() -> None:
    x = torch.randn(3, 5, dtype=torch.float32)
    w = torch.randn(5, dtype=torch.float32)
    y = rmsnorm_torch(x, w)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rmsnorm_torch_custom_eps_matches_ref() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32)
    w = torch.randn(32, dtype=torch.float32)
    eps = 1e-3
    got = rmsnorm_torch(x, w, eps=eps)
    expected = rmsnorm_ref(x, w, eps=eps)
    torch.testing.assert_close(got, expected)
