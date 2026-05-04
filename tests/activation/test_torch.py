"""Correctness tests for the torch reference of silu."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from inference_kernel.kernels.activation.torch_impl import silu

from tests.conftest import assert_close_for_dtype


@pytest.mark.parametrize("shape", [(8,), (32, 64), (4, 16, 128)], ids=str)
def test_silu_torch_matches_F_silu(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    got = silu(x)
    expected = F.silu(x)
    assert_close_for_dtype(got, expected, dtype)


def test_silu_torch_preserves_shape_and_dtype() -> None:
    x = torch.randn(3, 5, dtype=torch.float32)
    y = silu(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_silu_torch_zero_input_is_zero() -> None:
    x = torch.zeros(4, dtype=torch.float32)
    y = silu(x)
    torch.testing.assert_close(y, torch.zeros_like(y))
