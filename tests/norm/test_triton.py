"""Triton rmsnorm correctness vs torch eager reference."""

import pytest
import torch
from inference_kernel.kernels.norm.ref.eager_impl import rmsnorm as rmsnorm_ref
from inference_kernel.kernels.norm.naive.triton_impl import rmsnorm as rmsnorm_triton

from tests.conftest import assert_close_for_rmsnorm


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(8, 16), (32, 64), (4, 16, 128), (2, 3, 1023), (8, 8, 8, 16384)], ids=str)
def test_rmsnorm_triton_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[-1], dtype=dtype, device=device)
    got = rmsnorm_triton(x, weight)
    expected = rmsnorm_ref(x, weight)
    assert_close_for_rmsnorm(got, expected, dtype)


@pytest.mark.triton
def test_rmsnorm_triton_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.randn(3, 5, dtype=dtype, device=device)
    w = torch.randn(5, dtype=dtype, device=device)
    y = rmsnorm_triton(x, w)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


@pytest.mark.triton
def test_rmsnorm_triton_custom_eps_matches_ref(device: torch.device) -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, device=device)
    w = torch.randn(32, dtype=torch.float32, device=device)
    eps = 1e-3
    got = rmsnorm_triton(x, w, eps=eps)
    expected = rmsnorm_ref(x, w, eps=eps)
    torch.testing.assert_close(got, expected)


@pytest.mark.triton
def test_rmsnorm_triton_non_contiguous_raises(device: torch.device) -> None:
    """Triton backend requires contiguous input; non-contiguous must raise."""
    x = torch.randn(8, 8, device=device).t()
    w = torch.randn(8, device=device)
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        rmsnorm_triton(x, w)
