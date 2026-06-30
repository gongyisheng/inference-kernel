"""Triton gemm correctness vs torch reference."""

import pytest
import torch

from jit_kernel.gemm import gemm as gemm_triton
from ref.gemm import gemm as gemm_ref
from tests.conftest import assert_close_for_gemm


@pytest.mark.triton
@pytest.mark.parametrize(
    "shape",
    [(8, 16, 8), (32, 64, 32), (17, 33, 23), (64, 128, 32), (128, 256, 128)],
    ids=str,
)
def test_gemm_triton_matches_ref(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_triton(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.triton
def test_gemm_triton_preserves_shape_and_dtype(
    dtype: torch.dtype, device: torch.device
) -> None:
    a = torch.randn(3, 5, dtype=dtype, device=device)
    b = torch.randn(5, 7, dtype=dtype, device=device)
    c = gemm_triton(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype


@pytest.mark.triton
def test_gemm_triton_rejects_cpu_tensor() -> None:
    a = torch.randn(8, 16)
    b = torch.randn(16, 8)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_triton(a, b)


@pytest.mark.triton
def test_gemm_triton_non_contiguous_raises(device: torch.device) -> None:
    """Triton backend requires contiguous inputs; non-contiguous must raise."""
    a = torch.randn(8, 8, device=device).t()
    b = torch.randn(8, 8, device=device)
    assert not a.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        gemm_triton(a, b)
