"""CUDA rmsnorm correctness vs torch eager reference (also smoke-tests the JIT loader)."""
from __future__ import annotations

import pytest
import torch
from inference_kernel.kernels.norm.eager_impl import rmsnorm as rmsnorm_ref

from tests.conftest import assert_close_for_dtype


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(8, 16), (32, 64), (4, 16, 128), (2, 3, 1023), (8, 8, 8, 16384)], ids=str)
def test_rmsnorm_cuda_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    from inference_kernel.kernels.norm.cuda_impl import rmsnorm as rmsnorm_cuda

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[-1], dtype=dtype, device=device)
    got = rmsnorm_cuda(x, weight)
    expected = rmsnorm_ref(x, weight)
    assert_close_for_dtype(got, expected, dtype)


@pytest.mark.cuda
def test_rmsnorm_cuda_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    from inference_kernel.kernels.norm.cuda_impl import rmsnorm as rmsnorm_cuda

    x = torch.randn(3, 5, dtype=dtype, device=device)
    w = torch.randn(5, dtype=dtype, device=device)
    y = rmsnorm_cuda(x, w)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


@pytest.mark.cuda
def test_rmsnorm_cuda_custom_eps_matches_ref(device: torch.device) -> None:
    from inference_kernel.kernels.norm.cuda_impl import rmsnorm as rmsnorm_cuda

    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, device=device)
    w = torch.randn(32, dtype=torch.float32, device=device)
    eps = 1e-3
    got = rmsnorm_cuda(x, w, eps=eps)
    expected = rmsnorm_ref(x, w, eps=eps)
    torch.testing.assert_close(got, expected)


@pytest.mark.cuda
def test_rmsnorm_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.norm.cuda_impl import rmsnorm as rmsnorm_cuda

    x = torch.randn(8, 16)
    w = torch.randn(16)
    with pytest.raises((ValueError, RuntimeError)):
        rmsnorm_cuda(x, w)


@pytest.mark.cuda
def test_rmsnorm_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from inference_kernel.kernels.norm.cuda_impl import rmsnorm as rmsnorm_cuda

    x = torch.randn(8, 8, device=device).t()
    w = torch.randn(8, device=device)
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        rmsnorm_cuda(x, w)
