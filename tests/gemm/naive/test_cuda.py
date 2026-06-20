"""CUDA (naive tier) gemm correctness vs ref eager reference (also smoke-tests the JIT loader)."""

import pytest
import torch
from inference_kernel.kernels.gemm.ref.eager_impl import gemm as gemm_ref

from tests.conftest import assert_close_for_gemm


@pytest.mark.cuda
@pytest.mark.parametrize(
    "shape",
    [(8, 16, 8), (32, 64, 32), (17, 33, 23), (64, 128, 32), (128, 256, 128)],
    ids=str,
)
def test_gemm_cuda_matches_ref(
    shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm as gemm_cuda

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_cuda(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_cuda_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm as gemm_cuda

    a = torch.randn(3, 5, dtype=dtype, device=device)
    b = torch.randn(5, 7, dtype=dtype, device=device)
    c = gemm_cuda(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype


@pytest.mark.cuda
def test_gemm_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm as gemm_cuda

    a = torch.randn(8, 16)
    b = torch.randn(16, 8)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_cuda(a, b)


@pytest.mark.cuda
def test_gemm_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm as gemm_cuda

    a = torch.randn(8, 8, device=device).t()
    b = torch.randn(8, 8, device=device)
    assert not a.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        gemm_cuda(a, b)


@pytest.mark.cuda
def test_gemm_cuda_rejects_mismatched_dtype(device: torch.device) -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm as gemm_cuda

    a = torch.randn(8, 8, device=device, dtype=torch.float32)
    b = torch.randn(8, 8, device=device, dtype=torch.float16)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_cuda(a, b)
