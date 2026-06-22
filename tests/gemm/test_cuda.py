"""CUDA gemm correctness vs torch reference (also smoke-tests the JIT loader).

Covers both the dispatching `gemm` (opt tensor-core kernel, naive fallback)
and `gemm_naive` (always the naive WMMA kernel).
"""

import pytest
import torch
from inference_kernel.kernels.gemm.torch_impl import gemm as gemm_ref

from tests.conftest import assert_close_for_gemm


@pytest.mark.cuda
@pytest.mark.parametrize("impl", ["gemm", "gemm_naive"])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 16, 8),        # smaller than one block tile (partial M/N)
        (32, 64, 32),
        (17, 33, 23),      # K=33 unaligned -> naive fallback
        (64, 128, 32),
        (128, 256, 128),   # exactly one block tile wide
        (130, 64, 130),    # partial trailing block in M and N
        (256, 256, 256),   # multiple full block tiles
    ],
    ids=str,
)
def test_gemm_cuda_matches_ref(
    impl: str, shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    import inference_kernel.kernels.gemm.cuda_impl as cuda_impl
    gemm_cuda = getattr(cuda_impl, impl)

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_cuda(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_cuda_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    from inference_kernel.kernels.gemm.cuda_impl import gemm as gemm_cuda

    a = torch.randn(3, 5, dtype=dtype, device=device)
    b = torch.randn(5, 7, dtype=dtype, device=device)
    c = gemm_cuda(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype


@pytest.mark.cuda
def test_gemm_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.gemm.cuda_impl import gemm as gemm_cuda

    a = torch.randn(8, 16)
    b = torch.randn(16, 8)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_cuda(a, b)


@pytest.mark.cuda
def test_gemm_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from inference_kernel.kernels.gemm.cuda_impl import gemm as gemm_cuda

    a = torch.randn(8, 8, device=device).t()
    b = torch.randn(8, 8, device=device)
    assert not a.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        gemm_cuda(a, b)


@pytest.mark.cuda
def test_gemm_cuda_rejects_mismatched_dtype(device: torch.device) -> None:
    from inference_kernel.kernels.gemm.cuda_impl import gemm as gemm_cuda

    a = torch.randn(8, 8, device=device, dtype=torch.float32)
    b = torch.randn(8, 8, device=device, dtype=torch.float16)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_cuda(a, b)
