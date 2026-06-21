"""CUDA (opt tier) register-blocked gemm correctness vs ref eager reference."""

import pytest
import torch
from inference_kernel.kernels.gemm.ref.eager_impl import gemm as gemm_ref

from tests.conftest import assert_close_for_gemm


@pytest.mark.cuda
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
def test_gemm_opt_cuda_matches_ref(
    shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    from inference_kernel.kernels.gemm.opt.cuda_impl import gemm as gemm_opt

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_opt(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_opt_cuda_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    from inference_kernel.kernels.gemm.opt.cuda_impl import gemm as gemm_opt

    a = torch.randn(3, 5, dtype=dtype, device=device)
    b = torch.randn(5, 7, dtype=dtype, device=device)
    c = gemm_opt(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype


@pytest.mark.cuda
def test_gemm_opt_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.gemm.opt.cuda_impl import gemm as gemm_opt

    a = torch.randn(8, 16)
    b = torch.randn(16, 8)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_opt(a, b)
