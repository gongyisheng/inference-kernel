"""Fast torch gemm correctness vs torch reference."""

import pytest
import torch
from inference_kernel.kernels.gemm.reference.eager_impl import gemm as gemm_ref
from inference_kernel.kernels.gemm.reference.torch_impl import gemm as gemm_torch

from tests.conftest import assert_close_for_gemm


@pytest.mark.parametrize(
    "shape",
    [(8, 16, 8), (32, 64, 32), (17, 33, 23), (64, 128, 32), (128, 256, 128)],
    ids=str,
)
def test_gemm_torch_matches_ref(shape: tuple[int, int, int], dtype: torch.dtype) -> None:
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype)
    b = torch.randn(K, N, dtype=dtype)
    got = gemm_torch(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


def test_gemm_torch_preserves_shape_and_dtype() -> None:
    a = torch.randn(3, 5, dtype=torch.float32)
    b = torch.randn(5, 7, dtype=torch.float32)
    c = gemm_torch(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype
