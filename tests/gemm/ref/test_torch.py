"""Torch (ref tier) gemm invariants."""

import torch
from inference_kernel.kernels.gemm.ref.torch_impl import gemm as gemm_torch


def test_gemm_torch_preserves_shape_and_dtype() -> None:
    a = torch.randn(3, 5, dtype=torch.float32)
    b = torch.randn(5, 7, dtype=torch.float32)
    c = gemm_torch(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype
