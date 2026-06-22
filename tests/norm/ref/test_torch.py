"""Torch (ref tier) rmsnorm invariants."""

import torch
from inference_kernel.kernels.norm.ref.torch_impl import rmsnorm as rmsnorm_torch


def test_rmsnorm_torch_preserves_shape_and_dtype() -> None:
    x = torch.randn(3, 5, dtype=torch.float32)
    w = torch.randn(5, dtype=torch.float32)
    y = rmsnorm_torch(x, w)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
