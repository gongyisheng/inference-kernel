"""Torch attention invariants."""

import torch
from inference_kernel.kernels.attention.torch_impl import attention as attention_torch


def test_attention_torch_preserves_shape_and_dtype() -> None:
    Q = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    K = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    V = torch.randn(2, 4, 16, 32, dtype=torch.float32)
    out = attention_torch(Q, K, V)
    assert out.shape == (2, 4, 16, 32)
    assert out.dtype == Q.dtype
