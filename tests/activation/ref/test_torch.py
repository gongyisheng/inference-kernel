"""Torch (ref tier) silu/relu invariants."""

import torch
from inference_kernel.kernels.activation.ref.torch_impl import relu as relu_torch
from inference_kernel.kernels.activation.ref.torch_impl import silu as silu_torch


def test_silu_torch_preserves_shape_and_dtype() -> None:
    x = torch.randn(3, 5, dtype=torch.float32)
    y = silu_torch(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_silu_torch_zero_input_is_zero() -> None:
    x = torch.zeros(4, dtype=torch.float32)
    y = silu_torch(x)
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_relu_torch_clamps_negatives_to_zero() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = relu_torch(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]))
