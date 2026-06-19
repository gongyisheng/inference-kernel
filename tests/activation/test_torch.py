"""Fast torch silu/relu correctness vs torch eager reference."""

import pytest
import torch
from inference_kernel.kernels.activation.reference.eager_impl import relu as relu_ref
from inference_kernel.kernels.activation.reference.eager_impl import silu as silu_ref
from inference_kernel.kernels.activation.reference.torch_impl import relu as relu_torch
from inference_kernel.kernels.activation.reference.torch_impl import silu as silu_torch

from tests.conftest import assert_close_for_relu, assert_close_for_silu


@pytest.mark.parametrize("shape", [(8,), (32, 64), (4, 16, 128)], ids=str)
def test_silu_torch_matches_ref(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    got = silu_torch(x)
    expected = silu_ref(x)
    assert_close_for_silu(got, expected, dtype)


def test_silu_torch_preserves_shape_and_dtype() -> None:
    x = torch.randn(3, 5, dtype=torch.float32)
    y = silu_torch(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_silu_torch_zero_input_is_zero() -> None:
    x = torch.zeros(4, dtype=torch.float32)
    y = silu_torch(x)
    torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.parametrize("shape", [(8,), (32, 64), (4, 16, 128)], ids=str)
def test_relu_torch_matches_ref(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    got = relu_torch(x)
    expected = relu_ref(x)
    assert_close_for_relu(got, expected, dtype)


def test_relu_torch_clamps_negatives_to_zero() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = relu_torch(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]))
