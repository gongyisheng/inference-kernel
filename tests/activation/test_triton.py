"""Triton silu/relu correctness vs torch eager reference."""

import pytest
import torch
from inference_kernel.kernels.activation.reference.eager_impl import relu as relu_ref
from inference_kernel.kernels.activation.reference.eager_impl import silu as silu_ref
from inference_kernel.kernels.activation.naive.triton_impl import relu as relu_triton
from inference_kernel.kernels.activation.naive.triton_impl import silu as silu_triton

from tests.conftest import assert_close_for_relu, assert_close_for_silu


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_silu_triton_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = silu_triton(x)
    expected = silu_ref(x)
    assert_close_for_silu(got, expected, dtype)


@pytest.mark.triton
def test_silu_triton_non_contiguous_raises_or_handles(device: torch.device) -> None:
    """Spec says CUDA backends require contiguous input; triton should match."""
    x = torch.randn(8, 8, device=device).t()  # non-contiguous
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        from inference_kernel.kernels.activation.naive.triton_impl import silu

        silu(x)


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_relu_triton_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = relu_triton(x)
    expected = relu_ref(x)
    assert_close_for_relu(got, expected, dtype)


@pytest.mark.triton
def test_relu_triton_non_contiguous_raises_or_handles(device: torch.device) -> None:
    x = torch.randn(8, 8, device=device).t()  # non-contiguous
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        relu_triton(x)
