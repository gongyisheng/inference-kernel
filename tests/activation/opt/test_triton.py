"""Triton (opt tier) relu correctness vs ref eager reference."""

import pytest
import torch
from inference_kernel.kernels.activation.ref.eager_impl import relu as relu_ref
from inference_kernel.kernels.activation.opt.triton_impl import relu as relu_triton_opt

from tests.conftest import assert_close_for_relu


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_relu_triton_opt_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    """Autotune picks the launch config; kernel must stay correct across shapes."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = relu_triton_opt(x)
    expected = relu_ref(x)
    assert_close_for_relu(got, expected, dtype)


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (1023,)], ids=str)
def test_relu_triton_opt_inplace(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    """out=x writes the result back into the input buffer."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    expected = relu_ref(x)
    out = relu_triton_opt(x, out=x)
    assert out.data_ptr() == x.data_ptr()
    assert_close_for_relu(out, expected, dtype)


@pytest.mark.triton
def test_relu_triton_opt_non_contiguous_raises_or_handles(device: torch.device) -> None:
    x = torch.randn(8, 8, device=device).t()  # non-contiguous
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        relu_triton_opt(x)
