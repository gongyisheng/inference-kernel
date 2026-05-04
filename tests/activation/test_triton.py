"""Triton silu correctness vs torch reference."""
from __future__ import annotations

import pytest
import torch

from inference_kernel.kernels.activation.torch_impl import silu as silu_torch
from inference_kernel.kernels.activation.triton_impl import silu as silu_triton

from tests.conftest import assert_close_for_dtype


@pytest.mark.triton
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_silu_triton_matches_torch(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = silu_triton(x)
    expected = silu_torch(x)
    assert_close_for_dtype(got, expected, dtype)


@pytest.mark.triton
def test_silu_triton_non_contiguous_raises_or_handles(device: torch.device) -> None:
    """Spec says CUDA backends require contiguous input; triton should match."""
    x = torch.randn(8, 8, device=device).t()  # non-contiguous
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        from inference_kernel.kernels.activation.triton_impl import silu

        silu(x)
