"""CUDA silu correctness vs torch eager reference (also smoke-tests the JIT loader)."""
from __future__ import annotations

import pytest
import torch

from inference_kernel.kernels.activation.eager_impl import silu as silu_ref

from tests.conftest import assert_close_for_dtype


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_silu_cuda_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    from inference_kernel.kernels.activation.cuda_impl import silu as silu_cuda

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = silu_cuda(x)
    expected = silu_ref(x)
    assert_close_for_dtype(got, expected, dtype)


@pytest.mark.cuda
def test_silu_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.activation.cuda_impl import silu as silu_cuda

    x = torch.randn(8)  # CPU
    with pytest.raises((ValueError, RuntimeError)):
        silu_cuda(x)


@pytest.mark.cuda
def test_silu_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from inference_kernel.kernels.activation.cuda_impl import silu as silu_cuda

    x = torch.randn(8, 8, device=device).t()
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        silu_cuda(x)
