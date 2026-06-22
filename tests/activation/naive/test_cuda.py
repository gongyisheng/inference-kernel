"""CUDA (naive tier) silu/relu correctness vs torch reference (also smoke-tests the JIT loader)."""

import pytest
import torch
from inference_kernel.kernels.activation.ref.torch_impl import relu as relu_ref
from inference_kernel.kernels.activation.ref.torch_impl import silu as silu_ref

from tests.conftest import assert_close_for_relu, assert_close_for_silu


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_silu_cuda_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    from inference_kernel.kernels.activation.naive.cuda_impl import silu as silu_cuda

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = silu_cuda(x)
    expected = silu_ref(x)
    assert_close_for_silu(got, expected, dtype)


@pytest.mark.cuda
def test_silu_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.activation.naive.cuda_impl import silu as silu_cuda

    x = torch.randn(8)  # CPU
    with pytest.raises((ValueError, RuntimeError)):
        silu_cuda(x)


@pytest.mark.cuda
def test_silu_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from inference_kernel.kernels.activation.naive.cuda_impl import silu as silu_cuda

    x = torch.randn(8, 8, device=device).t()
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        silu_cuda(x)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(1024,), (32, 1024), (4, 16, 1024), (1023,)], ids=str)
def test_relu_cuda_matches_ref(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    from inference_kernel.kernels.activation.naive.cuda_impl import relu as relu_cuda

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = relu_cuda(x)
    expected = relu_ref(x)
    assert_close_for_relu(got, expected, dtype)


@pytest.mark.cuda
def test_relu_cuda_rejects_cpu_tensor() -> None:
    from inference_kernel.kernels.activation.naive.cuda_impl import relu as relu_cuda

    x = torch.randn(8)  # CPU
    with pytest.raises((ValueError, RuntimeError)):
        relu_cuda(x)


@pytest.mark.cuda
def test_relu_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from inference_kernel.kernels.activation.naive.cuda_impl import relu as relu_cuda

    x = torch.randn(8, 8, device=device).t()
    assert not x.is_contiguous()
    with pytest.raises((ValueError, RuntimeError)):
        relu_cuda(x)
