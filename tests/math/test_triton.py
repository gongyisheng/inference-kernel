"""Triton max/min correctness vs torch reference."""

import pytest
import torch
from inference_kernel.kernels.math import torch_impl, triton_impl

from tests.conftest import assert_close_for_math

OPS = ["max", "min"]


@pytest.mark.triton
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1024,), -1),
        ((32, 1024), -1),
        ((4, 16, 1024), -1),
        ((1023,), -1),         # reduce_size not a multiple of BLOCK_SIZE
        ((8, 5000), -1),       # reduce_size > BLOCK_SIZE: loop runs >1 bite
        ((2, 3, 4), 1),        # reduce a non-last axis
        ((7, 3, 9), 0),        # reduce the first axis
    ],
    ids=str,
)
def test_triton_matches_ref(
    op: str, shape: tuple[int, ...], dim: int, dtype: torch.dtype, device: torch.device
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = getattr(triton_impl, op)(x, dim)
    expected = getattr(torch_impl, op)(x, dim)
    assert got.shape == expected.shape
    assert_close_for_math(got, expected, dtype)


@pytest.mark.triton
@pytest.mark.parametrize("op", OPS)
def test_triton_preserves_dtype(op: str, dtype: torch.dtype, device: torch.device) -> None:
    x = torch.randn(4, 32, dtype=dtype, device=device)
    y = getattr(triton_impl, op)(x)
    assert y.dtype == x.dtype
    assert y.shape == (4,)


@pytest.mark.triton
@pytest.mark.parametrize("op", OPS)
def test_triton_non_contiguous_matches_ref(op: str, device: torch.device) -> None:
    """Non-contiguous input is supported: wrapper makes it contiguous internally."""
    torch.manual_seed(0)
    x = torch.randn(16, 8, device=device).t()  # transposed -> non-contiguous
    assert not x.is_contiguous()
    got = getattr(triton_impl, op)(x, dim=-1)
    expected = getattr(torch_impl, op)(x, dim=-1)
    torch.testing.assert_close(got, expected)
