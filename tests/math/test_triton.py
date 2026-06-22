"""Triton max/min/sum/softmax correctness vs torch reference."""

import pytest
import torch
from inference_kernel.kernels.math import torch_impl, triton_impl

from tests.conftest import assert_close_for_math

OPS = ["max", "min", "sum"]


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


# softmax preserves shape (output == input shape), so it can't share the OPS
# reduction tests above; it gets its own block.


@pytest.mark.triton
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1024,), -1),
        ((32, 1024), -1),
        ((4, 16, 1024), -1),
        ((1023,), -1),
        ((8, 5000), -1),       # reduce_size > BLOCK_SIZE: online loop runs >1 bite
        ((2, 3, 4), 1),        # non-last axis: exercises movedim round-trip
        ((7, 3, 9), 0),
    ],
    ids=str,
)
def test_softmax_triton_matches_ref(
    shape: tuple[int, ...], dim: int, dtype: torch.dtype, device: torch.device
) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=device)
    got = triton_impl.softmax(x, dim)
    expected = torch_impl.softmax(x, dim)
    assert got.shape == expected.shape
    assert_close_for_math(got, expected, dtype)


@pytest.mark.triton
def test_softmax_triton_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    x = torch.randn(4, 32, dtype=dtype, device=device)
    y = triton_impl.softmax(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


@pytest.mark.triton
def test_softmax_triton_non_contiguous_matches_ref(device: torch.device) -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 8, device=device).t()
    assert not x.is_contiguous()
    got = triton_impl.softmax(x, dim=-1)
    expected = torch_impl.softmax(x, dim=-1)
    torch.testing.assert_close(got, expected)
