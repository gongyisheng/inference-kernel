"""Shared pytest fixtures for inference_kernel tests."""
from __future__ import annotations

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cuda:0",
        help="CUDA device for GPU tests (e.g. cuda:0, cuda:1). Default: cuda:0.",
    )


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> torch.device:
    """The CUDA device tests should target (overridable via --device)."""
    return torch.device(request.config.getoption("--device"))


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Iterate over the supported floating dtypes."""
    return request.param


@pytest.fixture(autouse=True)
def _skip_if_marker(request: pytest.FixtureRequest) -> None:
    """Auto-apply skip logic for cuda/triton markers."""
    if request.node.get_closest_marker("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if request.node.get_closest_marker("triton"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available (required for triton)")
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not installed")


def assert_close_for_dtype(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """allclose with tolerances picked per dtype.

    fp32: tight (default). fp16: loose. bf16: looser still (3-bit mantissa less than fp16).
    """
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-6
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-5, 1e-6
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
