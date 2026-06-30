"""Shared pytest fixtures for inference-kernel tests."""

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cuda",
        help="CUDA device for GPU tests (e.g. cuda, cuda:1). Default: cuda.",
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


def assert_close_common(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """allclose with tight tolerances picked per dtype.

    The default budget for kernels with little compounded floating-point error:
    per-element ops (e.g. silu) and reduction-style kernels whose error is
    softened by rsqrt (e.g. rmsnorm). fp32: tight (default). fp16: loose.
    bf16: looser still (3-bit mantissa less than fp16). Kernels with K-scaling
    accumulation error (e.g. GEMM) should use `assert_close_for_gemm` instead.
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


# Category-specific aliases. Promote to standalone functions if any one
# kernel category ever needs different tolerances.
assert_close_for_rmsnorm = assert_close_common
assert_close_for_silu = assert_close_common
assert_close_for_relu = assert_close_common
assert_close_for_math = assert_close_common  # max/min/sum/softmax reductions


def assert_close_for_attention(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """allclose with looser tolerances for attention's two-matmul + softmax error.

    Attention does QK^T (reduces over head_dim) then PV (reduces over seq_len),
    with a softmax between. Error compounds across both reductions, so tolerances
    track GEMM's rather than the per-element helper's.
    """
    assert_close_for_gemm(actual, expected, dtype)


def assert_close_for_gemm(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """allclose with looser tolerances tuned for GEMM accumulation error.

    GEMM sums K products, so reordering differences scale with K. Compared
    to the reduction-style helper above, fp32 needs ~10× looser, fp16 ~10× looser,
    bf16 ~2× looser to absorb the cuBLAS vs. naive-order accumulation gap.
    """
    if dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        rtol, atol = 2e-2, 2e-2
    else:
        rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
