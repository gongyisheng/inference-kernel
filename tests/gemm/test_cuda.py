"""CUDA gemm correctness vs torch reference (also smoke-tests the JIT loader).

Covers the dispatching `gemm` (opt tensor-core kernel, naive fallback),
`gemm_naive` (always the naive WMMA kernel), and `gemm_cutlass` (CUTLASS 2.x).
"""

import pytest
import torch
import torch.nn.functional as F

from ref.gemm import gemm as gemm_ref
from tests.conftest import assert_close_for_gemm


@pytest.mark.cuda
@pytest.mark.parametrize("impl", ["gemm", "gemm_naive", "gemm_cutlass"])
@pytest.mark.parametrize(
    "shape",
    [
        (8, 16, 8),        # smaller than one block tile (partial M/N)
        (32, 64, 32),
        (17, 33, 23),      # K=33 unaligned -> naive fallback
        (64, 128, 32),
        (128, 256, 128),   # exactly one block tile wide
        (130, 64, 130),    # partial trailing block in M and N
        (256, 256, 256),   # multiple full block tiles
    ],
    ids=str,
)
def test_gemm_cuda_matches_ref(
    impl: str, shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    import aot_kernel
    gemm_cuda = getattr(aot_kernel, impl)

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_cuda(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.cuda
def test_gemm_cuda_preserves_shape_and_dtype(dtype: torch.dtype, device: torch.device) -> None:
    from aot_kernel.gemm import gemm as gemm_cuda

    a = torch.randn(3, 5, dtype=dtype, device=device)
    b = torch.randn(5, 7, dtype=dtype, device=device)
    c = gemm_cuda(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype


@pytest.mark.cuda
def test_gemm_cuda_rejects_cpu_tensor() -> None:
    from aot_kernel.gemm import gemm as gemm_cuda

    a = torch.randn(8, 16)
    b = torch.randn(16, 8)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_cuda(a, b)


@pytest.mark.cuda
def test_gemm_cuda_rejects_non_contiguous(device: torch.device) -> None:
    from aot_kernel.gemm import gemm as gemm_cuda

    a = torch.randn(8, 8, device=device).t()
    b = torch.randn(8, 8, device=device)
    assert not a.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        gemm_cuda(a, b)


@pytest.mark.cuda
def test_gemm_cuda_rejects_mismatched_dtype(device: torch.device) -> None:
    from aot_kernel.gemm import gemm as gemm_cuda

    a = torch.randn(8, 8, device=device, dtype=torch.float32)
    b = torch.randn(8, 8, device=device, dtype=torch.float16)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_cuda(a, b)


_ACTIVATIONS = {
    "relu": F.relu,
    "silu": F.silu,
}


@pytest.mark.cuda
@pytest.mark.parametrize("activation", ["relu", "silu"])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 64, 32),      # aligned -> tensor-op path (fp16/bf16)
        (17, 33, 23),      # unaligned K/N -> SIMT fallback
        (128, 256, 128),   # one full block tile
        (130, 64, 130),    # partial trailing blocks
    ],
    ids=str,
)
def test_gemm_cutlass_fused_act_matches_ref(
    activation: str, shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> None:
    from aot_kernel import gemm_cutlass_fused_act

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_cutlass_fused_act(a, b, activation)
    expected = _ACTIVATIONS[activation](gemm_ref(a, b))
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("activation", ["gelu", None])
def test_gemm_cutlass_fused_act_rejects_bad_activation(activation, device: torch.device) -> None:
    from aot_kernel import gemm_cutlass_fused_act

    a = torch.randn(8, 16, device=device, dtype=torch.float16)
    b = torch.randn(16, 8, device=device, dtype=torch.float16)
    with pytest.raises(ValueError):
        gemm_cutlass_fused_act(a, b, activation)
