"""Per-generation tensor-core gemm kernels (wgmma / tcgen05).

Each kernel is gated on the GPU generation that owns its instruction:
  * gemm_tcgen05 -> Blackwell (sm_100+)
  * gemm_wgmma   -> Hopper only (sm_90); Blackwell removed wgmma.
Tests skip cleanly when the running GPU can't execute the kernel.
"""

import pytest
import torch
from inference_kernel.kernels.gemm.ref.eager_impl import gemm as gemm_ref

from tests.conftest import assert_close_for_gemm


def _cap() -> tuple[int, int]:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


# tcgen05 tile is 128x128x64; wgmma tile is 64x64x64.
TCGEN05_SHAPES = [(128, 64, 128), (256, 256, 256), (512, 128, 256), (1024, 1024, 1024)]
WGMMA_SHAPES = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 256, 128)]


@pytest.mark.cuda
@pytest.mark.skipif(_cap() < (10, 0), reason="tcgen05 requires Blackwell sm_100+")
@pytest.mark.parametrize("shape", TCGEN05_SHAPES, ids=str)
def test_tcgen05_matches_ref(shape: tuple[int, int, int], device: torch.device) -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm_tcgen05

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    got = gemm_tcgen05(a, b)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    assert_close_for_gemm(got, gemm_ref(a, b), torch.bfloat16)


@pytest.mark.cuda
@pytest.mark.skipif(_cap() != (9, 0), reason="wgmma runs on Hopper sm_90 only")
@pytest.mark.parametrize("shape", WGMMA_SHAPES, ids=str)
def test_wgmma_matches_ref(shape: tuple[int, int, int], device: torch.device) -> None:
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm_wgmma

    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    got = gemm_wgmma(a, b)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    assert_close_for_gemm(got, gemm_ref(a, b), torch.bfloat16)


@pytest.mark.cuda
def test_wgmma_errors_off_hopper(device: torch.device) -> None:
    """On non-Hopper GPUs wgmma must fail with a clear message, not crash."""
    if _cap() == (9, 0):
        pytest.skip("this GPU is Hopper; wgmma is expected to run")
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm_wgmma

    a = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    b = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="Hopper"):
        gemm_wgmma(a, b)


@pytest.mark.cuda
def test_tensorcore_auto_matches_ref(device: torch.device) -> None:
    """gemm_tensorcore auto-routes by capability+shape and falls back cleanly."""
    from inference_kernel.kernels.gemm.naive.cuda_impl import gemm_tensorcore

    torch.manual_seed(0)
    # bf16 aligned -> tensor-core path on Hopper/Blackwell, naive elsewhere.
    a = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
    b = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
    assert_close_for_gemm(gemm_tensorcore(a, b), gemm_ref(a, b), torch.bfloat16)

    # Odd shape -> must fall back to the universal naive kernel (no tile align).
    a2 = torch.randn(17, 33, dtype=torch.bfloat16, device=device)
    b2 = torch.randn(33, 23, dtype=torch.bfloat16, device=device)
    assert_close_for_gemm(gemm_tensorcore(a2, b2), gemm_ref(a2, b2), torch.bfloat16)
