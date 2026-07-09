"""Triton gemm correctness vs torch reference."""

import pytest
import torch

from jit_kernel.gemm import gemm as gemm_triton
from jit_kernel.gemm import scaled_gemm
from ref.gemm import gemm as gemm_ref
from tests.conftest import assert_close_for_gemm

FP8 = torch.float8_e4m3fn


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


@pytest.mark.triton
@pytest.mark.parametrize(
    "shape",
    [(8, 16, 8), (32, 64, 32), (17, 33, 23), (64, 128, 32), (128, 256, 128)],
    ids=str,
)
def test_gemm_triton_matches_ref(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    M, K, N = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    got = gemm_triton(a, b)
    expected = gemm_ref(a, b)
    assert_close_for_gemm(got, expected, dtype)


@pytest.mark.triton
def test_gemm_triton_preserves_shape_and_dtype(
    dtype: torch.dtype, device: torch.device
) -> None:
    a = torch.randn(3, 5, dtype=dtype, device=device)
    b = torch.randn(5, 7, dtype=dtype, device=device)
    c = gemm_triton(a, b)
    assert c.shape == (3, 7)
    assert c.dtype == a.dtype


@pytest.mark.triton
def test_gemm_triton_rejects_cpu_tensor() -> None:
    a = torch.randn(8, 16)
    b = torch.randn(16, 8)
    with pytest.raises((ValueError, RuntimeError)):
        gemm_triton(a, b)


@pytest.mark.triton
def test_gemm_triton_non_contiguous_raises(device: torch.device) -> None:
    """Triton backend requires contiguous inputs; non-contiguous must raise."""
    a = torch.randn(8, 8, device=device).t()
    b = torch.randn(8, 8, device=device)
    assert not a.is_contiguous()
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        gemm_triton(a, b)


# --- scaled_gemm (quantized fp8 / int8 with dequant scales) --------------------
#
# Inputs are fp8/int8; both the kernel and the reference dequantize the *same*
# quantized values (scale * q), so the comparison isolates the matmul + epilogue
# scaling rather than quantization error. Output is bf16.

_needs_fp8 = pytest.mark.skipif(
    not _fp8_supported(), reason="fp8 needs compute capability >= 8.9"
)


def _make_quant_inputs(qdtype, M, K, N, device):
    torch.manual_seed(0)
    if qdtype == FP8:
        a = torch.randn(M, K, device=device).to(FP8)
        b = torch.randn(K, N, device=device).to(FP8)
    else:  # int8
        a = torch.randint(-8, 8, (M, K), device=device, dtype=torch.int8)
        b = torch.randint(-8, 8, (K, N), device=device, dtype=torch.int8)
    return a, b


def _make_scales(scale_mode, M, N, device):
    if scale_mode == "tensor":
        return torch.tensor([0.05], device=device), torch.tensor([0.03], device=device)
    torch.manual_seed(1)
    scale_a = torch.rand(M, device=device) * 0.1 + 0.01
    scale_b = torch.rand(N, device=device) * 0.1 + 0.01
    return scale_a, scale_b


def _scaled_reference(a, b, scale_a, scale_b, out_dtype):
    a_deq = a.float() * scale_a.float().reshape(-1, 1)   # per-row / scalar broadcast
    b_deq = b.float() * scale_b.float().reshape(1, -1)   # per-col / scalar broadcast
    return (a_deq @ b_deq).to(out_dtype)


@pytest.mark.triton
@pytest.mark.parametrize("optimized", [False, True], ids=["naive", "optimized"])
@pytest.mark.parametrize("qdtype", [pytest.param(FP8, marks=_needs_fp8), torch.int8], ids=["fp8", "int8"])
@pytest.mark.parametrize("scale_mode", ["tensor", "row"])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),
        (128, 256, 128),
        (130, 64, 130),      # partial trailing block in M and N
        (256, 256, 256),
        (129, 257, 193),     # all three axes unaligned
    ],
    ids=str,
)
def test_scaled_gemm_matches_reference(
    optimized: bool,
    qdtype: torch.dtype,
    scale_mode: str,
    shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    # cuBLAS fp32 matmul would otherwise round inputs to TF32; keep the reference exact.
    torch.backends.cuda.matmul.allow_tf32 = False
    M, K, N = shape
    a, b = _make_quant_inputs(qdtype, M, K, N, device)
    scale_a, scale_b = _make_scales(scale_mode, M, N, device)

    out_dtype = torch.bfloat16
    got = scaled_gemm(a, b, scale_a, scale_b, out_dtype=out_dtype, optimized=optimized)
    expected = _scaled_reference(a, b, scale_a, scale_b, out_dtype)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.triton
@pytest.mark.parametrize("qdtype", [pytest.param(FP8, marks=_needs_fp8), torch.int8], ids=["fp8", "int8"])
def test_scaled_gemm_preserves_shape_and_dtype(qdtype: torch.dtype, device: torch.device) -> None:
    a, b = _make_quant_inputs(qdtype, 64, 128, 96, device)
    scale_a = torch.tensor([0.05], device=device)
    scale_b = torch.tensor([0.03], device=device)
    c = scaled_gemm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
    assert c.shape == (64, 96)
    assert c.dtype == torch.bfloat16


@pytest.mark.triton
def test_scaled_gemm_rejects_bad_scale_shape(device: torch.device) -> None:
    a, b = _make_quant_inputs(torch.int8, 64, 128, 96, device)
    scale_a = torch.rand(63, device=device)   # wrong length (M=64)
    scale_b = torch.rand(96, device=device)
    with pytest.raises((ValueError, AssertionError)):
        scaled_gemm(a, b, scale_a, scale_b)
