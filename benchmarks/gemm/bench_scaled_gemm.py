"""Benchmark scaled_gemm across quant dtype (fp8, int8) x scale granularity.

Run:  uv run python3 benchmarks/gemm/bench_scaled_gemm.py --device cuda

Backends are named "<framework>_<scale_mode>" so a single run_bench sweep covers
triton_tensor/triton_row and torch_tensor/torch_row. The input dtype (fp8/int8)
is the harness dtype axis.

torch baselines differ by dtype: fp8 -> torch._scaled_mm (fused);
int8 -> torch._int_mm (int32) + a manual dequant multiply, since torch has no
fused int8-scaled matmul. Both torch paths need the second operand column-major,
while our triton kernel wants it row-major contiguous; both layouts are built in
make_input (values are irrelevant to timing).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._harness import run_bench

KERNEL = "scaled_gemm"
FP8 = torch.float8_e4m3fn
SHAPES: list[tuple[int, int, int]] = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]
DTYPES = [FP8, torch.int8]
FLOPS_PER_ELEMENT = 2.0  # C = A @ B is 2*M*N*K FLOPs; harness scales by M*K*N

_B_ROW: torch.Tensor | None = None   # (K, N) row-major, for triton
_B_COL: torch.Tensor | None = None   # (K, N) column-major, for torch
_SA_T: torch.Tensor | None = None    # tensorwise scalars
_SB_T: torch.Tensor | None = None
_SA_R: torch.Tensor | None = None    # rowwise (M,) / (N,)
_SB_R: torch.Tensor | None = None


def _quant(shape, dtype, device):
    M, K, N = shape
    if dtype == FP8:
        a = torch.randn(M, K, device=device).to(FP8)
        b_row = torch.randn(K, N, device=device).to(FP8)
        b_col = torch.randn(N, K, device=device).to(FP8).t()
    else:  # int8
        a = torch.randint(-8, 8, (M, K), device=device, dtype=torch.int8)
        b_row = torch.randint(-8, 8, (K, N), device=device, dtype=torch.int8)
        b_col = torch.randint(-8, 8, (N, K), device=device, dtype=torch.int8).t()
    return a, b_row, b_col


def _make_input(shape, dtype, device):
    global _B_ROW, _B_COL, _SA_T, _SB_T, _SA_R, _SB_R
    M, _, N = shape
    a, _B_ROW, _B_COL = _quant(shape, dtype, device)
    _SA_T = torch.tensor([0.05], device=device)
    _SB_T = torch.tensor([0.03], device=device)
    _SA_R = torch.rand(M, device=device) * 0.1 + 0.01
    _SB_R = torch.rand(N, device=device) * 0.1 + 0.01
    return a


def _int8_scaled(a, b_col, sa, sb):
    acc = torch._int_mm(a, b_col)  # int32 (M, N)
    return (acc.float() * sa * sb).to(torch.bfloat16)


_int8_scaled_compiled = torch.compile(_int8_scaled)


def _torch_baseline(a, scale_mode, compiled=False):
    if scale_mode == "tensor":
        sa, sb = _SA_T, _SB_T
    else:
        sa, sb = _SA_R.reshape(-1, 1), _SB_R.reshape(1, -1)
    if a.dtype == FP8:
        if compiled:
            raise ValueError("torch.compile int8 baseline skips fp8")
        return torch._scaled_mm(a, _B_COL, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    # int8: no fused scaled path -> int_mm then dequant.
    return (_int8_scaled_compiled if compiled else _int8_scaled)(a, _B_COL, sa, sb)


def _backends() -> dict:
    backends: dict = {
        "torch_tensor": lambda a: _torch_baseline(a, "tensor"),
        "torch_row": lambda a: _torch_baseline(a, "row"),
        "torch_compile_tensor": lambda a: _torch_baseline(a, "tensor", compiled=True),
        "torch_compile_row": lambda a: _torch_baseline(a, "row", compiled=True),
    }
    try:
        from jit_kernel.gemm import scaled_gemm

        backends["triton_naive_tensor"] = lambda a: scaled_gemm(
            a, _B_ROW, _SA_T, _SB_T, out_dtype=torch.bfloat16, optimized=False
        )
        backends["triton_opt_tensor"] = lambda a: scaled_gemm(
            a, _B_ROW, _SA_T, _SB_T, out_dtype=torch.bfloat16, optimized=True
        )
        backends["triton_naive_row"] = lambda a: scaled_gemm(
            a, _B_ROW, _SA_R, _SB_R, out_dtype=torch.bfloat16, optimized=False
        )
        backends["triton_opt_row"] = lambda a: scaled_gemm(
            a, _B_ROW, _SA_R, _SB_R, out_dtype=torch.bfloat16, optimized=True
        )
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")
    return backends


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    run_bench(
        kernel=KERNEL,
        backends=_backends(),
        make_input=_make_input,
        shapes=SHAPES,
        dtypes=DTYPES,
        device=device,
        flops_per_element=FLOPS_PER_ELEMENT,
        x_axis=lambda s: s[0],
        x_label="size (M=K=N)",
    )


if __name__ == "__main__":
    main()
