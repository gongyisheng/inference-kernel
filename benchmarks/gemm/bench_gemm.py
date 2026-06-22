"""Benchmark gemm across torch / triton / cuda backends.

Run:  uv run python3 benchmarks/gemm/bench_gemm.py --device cuda
"""

import argparse

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._harness import run_bench

KERNEL = "gemm"
SHAPES: list[tuple[int, int, int]] = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# C = A @ B is 2*M*N*K FLOPs total. The harness multiplies FLOPS_PER_ELEMENT
# by prod(shape) = M*K*N, so 2.0 yields the correct total.
FLOPS_PER_ELEMENT = 2.0
_B: torch.Tensor | None = None


def _make_input(shape, dtype, device):
    global _B
    M, K, N = shape
    a = torch.randn(M, K, dtype=dtype, device=device)
    _B = torch.randn(K, N, dtype=dtype, device=device)
    return a


def _bind_b(impl):
    def call(a: torch.Tensor) -> torch.Tensor:
        return impl(a, _B)
    return call


def _backends() -> dict:
    backends: dict = {}

    from inference_kernel.kernels.gemm.ref.torch_impl import gemm as gemm_torch
    # torch_impl is a @ b → cuBLAS; the production-quality baseline.
    backends["torch"] = _bind_b(gemm_torch)

    try:
        from inference_kernel.kernels.gemm.naive.triton_impl import gemm as gemm_triton
        backends["triton"] = _bind_b(gemm_triton)
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")

    try:
        from inference_kernel.kernels.gemm.naive.cuda_impl import gemm as gemm_cuda
        backends["cuda"] = _bind_b(gemm_cuda)
    except ImportError as e:
        print(f"  [skip] cuda import failed: {e}")

    try:
        from inference_kernel.kernels.gemm.opt.cuda_impl import gemm as gemm_cuda_opt
        backends["cuda_opt"] = _bind_b(gemm_cuda_opt)
    except ImportError:
        pass  # no opt kernel yet

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
