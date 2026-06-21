"""Benchmark gemm across torch / triton / cuda backends.

Run:  uv run python3 benchmarks/gemm/bench_gemm.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

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
# Effective global-memory traffic of the *naive WMMA* kernel, which stages
# nothing in shared memory: each 16x16x16 MMA reloads two 16x16 fragments and
# there are (M/16)(N/16)(K/16) of them -> M*N*K/8 elements moved. The harness
# computes io_factor * M*N*K * element_size, so 1/8 reproduces that.
# Physically meaningful ONLY for the `cuda` backend; cuBLAS/Triton reuse
# operands via shared memory, so their reported GB/s is a model artifact.
IO_FACTOR = 1.0 / 8.0
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

    from inference_kernel.kernels.gemm.ref.eager_impl import gemm as gemm_eager
    from inference_kernel.kernels.gemm.ref.torch_impl import gemm as gemm_torch
    # Eager is the test oracle (broadcast-and-sum); materializes [M, N, K] so
    # it OOMs on large shapes — harness will skip those rows.
    backends["eager"] = _bind_b(gemm_eager)
    # torch_impl is just a @ b → cuBLAS; the production-quality baseline.
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

    # Per-generation tensor-core kernels (bf16 + tile-aligned only; the harness
    # skips the rows they don't support). wgmma is Hopper-only, so on Blackwell
    # every cuda_wgmma row skips with a clear "requires sm_90" message.
    try:
        from inference_kernel.kernels.gemm.naive.cuda_impl import (
            gemm_tcgen05,
            gemm_wgmma,
        )
        backends["cuda_wgmma"] = _bind_b(gemm_wgmma)
        backends["cuda_tcgen05"] = _bind_b(gemm_tcgen05)
    except ImportError:
        pass

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
        io_factor=IO_FACTOR,
        x_axis=lambda s: s[0],
        x_label="size (M=K=N)",
    )


if __name__ == "__main__":
    main()
