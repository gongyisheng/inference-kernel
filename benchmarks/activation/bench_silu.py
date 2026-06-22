"""Benchmark silu across torch / triton / cuda backends.

Run:  uv run python3 benchmarks/activation/bench_silu.py --device cuda
"""

import argparse

import torch

import sys
from pathlib import Path

# Runnable directly (python benchmarks/<cat>/bench_*.py), not only via -m:
# put the repo root on sys.path so `benchmarks._harness` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._harness import run_bench

KERNEL = "silu"
SHAPES: list[tuple[int, ...]] = [
    (1, 16384),       # decode, batch=1    (16K)
    (32, 16384),      # decode, batch=32   (524K)
    (128, 16384),     # decode, batch=128  (2.1M)
    (2048, 16384),    # prefill, 2K tokens (34M)
    (4096, 16384),    # prefill, 4K tokens (67M)
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOPS_PER_ELEMENT = 2.0  # silu = mul + sigmoid; counted as ~2 ops/elem
IO_PER_ELEMENT = 2.0  # one read + one write


def _backends() -> dict:
    backends = {}

    from inference_kernel.kernels.activation.torch_impl import silu as silu_torch
    backends["torch"] = silu_torch

    try:
        from inference_kernel.kernels.activation.triton_impl import silu as silu_triton
        backends["triton"] = silu_triton
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")

    try:
        from inference_kernel.kernels.activation.cuda_impl import silu as silu_cuda
        backends["cuda"] = silu_cuda
    except ImportError as e:
        print(f"  [skip] cuda import failed: {e}")

    return backends


def _make_input(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)


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
        io_factor=IO_PER_ELEMENT,
    )


if __name__ == "__main__":
    main()
