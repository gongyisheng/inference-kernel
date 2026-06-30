"""Benchmark relu across torch / triton / cuda backends.

Run:  uv run python benchmarks/activation/bench_relu.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

# Runnable directly (python benchmarks/<cat>/bench_*.py), not only via -m:
# put the repo root on sys.path so `benchmarks._harness` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._harness import run_bench

KERNEL = "relu"
SHAPES: list[tuple[int, ...]] = [
    (1, 16384),       # decode, batch=1    (16K)
    (32, 16384),      # decode, batch=32   (524K)
    (128, 16384),     # decode, batch=128  (2.1M)
    (2048, 16384),    # prefill, 2K tokens (34M)
    (4096, 16384),    # prefill, 4K tokens (67M)
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOPS_PER_ELEMENT = 1.0
IO_PER_ELEMENT = 2.0


def _backends() -> dict:
    backends = {}

    from ref.activation import relu as relu_torch

    backends["torch"] = relu_torch

    try:
        from jit_kernel.activation import relu as relu_triton
        backends["triton"] = relu_triton
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")

    try:
        from aot_kernel.activation import relu as relu_cuda
        backends["cuda"] = relu_cuda
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
