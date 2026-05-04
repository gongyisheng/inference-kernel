"""Benchmark silu across torch / triton / cuda backends.

Run:  uv run python -m benchmarks.activation.bench_silu --device cuda:0
"""
from __future__ import annotations

import argparse

import torch
import torch._dynamo

from benchmarks._harness import run_bench

# torch.compile caches per (shape, dtype) signature. Default cache size is 8;
# we sweep 4 shapes x 3 dtypes = 12 combos, which would exceed the limit and
# silently fall back to eager. Bump it so all combos stay compiled.
torch._dynamo.config.cache_size_limit = 64

KERNEL = "silu"
SHAPES: list[tuple[int, ...]] = [
    (1 << 14,),         # 16K
    (1 << 16,),         # 64K
    (1 << 20,),         # 1M
    (4096, 4096),       # 16M
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOPS_PER_ELEMENT = 2.0  # silu = mul + sigmoid; counted as ~2 ops/elem


def _backends() -> dict:
    backends = {}

    from inference_kernel.kernels.activation.torch_impl import silu as silu_torch
    # Eager torch is the test oracle; for benchmarking it's not a fair baseline.
    # Use torch.compile so we measure what an actual user would deploy.
    backends["torch_compile"] = torch.compile(silu_torch, mode="reduce-overhead")

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
    p.add_argument("--device", default="cuda:0")
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
    )


if __name__ == "__main__":
    main()
