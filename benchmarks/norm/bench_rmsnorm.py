"""Benchmark rmsnorm across torch / triton backends.

Run:  uv run python -m benchmarks.norm.bench_rmsnorm --device cuda:0

The harness times callables of the form fn(x), but rmsnorm needs
(x, weight). We cache one weight tensor per (last_dim, dtype, device)
and wrap each backend with a closure that injects it. The cache lookup
is a dict access (~ns) and runs outside the timed kernel call, so it
does not pollute the measurement.
"""
from __future__ import annotations

import argparse

import torch
import torch._dynamo

from benchmarks._harness import run_bench

torch._dynamo.config.cache_size_limit = 64

KERNEL = "rmsnorm"
SHAPES: list[tuple[int, ...]] = [
    (1024, 4096),         # small prefill batch, llama-3 hidden
    (4096, 4096),         # larger 2D
    (16, 2048, 4096),     # typical batched inference activation
    (8, 8, 8, 16384),     # high-rank, large hidden — exercises BLOCK_SIZE=16384
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# RMSNorm per element: x*x (1) + reduction add (1) + rstd*x (1) + *w (1) ≈ 4-5 FLOPs.
# Memory-bound in practice; TFLOPS is informational.
FLOPS_PER_ELEMENT = 5.0

_WEIGHT_CACHE: dict[tuple, torch.Tensor] = {}


def _weight_for(x: torch.Tensor) -> torch.Tensor:
    key = (x.shape[-1], x.dtype, x.device)
    w = _WEIGHT_CACHE.get(key)
    if w is None:
        w = torch.randn(x.shape[-1], dtype=x.dtype, device=x.device)
        _WEIGHT_CACHE[key] = w
    return w


def _bind_weight(impl):
    def call(x: torch.Tensor) -> torch.Tensor:
        return impl(x, _weight_for(x))
    return call


def _backends() -> dict:
    backends: dict = {}

    from inference_kernel.kernels.norm.eager_impl import rmsnorm as rmsnorm_eager
    from inference_kernel.kernels.norm.torch_impl import rmsnorm as rmsnorm_torch
    backends["eager"] = _bind_weight(rmsnorm_eager)
    backends["torch"] = _bind_weight(
        torch.compile(rmsnorm_torch, mode="reduce-overhead")
    )

    try:
        from inference_kernel.kernels.norm.triton_impl import rmsnorm as rmsnorm_triton
        backends["triton"] = _bind_weight(rmsnorm_triton)
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")

    try:
        from inference_kernel.kernels.norm.cuda_impl import rmsnorm as rmsnorm_cuda
        backends["cuda"] = _bind_weight(rmsnorm_cuda)
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
