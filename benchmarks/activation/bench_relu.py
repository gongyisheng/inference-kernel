"""Benchmark relu across torch / triton / cuda backends.

Run:  uv run python -m benchmarks.activation.bench_relu --device cuda:0
"""

import argparse

import torch
import torch._dynamo

from benchmarks._harness import run_bench

# torch.compile caches per (shape, dtype) signature. Default cache size is 8;
# we sweep 4 shapes x 3 dtypes = 12 combos, which would exceed the limit and
# silently fall back to eager. Bump it so all combos stay compiled.
torch._dynamo.config.cache_size_limit = 64

KERNEL = "relu"
SHAPES: list[tuple[int, ...]] = [
    (1 << 14,),         # 16K
    (1 << 16,),         # 64K
    (1 << 20,),         # 1M
    (4096, 4096),       # 16M
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
FLOPS_PER_ELEMENT = 1.0  # relu


def _backends() -> dict:
    backends = {}

    from inference_kernel.kernels.activation.reference.eager_impl import relu as relu_eager
    from inference_kernel.kernels.activation.reference.torch_impl import relu as relu_torch

    backends["torch"] = relu_torch
    backends["torch_compile"] = torch.compile(relu_eager, mode="reduce-overhead")

    try:
        from inference_kernel.kernels.activation.naive.triton_impl import relu as relu_triton
        backends["triton"] = relu_triton
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")

    try:
        from inference_kernel.kernels.activation.naive.cuda_impl import relu as relu_cuda
        backends["cuda"] = relu_cuda
    except ImportError as e:
        print(f"  [skip] cuda import failed: {e}")

    try:
        from inference_kernel.kernels.activation.opt.cuda_impl import relu as relu_cuda_opt
        backends["cuda_opt"] = relu_cuda_opt
    except ImportError:
        pass  # no opt kernel yet

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
