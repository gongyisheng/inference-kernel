"""Benchmark rmsnorm across torch / triton backends.

Run:  uv run python -m benchmarks.norm.bench_rmsnorm --device cuda:0
"""

import argparse

import torch
import torch._dynamo

from benchmarks._harness import run_bench

torch._dynamo.config.cache_size_limit = 64

KERNEL = "rmsnorm"
SHAPES: list[tuple[int, ...]] = [
    (1024, 8),
    (1024, 16),
    (1024, 32),
    (1024, 64),
    (1024, 128),
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (1024, 2048),
    (1024, 4096),
    (1024, 8192),
    (1024, 16384),
    (1024, 32768),
    (1024, 65536),
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

    from inference_kernel.kernels.norm.reference.eager_impl import rmsnorm as rmsnorm_eager
    from inference_kernel.kernels.norm.reference.torch_impl import rmsnorm as rmsnorm_torch
    backends["eager"] = _bind_weight(rmsnorm_eager)
    backends["torch"] = _bind_weight(
        torch.compile(rmsnorm_torch, mode="reduce-overhead")
    )

    try:
        from inference_kernel.kernels.norm.naive.triton_impl import rmsnorm as rmsnorm_triton
        backends["triton"] = _bind_weight(rmsnorm_triton)
    except ImportError as e:
        print(f"  [skip] triton import failed: {e}")

    try:
        from inference_kernel.kernels.norm.naive.cuda_impl import rmsnorm as rmsnorm_cuda
        backends["cuda"] = _bind_weight(rmsnorm_cuda)
    except ImportError as e:
        print(f"  [skip] cuda import failed: {e}")

    try:
        from inference_kernel.kernels.norm.opt.cuda_impl import rmsnorm as rmsnorm_cuda_opt
        backends["cuda_opt"] = rmsnorm_cuda_opt
    except ImportError:
        pass  # no opt kernel yet

    try:
        from flashinfer.norm import rmsnorm as rmsnorm_fi

        # flashinfer.rmsnorm only accepts 2D (N, D) or 3D (B, H, D); flatten higher ranks.
        def rmsnorm_flashinfer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            if x.dim() <= 3:
                return rmsnorm_fi(x, w)
            return rmsnorm_fi(x.reshape(-1, x.shape[-1]), w).reshape(x.shape)

        backends["flashinfer"] = _bind_weight(rmsnorm_flashinfer)
    except ImportError as e:
        print(f"  [skip] flashinfer import failed: {e}")

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
        x_axis=lambda s: s[-1],
        x_label="hidden_dim",
    )


if __name__ == "__main__":
    main()
