"""One-shot GEMM runner for ncu profiling.

Builds a single (backend, M, N, K, dtype) input and runs the kernel once
after warmup. Intended to be wrapped by ncu (see profile_gemm.sh).
"""

import argparse

import torch

import aot_kernel as cuda_impl
import jit_kernel as triton_impl
import ref as torch_impl

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_fn(backend: str, triton_impl_kind: str):
    if backend == "cuda":
        return cuda_impl.gemm
    if backend == "torch":
        return torch_impl.gemm
    if backend == "triton":
        return lambda a, b: triton_impl.gemm(a, b, kernel_implementation=triton_impl_kind)
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cuda", "triton", "torch"], required=True)
    parser.add_argument("--triton-impl", choices=["thread", "tile"], default="tile")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    a = torch.randn((args.m, args.k), dtype=dtype, device=device)
    b = torch.randn((args.k, args.n), dtype=dtype, device=device)

    fn = _make_fn(args.backend, args.triton_impl)

    for _ in range(args.warmup):
        fn(a, b)
    torch.cuda.synchronize(device)

    for _ in range(args.iters):
        fn(a, b)
    torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
