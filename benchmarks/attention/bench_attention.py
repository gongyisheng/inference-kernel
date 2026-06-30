"""Benchmark attention across torch (SDPA) / triton backends.

Run:  uv run python3 benchmarks/attention/bench_attention.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._harness import run_bench

KERNEL = "attention"
# Shape is (B, H, M, N, D) with M == N. Folding N into the shape makes numel =
# B*H*M*N*D, so the FLOP count is a constant multiple of numel (see below).
SHAPES: list[tuple[int, int, int, int, int]] = [
    (4, 8, 512, 512, 64),
    (4, 8, 1024, 1024, 64),
    (4, 8, 2048, 2048, 64),
    (2, 8, 4096, 4096, 64),
    (4, 8, 1024, 1024, 128),
    (2, 8, 2048, 2048, 128),
]
DTYPES = [torch.float16, torch.bfloat16]
# QK^T (2*B*H*M*N*D) + PV (2*B*H*M*N*D) = 4 * numel FLOPs (softmax ignored).
FLOPS_PER_ELEMENT = 4.0
_K: torch.Tensor | None = None
_V: torch.Tensor | None = None


def _make_input(shape, dtype, device):
    global _K, _V
    B, H, M, N, D = shape
    q = torch.randn(B, H, M, D, dtype=dtype, device=device)
    _K = torch.randn(B, H, N, D, dtype=dtype, device=device)
    _V = torch.randn(B, H, N, D, dtype=dtype, device=device)
    return q


def _bind_kv(impl):
    def call(q: torch.Tensor) -> torch.Tensor:
        return impl(q, _K, _V)
    return call


def _backends() -> dict:
    backends: dict = {}

    from ref.attention import attention as attn_torch
    # torch_impl is F.scaled_dot_product_attention → the production baseline.
    backends["torch"] = _bind_kv(attn_torch)

    try:
        from jit_kernel.attention import attention as attn_triton
        backends["triton"] = _bind_kv(attn_triton)
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
        x_axis=lambda s: s[2],
        x_label="seq_len (M=N)",
    )


if __name__ == "__main__":
    main()
