import torch

from inference_kernel._build.jit import load_kernel
from inference_kernel._common.utils import (
    assert_contiguous,
    assert_is_cuda,
    assert_same_device,
    assert_same_dtype,
    cuda_capability,
)


def _cuda_cflags() -> list[str] | None:
    """nvcc flags for the gemm extension.

    The generational tensor-core kernels need the *arch-accelerated* targets:
    ``wgmma`` (gemm_wgmma) requires ``sm_90a``, ``tcgen05`` (gemm_tcgen05)
    requires ``sm_100a`` — plain ``sm_90``/``sm_100`` cannot even assemble those
    instructions. We add a ``-gencode`` for every accelerated target up to the
    running device so each kernel compiles into the fatbin (its body is
    ``__CUDA_ARCH__``-guarded, so the irrelevant arch slices stay empty).

    Returns None on pre-Hopper GPUs so the loader keeps its default flags (only
    the wmma/simt kernels are reachable there anyway).
    """
    flags = ["-O3", "--use_fast_math"]
    try:
        cap = cuda_capability()
    except Exception:
        return None
    added = False
    if cap >= (9, 0):   # Hopper: enable wgmma
        flags += ["-gencode", "arch=compute_90a,code=sm_90a"]
        added = True
    if cap >= (10, 0):  # Blackwell: enable tcgen05
        flags += ["-gencode", "arch=compute_100a,code=sm_100a"]
        added = True
    return flags if added else None


# Import for its registration side effect; ops are called via torch.ops below.
load_kernel(
    package="inference_kernel.kernels.gemm",
    sources=[
        "naive/gemm.cu",
        "naive/gemm_wgmma.cu",
        "naive/gemm_tcgen05.cu",
        "opt/gemm_opt.cu",
        "binding.cpp",
    ],
    extra_cuda_cflags=_cuda_cflags(),
    # gemm_tcgen05 calls cuTensorMapEncodeTiled (CUDA driver API).
    extra_ldflags=["-lcuda"],
)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM via custom CUDA kernel. Requires CUDA + contiguous inputs."""
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"gemm expects 2D tensors, got a.dim={a.dim()} b.dim={b.dim()}")
    if a.size(1) != b.size(0):
        raise ValueError(f"inner dims mismatch: {a.size(1)} vs {b.size(0)}")
    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    torch.ops.inference_kernel.gemm(out, a, b)
    return out


# --- Per-generation tensor-core kernels (educative) --------------------------
# Each demonstrates one tensor-core programming model. They share a tight scope:
# bf16 in/out and tile-aligned dims. The naive `gemm` above stays the universal
# fallback for everything else (fp32, odd shapes).
#
#   gemm_wgmma    -> Hopper   (sm_90a) warpgroup wgmma.mma_async
#   gemm_tcgen05  -> Blackwell (sm_100a) single-thread tcgen05.mma
#
# wgmma is Hopper-only (Blackwell removed it for tcgen05), so on a B200 it
# compiles but cannot run; gemm_tensorcore() below never selects it there.

# (BLOCK_M, BLOCK_N, BLOCK_K) alignment each kernel requires.
_WGMMA_TILE = (64, 64, 64)
_TCGEN05_TILE = (128, 128, 64)


def _check_2d(a: torch.Tensor, b: torch.Tensor):
    assert_is_cuda(a, b)
    assert_contiguous(a, b)
    device = assert_same_device(a, b)
    dtype = assert_same_dtype(a, b)
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"gemm expects 2D tensors, got a.dim={a.dim()} b.dim={b.dim()}")
    if a.size(1) != b.size(0):
        raise ValueError(f"inner dims mismatch: {a.size(1)} vs {b.size(0)}")
    return device, dtype


def _tile_aligned(a: torch.Tensor, b: torch.Tensor, tile: tuple[int, int, int]) -> bool:
    bm, bn, bk = tile
    M, K, N = a.size(0), a.size(1), b.size(1)
    return M % bm == 0 and N % bn == 0 and K % bk == 0


def gemm_wgmma(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM via Hopper warpgroup ``wgmma`` (sm_90a). bf16, tile-aligned only.

    Runs on Hopper; raises on other GPUs (Blackwell removed wgmma — use
    gemm_tcgen05 there)."""
    device, dtype = _check_2d(a, b)
    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    torch.ops.inference_kernel.gemm_wgmma(out, a, b)
    return out


def gemm_tcgen05(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM via Blackwell 5th-gen ``tcgen05.mma`` (sm_100a). bf16, tile-aligned
    only. Requires an sm_100+ GPU."""
    device, dtype = _check_2d(a, b)
    out = torch.empty((a.size(0), b.size(1)), device=device, dtype=dtype)
    torch.ops.inference_kernel.gemm_tcgen05(out, a, b)
    return out


def gemm_tensorcore(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pick the newest tensor-core kernel this GPU supports for the given shape,
    falling back to the universal naive ``gemm``.

    bf16 + tile-aligned shapes go to tcgen05 on Blackwell / wgmma on Hopper;
    everything else (fp32, odd shapes, older GPUs) uses the naive kernel."""
    assert_is_cuda(a, b)
    cap = cuda_capability(a.device)
    if a.dtype is torch.bfloat16:
        if cap >= (10, 0) and _tile_aligned(a, b, _TCGEN05_TILE):
            return gemm_tcgen05(a, b)
        if cap == (9, 0) and _tile_aligned(a, b, _WGMMA_TILE):
            return gemm_wgmma(a, b)
    return gemm(a, b)