"""CUDA kernels, ahead-of-time compiled into the `_C` extension.

Importing this package loads `_C` (the prebuilt `.so`), registering ops under
the `aot_kernel` torch namespace. Same operation names and signatures as `ref`
and `jit_kernel` (no `attention`/`math` until those kernels exist).
"""

from .activation import relu, silu
from .gemm import gemm, gemm_cutlass, gemm_cutlass_fused_act, gemm_naive
from .norm import rmsnorm

__version__ = "0.0.1"

__all__ = [
    "gemm",
    "gemm_cutlass",
    "gemm_cutlass_fused_act",
    "gemm_naive",
    "relu",
    "rmsnorm",
    "silu",
]
