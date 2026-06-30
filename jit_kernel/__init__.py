"""Triton kernels (compiled at runtime / first call).

Same operation names and signatures as `ref` and `aot_kernel`.
"""

from .activation import relu, silu
from .attention import attention
from .gemm import gemm
from .math import avg, max, min, softmax, sum
from .norm import rmsnorm

__version__ = "0.0.1"

__all__ = [
    "attention",
    "avg",
    "gemm",
    "max",
    "min",
    "relu",
    "rmsnorm",
    "silu",
    "softmax",
    "sum",
]
