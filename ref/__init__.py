"""Torch/eager reference implementations — the correctness oracle.

Same operation names and signatures as `jit_kernel` and `aot_kernel`, so the
three are drop-in swappable.
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
