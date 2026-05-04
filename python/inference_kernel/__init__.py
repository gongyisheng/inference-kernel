"""Inference kernels with torch / triton / cuda backends.

Backends are imported explicitly per kernel:

    from inference_kernel.kernels.activation.silu.torch_impl  import silu
    from inference_kernel.kernels.activation.silu.triton_impl import silu
    from inference_kernel.kernels.activation.silu.cuda_impl   import silu
"""

__version__ = "0.0.1"
