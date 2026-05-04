"""Inference kernels with torch / triton / cuda backends.

Backends are imported explicitly per category; each backend file holds
every kernel function in that category:

    from inference_kernel.kernels.activation.torch_impl  import silu
    from inference_kernel.kernels.activation.triton_impl import silu
    from inference_kernel.kernels.activation.cuda_impl   import silu
"""

__version__ = "0.0.1"
