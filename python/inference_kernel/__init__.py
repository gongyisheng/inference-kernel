"""Inference kernels with torch / triton / cuda backends.

Each category exposes one file per backend, holding every kernel function in
that category:

    from inference_kernel.kernels.activation.torch_impl  import silu  # torch reference
    from inference_kernel.kernels.activation.triton_impl import silu
    from inference_kernel.kernels.activation.cuda_impl   import silu
"""

__version__ = "0.0.1"
