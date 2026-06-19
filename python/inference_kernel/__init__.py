"""Inference kernels with torch / triton / cuda backends.

Each category exposes multiple backend files holding every kernel function
in that category:

    from inference_kernel.kernels.activation.reference.eager_impl  import silu  # slow oracle
    from inference_kernel.kernels.activation.reference.torch_impl  import silu  # F.silu
    from inference_kernel.kernels.activation.naive.triton_impl import silu
    from inference_kernel.kernels.activation.naive.cuda_impl   import silu
"""

__version__ = "0.0.1"
