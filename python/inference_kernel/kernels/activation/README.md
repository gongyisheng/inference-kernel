# Activation kernels

Element-wise activations used in transformer FFN blocks. Backends are
organized into tier subfolders: `reference/` (`eager_impl.py`, `torch_impl.py`),
`naive/` (`triton_impl.py`, `cuda_impl.py`), and `opt/` (production-grade
kernels, empty for now). Each tier file exposes one function per activation.
CUDA sources live in `csrc/activation/naive/` and build into a single
per-category extension.

## Kernels

### SiLU (a.k.a. Swish)

`y = x * sigmoid(x)`.

- `reference/eager_impl.silu(x)` — eager reference, correctness oracle.
- `reference/torch_impl.silu(x)` — `F.silu` (fused PyTorch op).
- `naive/triton_impl.silu(x)` — Triton block kernel.
- `naive/cuda_impl.silu(x)` — custom CUDA kernel.

### ReLU

`y = max(x, 0)`.

- `reference/eager_impl.relu(x)` — eager reference, correctness oracle.
- `reference/torch_impl.relu(x)` — `F.relu` (fused PyTorch op).
- `naive/triton_impl.relu(x)` — Triton block kernel.
- `naive/cuda_impl.relu(x)` — custom CUDA kernel.

All four backends per kernel accept a tensor of any shape and return a tensor
of the same shape and dtype on the same device. The triton and cuda backends
require a CUDA tensor that is contiguous.

## References

- Hendrycks & Gimpel (2016), "Gaussian Error Linear Units (GELUs)" — discusses SiLU as a baseline.
- Ramachandran et al. (2017), "Searching for Activation Functions" — popularized as Swish.
