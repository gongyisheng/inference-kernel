# Activation kernels

Element-wise activations used in transformer FFN blocks. Each backend lives
in a single file at this directory level — `torch_impl.py`, `triton_impl.py`,
`cuda_impl.py` — and exposes one function per activation. CUDA sources live
in `csrc/activation/` and build into a single per-category extension.

## Kernels

### SiLU (a.k.a. Swish)

`y = x * sigmoid(x)`.

- `torch_impl.silu(x)` — eager reference, correctness oracle.
- `triton_impl.silu(x)` — Triton block kernel.
- `cuda_impl.silu(x)` — custom CUDA kernel.

All three accept a tensor of any shape and return a tensor of the same shape
and dtype on the same device. The triton and cuda backends require a CUDA
tensor that is contiguous.

## References

- Hendrycks & Gimpel (2016), "Gaussian Error Linear Units (GELUs)" — discusses SiLU as a baseline.
- Ramachandran et al. (2017), "Searching for Activation Functions" — popularized as Swish.
