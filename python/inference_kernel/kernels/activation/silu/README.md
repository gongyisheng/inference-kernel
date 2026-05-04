# SiLU (Sigmoid Linear Unit, a.k.a. Swish)

`y = x * sigmoid(x)` — element-wise activation used in modern transformer FFN blocks.

## Backends

- `torch_impl.silu(x)` — eager reference. Correctness oracle.
- `triton_impl.silu(x)` — Triton block kernel.
- `cuda_impl.silu(x)` — Custom CUDA kernel via `torch.utils.cpp_extension`.

All three accept a tensor of any shape on any device and return a tensor of
the same shape and dtype on the same device. CUDA backends require
`x.is_contiguous()` and a CUDA tensor.

## References

- Hendrycks & Gimpel (2016), "Gaussian Error Linear Units (GELUs)" — discusses SiLU as a baseline.
- Ramachandran et al. (2017), "Searching for Activation Functions" — popularized as Swish.
