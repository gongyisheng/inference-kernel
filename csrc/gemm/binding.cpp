#include <torch/extension.h>

torch::Tensor gemm(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "gemm cuda kernel");
}