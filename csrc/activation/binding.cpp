#include <torch/extension.h>

torch::Tensor silu_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_forward", &silu_forward, "silu forward cuda kernel");
}
