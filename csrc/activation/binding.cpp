#include <torch/extension.h>

torch::Tensor relu_forward(torch::Tensor x);
torch::Tensor silu_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "relu forward cuda kernel");
    m.def("silu_forward", &silu_forward, "silu forward cuda kernel");
}
