#include <torch/library.h>
#include <torch/types.h>

#include "registration.h"

void relu_forward(torch::Tensor out, torch::Tensor x);
void silu_forward(torch::Tensor out, torch::Tensor x);

TORCH_LIBRARY_FRAGMENT(inference_kernel, m) {
    m.def("relu_forward(Tensor! out, Tensor x) -> ()");
    m.impl("relu_forward", torch::kCUDA, &relu_forward);

    m.def("silu_forward(Tensor! out, Tensor x) -> ()");
    m.impl("silu_forward", torch::kCUDA, &silu_forward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
