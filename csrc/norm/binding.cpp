#include <torch/library.h>
#include <torch/types.h>

#include "registration.h"

void rmsnorm_forward(torch::Tensor out, torch::Tensor x, torch::Tensor weight, double eps);

TORCH_LIBRARY_FRAGMENT(inference_kernel, m) {
    m.def("rmsnorm_forward(Tensor! out, Tensor x, Tensor weight, float eps) -> ()");
    m.impl("rmsnorm_forward", torch::kCUDA, &rmsnorm_forward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
