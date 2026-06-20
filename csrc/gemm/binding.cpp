#include <torch/library.h>
#include <torch/types.h>

#include "registration.h"

void gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b);

TORCH_LIBRARY_FRAGMENT(inference_kernel, m) {
    m.def("gemm(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm", torch::kCUDA, &gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
