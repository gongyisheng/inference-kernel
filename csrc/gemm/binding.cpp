#include <torch/library.h>
#include <torch/types.h>

#include "registration.h"

void gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_opt(torch::Tensor out, torch::Tensor a, torch::Tensor b);

TORCH_LIBRARY_FRAGMENT(inference_kernel, m) {
    m.def("gemm(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm", torch::kCUDA, &gemm);

    m.def("gemm_opt(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm_opt", torch::kCUDA, &gemm_opt);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
