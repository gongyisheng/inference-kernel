#include <torch/library.h>
#include <torch/types.h>

#include "registration.h"

void gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_opt(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_wgmma(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_tcgen05(torch::Tensor out, torch::Tensor a, torch::Tensor b);

TORCH_LIBRARY_FRAGMENT(inference_kernel, m) {
    m.def("gemm(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm", torch::kCUDA, &gemm);

    m.def("gemm_opt(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm_opt", torch::kCUDA, &gemm_opt);

    // Educative per-generation tensor-core kernels (see naive/gemm_wgmma.cu,
    // naive/gemm_tcgen05.cu). Selected by compute capability in Python.
    m.def("gemm_wgmma(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm_wgmma", torch::kCUDA, &gemm_wgmma);

    m.def("gemm_tcgen05(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm_tcgen05", torch::kCUDA, &gemm_tcgen05);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
