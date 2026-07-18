#include <torch/library.h>
#include <torch/types.h>

#include "ops.h"
#include "registration.h"

// Single registration point for every aot_kernel CUDA op. The extension module
// is named _C; importing aot_kernel._C runs these static initializers and makes
// the ops callable as torch.ops.aot_kernel.<op>.
TORCH_LIBRARY(aot_kernel, m) {
    m.def("relu_forward(Tensor! out, Tensor x) -> ()");
    m.impl("relu_forward", torch::kCUDA, &relu_forward);

    m.def("silu_forward(Tensor! out, Tensor x) -> ()");
    m.impl("silu_forward", torch::kCUDA, &silu_forward);

    m.def("rmsnorm_forward(Tensor! out, Tensor x, Tensor weight, float eps) -> ()");
    m.impl("rmsnorm_forward", torch::kCUDA, &rmsnorm_forward);

    m.def("gemm(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm", torch::kCUDA, &gemm);

    m.def("gemm_opt(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm_opt", torch::kCUDA, &gemm_opt);

    m.def("gemm_cutlass(Tensor! out, Tensor a, Tensor b) -> ()");
    m.impl("gemm_cutlass", torch::kCUDA, &gemm_cutlass);

    m.def("gemm_cutlass_fused_act(Tensor! out, Tensor a, Tensor b, int activation) -> ()");
    m.impl("gemm_cutlass_fused_act", torch::kCUDA, &gemm_cutlass_fused_act);
}

REGISTER_EXTENSION(_C)
