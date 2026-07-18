#pragma once

#include <torch/types.h>

// All CUDA ops exposed by the aot_kernel extension. Implementations live in the
// per-category .cu files; schemas + registration live in registration.cc.

// activation
void relu_forward(torch::Tensor out, torch::Tensor x);
void silu_forward(torch::Tensor out, torch::Tensor x);

// norm
void rmsnorm_forward(torch::Tensor out, torch::Tensor x, torch::Tensor weight, double eps);

// gemm
void gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_opt(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_cutlass(torch::Tensor out, torch::Tensor a, torch::Tensor b);
void gemm_cutlass_fused_act(torch::Tensor out, torch::Tensor a, torch::Tensor b, int64_t activation);
