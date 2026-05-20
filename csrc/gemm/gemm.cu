#include <torch/eatension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename scalar_t>
__global__ void gemm_kernel(a, b) {
    const 
}

torch::Tensor gemm(
    torch::Tensor a,
    torch::Tensor b,
){
    TORCH_CHECK(a.is_cuda(), "[gemm] a must be a cuda tensor, got ", a.device());
    TORCH_CHECK(b.is_cuda(), "[gemm] b must be a cuda tensor, got ", b.device());
    TORCH_CHECK(a.device() == b.device(), "[gemm] a and b are not on the same device, got a.device=", a.device(), " b.device=", b.device());
    TORCH_CHECK(a.is_contiguous(), "[gemm] a is not contiguous");
    TORCH_CHECK(b.is_contiguous(), "[gemm] b is not contiguous");
    TORCH_CHECK(a.dtype() == b.dtype(), "[gemm] a and b has different dtype, got a.dtype=", a.dtype(), " b.dtype=", b.dtype());
    TORCH_CHECK(a.dim() == 2, "[gemm] dim of a is not 2, got ", a.dim());
    TORCH_CHECK(b.dim() == 2, "[gemm] dim of b is not 2, got ", b.dim());
    

    M, K1 = torch::a.shape
    K2, N = torch::b.shape
    TORCH_CHECK(K1 == K2, "[gemm] a.shape[1] must equal to b.shape[0], got a.shape[1]=", K1, " b.shape[0]=", K2);
    int64_t K = K1;
    auto device = a.device();
    auto dtype = a.dtype();
    
    auto c = torch::empty((M, N), device=device, dtype=dtype);

    int64_t block_m = 128;
    int64_t block_n = 128;
    int64_t block_k = 32;

}