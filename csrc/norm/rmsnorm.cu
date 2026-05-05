#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t stride,
    int64_t N,
    float eps
){
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_thread = blockDim.x;
    const int64_t row_off = row * stride;
    
    float sum_sq = 0.0f;
    for (int64_t i = tid; i < N; i += n_thread) {
        const float v = static_cast<float>(x[row_off + i]);
        sum_sq += v * v;
    }

    __shared__ float sdata[256];
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = n_thread / 2; s >= 32; s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    const float total_sq = sdata[0];

    const float mean = total_sq / static_cast<float>(N);
    const float rstd = rsqrtf(mean + eps);
    for (int64_t i = tid; i < N; i += n_thread) {
        const float v = static_cast<float>(x[row_off + i]);
        const float w = static_cast<float>(weight[i]);
        const float out = v * rstd * w;
        y[row_off + i] = static_cast<scalar_t>(out);
    }
}


torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
){
    TORCH_CHECK(x.is_cuda(), "[rmsnorm] x must be a cuda tensor, got ", x.device());
    TORCH_CHECK(weight.is_cuda(), "[rmsnorm] weight must be a cuda tensor, got ", weight.device());
    TORCH_CHECK(x.device() == weight.device(), "[rmsnorm] x and weight are not on the same device, got x.device=", x.device(), " weight.device=", weight.device());
    TORCH_CHECK(x.is_contiguous(), "[rmsnorm] x is not contiguous");
    TORCH_CHECK(weight.is_contiguous(), "[rmsnorm] weight is not contiguous");
    TORCH_CHECK(x.dtype() == weight.dtype(), "[rmsnorm] x and weight has different dtype, got x.dtype=", x.dtype(), " weight.dtype=", weight.dtype());
    TORCH_CHECK(weight.dim() == 1, "[rmsnorm] dim of weight is not 1, got ", weight.dim());
    TORCH_CHECK(weight.size(0) == x.size(-1), "[rmsnorm] weight length must equal to x.size(-1), got weight_length=", weight.size(0), " x.size(-1)=", x.size(-1));

    auto y = torch::empty_like(x);
    if (x.numel() == 0) return y;

    const int64_t N = x.size(-1);
    const int64_t M = x.numel() / N;
    const int threads = 256;

    dim3 grid(M);
    dim3 block(threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rmsnorm_forward", [&] {
            rmsnorm_kernel<scalar_t><<<grid, block>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                N,
                N,
                static_cast<float>(eps)
            );
        }
    );

    return y;
}