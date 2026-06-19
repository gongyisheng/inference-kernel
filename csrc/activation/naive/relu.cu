#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int64_t n
) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const float xv = static_cast<float>(x[i]);
        const float yv = max(xv, 0.0f);
        y[i] = static_cast<scalar_t>(yv);
    }
}

torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    if (n == 0) return y;

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "relu_forward", [&] {
            relu_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                n
            );
        }
    );
    return y;
}
