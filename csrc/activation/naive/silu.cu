#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dispatch.h"
#include "cast.cuh"

template <typename scalar_t>
__global__ void silu_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int64_t n
) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const float xv = to_float(x[i]);
        const float yv = xv / (1.0f + expf(-xv));
        y[i] = from_float<scalar_t>(yv);
    }
}

void silu_forward(torch::Tensor out, torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int64_t n = x.numel();
    if (n == 0) return;

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;

    const bool ok = DISPATCH_FLOATING_TYPES(x.scalar_type(), c_type, [&] {
        silu_kernel<c_type><<<blocks, threads>>>(
            static_cast<const c_type*>(x.data_ptr()),
            static_cast<c_type*>(out.data_ptr()),
            n
        );
        return true;
    });
    TORCH_CHECK(ok, "[silu] unsupported dtype: ", x.scalar_type());
}
