#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "dispatch.h"
#include "cast.cuh"

template <typename scalar_t, int BLOCK_K>
__global__ void gemm_kernel_thread(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int64_t K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn
){
    const int64_t m = blockIdx.x;
    const int64_t n = blockIdx.y;
    const int tid = threadIdx.x;
    const int n_thread = blockDim.x;

    const scalar_t* a_row = a + m * stride_am;
    const scalar_t* b_col = b + n * stride_bn;
    scalar_t* c_out = c + m * stride_cm + n * stride_cn;

    // block reduce sum
    extern __shared__ float sdata[];
    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < K; k0 += BLOCK_K) {
        const int64_t k = k0 + tid;
        if (k < K) {
            const float a_val = to_float(a_row[k * stride_ak]);
            const float b_val = to_float(b_col[k * stride_bk]);
            acc += a_val * b_val;
        }
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = n_thread / 2; s >=32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        // Everyone: look at the person at position (your_id+offset), copy their number, and add it to yours.
        float v = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (tid == 0) {
            *c_out = from_float<scalar_t>(v);
        }
    }
}

void gemm(
    torch::Tensor out,
    torch::Tensor a,
    torch::Tensor b
){
    // Input validation lives on the Python side (see gemm/naive/cuda_impl.py).
    const int64_t M = a.size(0);
    const int64_t N = b.size(1);
    const int64_t K = a.size(1);

    constexpr int BLOCK_K = 128;
    const dim3 grid(M, N);
    const dim3 block(BLOCK_K);
    const size_t smem = BLOCK_K * sizeof(float);

    const bool ok = DISPATCH_FLOATING_TYPES(a.scalar_type(), c_type, [&] {
        gemm_kernel_thread<c_type, BLOCK_K><<<grid, block, smem>>>(
            static_cast<const c_type*>(a.data_ptr()),
            static_cast<const c_type*>(b.data_ptr()),
            static_cast<c_type*>(out.data_ptr()), K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1)
        );
        return true;
    });
    TORCH_CHECK(ok, "[gemm] unsupported dtype: ", a.scalar_type());
}