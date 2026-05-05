#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename scalar_t, int MAX_CHUNK>
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

    const bool can_vec = (N * sizeof(scalar_t)) % 16 == 0;

    float sum_sq = 0.0f;
    if (can_vec) {
        if constexpr (std::is_same_v<scalar_t, float>) { 
            // float32 path
            const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
            const int64_t N4 = N / 4;
            
            // vectorized main loop
            for (int64_t i = tid; i < N4; i += n_thread) {
                const float4 v = x4[i];
                sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            }
        } else {
            // float16, bfloat16 path
            const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
            const int64_t N8 = N / 8;

            // vectorized main loop
            for (int64_t i = tid; i < N8; i += n_thread) {
                const float4 raw = x4[i];
                if constexpr (std::is_same_v<scalar_t, at::Half>) {
                    // fp16
                    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const float2 v = __half22float2(h2[k]);
                        sum_sq += v.x * v.x + v.y * v.y;
                    }
                } else {
                    // bf16
                    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const float2 v = __bfloat1622float2(h2[k]);
                        sum_sq += v.x * v.x + v.y * v.y;
                    }
                }
            }
        }
    } else {
        for (int64_t i = tid; i < N; i += n_thread) {
            const float v = static_cast<float>(x[row_off + i]);
            sum_sq += v * v;
        }
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

    if (can_vec) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
            const float4* w4 = reinterpret_cast<const float4*>(weight);
            const int64_t N4 = N / 4;
            float4* y4 = reinterpret_cast<float4*>(y + row_off);

            for (int64_t i = tid; i < N4; i += n_thread) {
                const float4 raw_x = x4[i];
                const float4 raw_w = w4[i];
                const scalar_t* xh = reinterpret_cast<const scalar_t*>(&raw_x);
                const scalar_t* wh = reinterpret_cast<const scalar_t*>(&raw_w);

                float4 raw_y;
                scalar_t* yh = reinterpret_cast<scalar_t*>(&raw_y);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const float v = static_cast<float>(xh[k]);
                    const float wv = static_cast<float>(wh[k]);
                    const float out = v * rstd * wv;
                    yh[k] = static_cast<scalar_t>(out);
                }
                y4[i] = raw_y;
            }
        } else {
            const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
            const float4* w4 = reinterpret_cast<const float4*>(weight);
            const int64_t N8 = N / 8;
            float4* y4 = reinterpret_cast<float4*>(y + row_off);

            for (int64_t i = tid; i < N8; i += n_thread) {
                const float4 raw_x = x4[i];
                const float4 raw_w = w4[i];
                const scalar_t* xh = reinterpret_cast<const scalar_t*>(&raw_x);
                const scalar_t* wh = reinterpret_cast<const scalar_t*>(&raw_w);

                float4 raw_y;
                scalar_t* yh = reinterpret_cast<scalar_t*>(&raw_y);
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    const float v = static_cast<float>(xh[k]);
                    const float wv = static_cast<float>(wh[k]);
                    const float out = v * rstd * wv;
                    yh[k] = static_cast<scalar_t>(out);
                }
                y4[i] = raw_y;
            }
        }
    } else {
        for (int64_t i = tid; i < N; i += n_thread) {
            const float v = static_cast<float>(x[row_off + i]);
            const float w = static_cast<float>(weight[i]);
            const float out = v * rstd * w;
            y[row_off + i] = static_cast<scalar_t>(out);
        }
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
    const int64_t bytes_per_row = N * x.element_size();
    if (bytes_per_row % 16 != 0) {
        TORCH_WARN_ONCE(
            "[rmsnorm] N * dtype_size = ", bytes_per_row, " is not 16-byte aligned; falling back to scalar kernel. ",
            "For peak performance, use N divisible by 4 (fp32) or 8 (bf16/fp16)");
    }
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