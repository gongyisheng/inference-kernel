#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Block-wide sum reduction: shared-memory tree down to 32, then warp shuffle.
__device__ inline float block_reduce_sum(float val, float* sdata) {
    const int tid = threadIdx.x;
    const int n_thread = blockDim.x;

    sdata[tid] = val;
    __syncthreads();

    for (int s = n_thread / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float v = sdata[tid];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    return sdata[0];
}


__global__ void rmsnorm_kernel_fp32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int64_t stride,
    int64_t N,
    float eps
){
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_thread = blockDim.x;
    const int64_t row_off = row * stride;
    const bool can_vec = (N * sizeof(float)) % 16 == 0;

    float sum_sq = 0.0f;
    if (can_vec) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
        const int64_t N4 = N / 4;
        for (int64_t i = tid; i < N4; i += n_thread) {
            const float4 v = x4[i];
            sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        for (int64_t i = tid; i < N; i += n_thread) {
            const float v = x[row_off + i];
            sum_sq += v * v;
        }
    }

    extern __shared__ float sdata[];
    const float total_sq = block_reduce_sum(sum_sq, sdata);
    const float rstd = rsqrtf(total_sq / static_cast<float>(N) + eps);

    if (can_vec) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        float4* y4 = reinterpret_cast<float4*>(y + row_off);
        const int64_t N4 = N / 4;
        for (int64_t i = tid; i < N4; i += n_thread) {
            const float4 xv = x4[i];
            const float4 wv = w4[i];
            float4 yv;
            yv.x = xv.x * rstd * wv.x;
            yv.y = xv.y * rstd * wv.y;
            yv.z = xv.z * rstd * wv.z;
            yv.w = xv.w * rstd * wv.w;
            y4[i] = yv;
        }
    } else {
        for (int64_t i = tid; i < N; i += n_thread) {
            y[row_off + i] = x[row_off + i] * rstd * weight[i];
        }
    }
}


// Traits to abstract the half2 vector type and its float<->half conversions.
template <typename scalar_t> struct HalfTraits;

template <> struct HalfTraits<__half> {
    using half2_t = __half2;
    static __device__ inline float to_float(__half v) { return __half2float(v); }
    static __device__ inline __half from_float(float v) { return __float2half_rn(v); }
    static __device__ inline float2 to_float2(half2_t v) { return __half22float2(v); }
    static __device__ inline half2_t from_float2(float2 v) { return __float22half2_rn(v); }
};

template <> struct HalfTraits<__nv_bfloat16> {
    using half2_t = __nv_bfloat162;
    static __device__ inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ inline __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
    static __device__ inline float2 to_float2(half2_t v) { return __bfloat1622float2(v); }
    static __device__ inline half2_t from_float2(float2 v) { return __float22bfloat162_rn(v); }
};


template <typename scalar_t>
__global__ void rmsnorm_kernel_fp16_bf16(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t stride,
    int64_t N,
    float eps
){
    using T = HalfTraits<scalar_t>;
    using half2_t = typename T::half2_t;

    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_thread = blockDim.x;
    const int64_t row_off = row * stride;
    const bool can_vec = (N * sizeof(scalar_t)) % 16 == 0;

    float sum_sq = 0.0f;
    if (can_vec) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
        const int64_t N8 = N / 8;
        for (int64_t i = tid; i < N8; i += n_thread) {
            const float4 raw = x4[i];
            const half2_t* h2 = reinterpret_cast<const half2_t*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const float2 v = T::to_float2(h2[k]);
                sum_sq += v.x * v.x + v.y * v.y;
            }
        }
    } else {
        for (int64_t i = tid; i < N; i += n_thread) {
            const float v = T::to_float(x[row_off + i]);
            sum_sq += v * v;
        }
    }

    extern __shared__ float sdata[];
    const float total_sq = block_reduce_sum(sum_sq, sdata);
    const float rstd = rsqrtf(total_sq / static_cast<float>(N) + eps);

    if (can_vec) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row_off);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        float4* y4 = reinterpret_cast<float4*>(y + row_off);
        const int64_t N8 = N / 8;
        for (int64_t i = tid; i < N8; i += n_thread) {
            const float4 raw_x = x4[i];
            const float4 raw_w = w4[i];
            const half2_t* xh2 = reinterpret_cast<const half2_t*>(&raw_x);
            const half2_t* wh2 = reinterpret_cast<const half2_t*>(&raw_w);

            float4 raw_y;
            half2_t* yh2 = reinterpret_cast<half2_t*>(&raw_y);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const float2 xv = T::to_float2(xh2[k]);
                const float2 wv = T::to_float2(wh2[k]);
                const float2 out{xv.x * rstd * wv.x, xv.y * rstd * wv.y};
                yh2[k] = T::from_float2(out);
            }
            y4[i] = raw_y;
        }
    } else {
        for (int64_t i = tid; i < N; i += n_thread) {
            const float v = T::to_float(x[row_off + i]);
            const float w = T::to_float(weight[i]);
            y[row_off + i] = T::from_float(v * rstd * w);
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
    const int vec_size = (bytes_per_row % 16 == 0) ? (16 / x.element_size()) : 1;
    const int64_t work_units = N / vec_size;
    int threads = 32;
    while (threads < work_units && threads < 1024) {
        threads *= 2;
    }
    const size_t smem_bytes = threads * sizeof(float);

    dim3 grid(M);
    dim3 block(threads);
    const float eps_f = static_cast<float>(eps);

    switch (x.scalar_type()) {
        case at::ScalarType::Float:
            rmsnorm_kernel_fp32<<<grid, block, smem_bytes>>>(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                y.data_ptr<float>(),
                N, N, eps_f);
            break;
        case at::ScalarType::Half:
            rmsnorm_kernel_fp16_bf16<__half><<<grid, block, smem_bytes>>>(
                reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
                N, N, eps_f);
            break;
        case at::ScalarType::BFloat16:
            rmsnorm_kernel_fp16_bf16<__nv_bfloat16><<<grid, block, smem_bytes>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
                N, N, eps_f);
            break;
        default:
            TORCH_CHECK(false, "[rmsnorm] unsupported dtype: ", x.scalar_type());
    }

    return y;
}
