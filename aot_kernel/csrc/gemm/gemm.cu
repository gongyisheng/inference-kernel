#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "dispatch.h"
#include "cast.cuh"

using namespace nvcuda;

// WM/WN/WK are the WMMA fragment shape (must be a hardware-supported combo,
// e.g. 16x16x16); passed as template params so they're compile-time constants.
template <typename scalar_t, int BLOCK_M, int BLOCK_N, int WM, int WN, int WK>
__global__ void gemm_wmma_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn
){
    // warp index in block
    const int warp_m = threadIdx.y;
    const int warp_n = threadIdx.x / 32;
    // tile's top-left element in the block
    const int tile_m = warp_m * WM;
    const int tile_n = warp_n * WN;
    // tile's top-left element in the global matrix (a/b/c)
    const int global_m = blockIdx.y * BLOCK_M + tile_m;
    const int global_n = blockIdx.x * BLOCK_N + tile_n;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, scalar_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, scalar_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += WK) {
        wmma::load_matrix_sync(a_frag, a + global_m * stride_am + k * stride_ak, stride_am);
        wmma::load_matrix_sync(b_frag, b + k * stride_bk + global_n * stride_bn, stride_bk);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // store_matrix_sync only emits fp32, so stage this warp's tile in shared
    // memory, then convert fp32 -> scalar_t while writing back the valid region.
    __shared__ float c_tile[BLOCK_M][BLOCK_N];
    wmma::store_matrix_sync(&c_tile[tile_m][tile_n], acc, BLOCK_N, wmma::mem_row_major);

    const int lane = threadIdx.x % 32;
    for (int i = lane; i < WM * WN; i += 32) {
        const int frag_m = i / WN, frag_n = i % WN;   // element inside the warp's tile
        if (global_m + frag_m < M && global_n + frag_n < N)
            c[(global_m + frag_m) * stride_cm + (global_n + frag_n) * stride_cn] =
                from_float<scalar_t>(c_tile[tile_m + frag_m][tile_n + frag_n]);
    }
}

// Generic SIMT fallback: shared-memory tiled, fp32 accumulate, one thread per
// output element. Cooperatively stages BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N
// tiles, bounds-checked. Handles any dtype (fp32/fp16/bf16) and any M/N/K.
template <typename scalar_t, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_simt_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn
){
    __shared__ float a_shared[BLOCK_M][BLOCK_K];
    __shared__ float b_shared[BLOCK_K][BLOCK_N];

    const int64_t m  = blockIdx.y * blockDim.y + threadIdx.y; // global row (A tile row origin + ty)
    const int64_t n  = blockIdx.x * blockDim.x + threadIdx.x; // global col (B tile col origin + tx)
    const int     ty = threadIdx.y;                           // [0, BLOCK_M)
    const int     tx = threadIdx.x;                           // [0, BLOCK_N)

    float acc = 0.0f;   // this thread owns one output element

    for (int64_t k = 0; k < K; k += BLOCK_K) {
        // A tile [BLOCK_M][BLOCK_K]: row fixed = ty, loop columns with x-threads
        for (int kc = tx; kc < BLOCK_K; kc += BLOCK_N) {
            const int64_t a_col = k + kc;
            a_shared[ty][kc] = (m < M && a_col < K)
                ? to_float(a[m * stride_am + a_col * stride_ak]) : 0.0f;
        }
        // B tile [BLOCK_K][BLOCK_N]: col fixed = tx, loop rows with y-threads
        for (int kr = ty; kr < BLOCK_K; kr += BLOCK_M) {
            const int64_t b_row = k + kr;
            b_shared[kr][tx] = (b_row < K && n < N)
                ? to_float(b[b_row * stride_bk + n * stride_bn]) : 0.0f;
        }
        __syncthreads();   // tile ready

        for (int kk = 0; kk < BLOCK_K; ++kk)
            acc += a_shared[ty][kk] * b_shared[kk][tx];
        __syncthreads();   // before next k overwrites the tile
    }

    if (m < M && n < N)
        c[m * stride_cm + n * stride_cn] = from_float<scalar_t>(acc);
}

void gemm(
    torch::Tensor out,
    torch::Tensor a,
    torch::Tensor b
){
    const int64_t M = a.size(0);
    const int64_t N = b.size(1);
    const int64_t K = a.size(1);

    const auto dtype = a.scalar_type();
    const bool is_half = (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16);

    constexpr int WM = 16, WN = 16, WK = 16;   // WMMA fragment shape

    bool ok;
    if (is_half && K % WK == 0) {
        // Tensor-core fast path: fp16/bf16 with K aligned to the fragment depth.
        constexpr int BLOCK_M = 64, BLOCK_N = 64;
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        const dim3 block(128, 4);  // 16 warps -> one 64x64 C tile
        ok = DISPATCH_HALF_TYPES(dtype, c_type, [&] {
            gemm_wmma_kernel<c_type, BLOCK_M, BLOCK_N, WM, WN, WK><<<grid, block>>>(
                static_cast<const c_type*>(a.data_ptr()),
                static_cast<const c_type*>(b.data_ptr()),
                static_cast<c_type*>(out.data_ptr()),
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                out.stride(0), out.stride(1)
            );
            return true;
        });
    } else {
        // Universal fallback: fp32, or any shape the WMMA path can't handle.
        constexpr int BLOCK_M = 16, BLOCK_N = 16, BLOCK_K = 16;
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        const dim3 block(BLOCK_N, BLOCK_M);
        ok = DISPATCH_FLOATING_TYPES(dtype, c_type, [&] {
            gemm_simt_kernel<c_type, BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block>>>(
                static_cast<const c_type*>(a.data_ptr()),
                static_cast<const c_type*>(b.data_ptr()),
                static_cast<c_type*>(out.data_ptr()),
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                out.stride(0), out.stride(1)
            );
            return true;
        });
    }
    TORCH_CHECK(ok, "[gemm] unsupported dtype: ", dtype);
}
