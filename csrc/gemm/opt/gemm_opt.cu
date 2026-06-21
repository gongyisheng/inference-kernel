#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdint.h>

#include "dispatch.h"
#include "cast.cuh"

using namespace nvcuda;

// Register-blocked, software-pipelined WMMA GEMM.
//
// Builds on three ideas, each fixing the bottleneck the previous one exposes:
//   1. Register blocking: each warp keeps a WARP_FRAGS_M x WARP_FRAGS_N grid of
//      accumulator fragments live in registers and reuses each loaded operand
//      fragment across the grid -> load:MMA drops below 1 (relieves the MIO/L1
//      load-pipe throttle that caps the naive WMMA kernel).
//   2. Shared-memory staging with zero-padding: partial M/N/K tiles are correct
//      without out-of-bounds global reads.
//   3. cp.async double-buffering: the next K-strip's global->shared copy is
//      issued asynchronously and overlapped with the current strip's MMA work,
//      so the tensor cores no longer stall at the load barrier. Copies use
//      16-byte (8-element) cp.async, falling back to scalar stores for
//      unaligned / partial edge chunks.
//
// WMMA fragment shape is fixed at 16x16x16. BLOCK_K must be a multiple of 16.
template <
    typename scalar_t,
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int WARPS_M, int WARPS_N,
    int WARP_FRAGS_M, int WARP_FRAGS_N>
__global__ void gemm_wmma_pipe_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn
){
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int VEC = 16 / sizeof(scalar_t);  // 8 elements = 16 bytes per cp.async
    // Pad each shared row by 16 bytes so consecutive rows skew across the 32
    // banks (un-padded strides of 32/128 halves collide hard on the fragment
    // reads). A multiple of VEC keeps every cp.async destination 16B-aligned.
    constexpr int SKEW = VEC;
    constexpr int LDA = BLOCK_K + SKEW;
    constexpr int LDB = BLOCK_N + SKEW;
    static_assert(BLOCK_K % WK == 0, "BLOCK_K must be a multiple of 16");
    static_assert(BLOCK_K % VEC == 0 && BLOCK_N % VEC == 0, "tile dims must be VEC-aligned");
    static_assert(WARPS_M * WARP_FRAGS_M * WM == BLOCK_M, "M tiling mismatch");
    static_assert(WARPS_N * WARP_FRAGS_N * WN == BLOCK_N, "N tiling mismatch");

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    const int warp_row = warp_m * WARP_FRAGS_M * WM;
    const int warp_col = warp_n * WARP_FRAGS_N * WN;
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    __shared__ scalar_t a_buf[2][BLOCK_M][LDA];
    __shared__ scalar_t b_buf[2][BLOCK_K][LDB];

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[WARP_FRAGS_M][WARP_FRAGS_N];
    for (int i = 0; i < WARP_FRAGS_M; ++i)
        for (int j = 0; j < WARP_FRAGS_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int tid = threadIdx.x;
    const int nthreads = WARPS_M * WARPS_N * 32;
    const scalar_t zero = from_float<scalar_t>(0.0f);
    const int num_strips = (K + BLOCK_K - 1) / BLOCK_K;

    // Issue (async where possible) the global->shared copy of one K-strip into
    // buffer `st`. cp.async needs 16B-aligned, fully in-range chunks; anything
    // else (matrix edge, unaligned row pitch) falls back to a scalar store.
    auto load_strip = [&](int st, int k0) {
        for (int ci = tid; ci < BLOCK_M * (BLOCK_K / VEC); ci += nthreads) {
            const int r = ci / (BLOCK_K / VEC);
            const int col0 = (ci % (BLOCK_K / VEC)) * VEC;
            const int gm = block_row + r, gk = k0 + col0;
            scalar_t* dst = &a_buf[st][r][col0];
            const scalar_t* src = a + gm * stride_am + gk * stride_ak;
            if (gm < M && gk + VEC <= K && stride_ak == 1 &&
                (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                for (int i = 0; i < VEC; ++i)
                    dst[i] = (gm < M && gk + i < K) ? a[gm * stride_am + (gk + i) * stride_ak] : zero;
            }
        }
        for (int ci = tid; ci < BLOCK_K * (BLOCK_N / VEC); ci += nthreads) {
            const int r = ci / (BLOCK_N / VEC);
            const int col0 = (ci % (BLOCK_N / VEC)) * VEC;
            const int gk = k0 + r, gn = block_col + col0;
            scalar_t* dst = &b_buf[st][r][col0];
            const scalar_t* src = b + gk * stride_bk + gn * stride_bn;
            if (gk < K && gn + VEC <= N && stride_bn == 1 &&
                (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                for (int i = 0; i < VEC; ++i)
                    dst[i] = (gk < K && gn + i < N) ? b[gk * stride_bk + (gn + i) * stride_bn] : zero;
            }
        }
    };

    load_strip(0, 0);
    __pipeline_commit();

    int stage = 0;
    for (int s = 0; s < num_strips; ++s) {
        const bool has_next = s + 1 < num_strips;
        if (has_next) {
            load_strip(stage ^ 1, (s + 1) * BLOCK_K);
            __pipeline_commit();
        }
        __pipeline_wait_prior(has_next ? 1 : 0);  // current strip's copy is now done
        __syncthreads();

        for (int kk = 0; kk < BLOCK_K; kk += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, scalar_t, wmma::row_major> a_frag[WARP_FRAGS_M];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, scalar_t, wmma::row_major> b_frag[WARP_FRAGS_N];
            for (int i = 0; i < WARP_FRAGS_M; ++i)
                wmma::load_matrix_sync(a_frag[i], &a_buf[stage][warp_row + i * WM][kk], LDA);
            for (int j = 0; j < WARP_FRAGS_N; ++j)
                wmma::load_matrix_sync(b_frag[j], &b_buf[stage][kk][warp_col + j * WN], LDB);
            for (int i = 0; i < WARP_FRAGS_M; ++i)
                for (int j = 0; j < WARP_FRAGS_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();  // done reading this buffer before it is reused
        stage ^= 1;
    }

    // Epilogue: store_matrix_sync only emits fp32. Stage one fragment at a time
    // in a small per-warp shared buffer, then convert fp32 -> scalar_t while
    // writing back the in-range region.
    __shared__ float c_buf[WARPS_M * WARPS_N][WM][WN];
    for (int i = 0; i < WARP_FRAGS_M; ++i) {
        for (int j = 0; j < WARP_FRAGS_N; ++j) {
            wmma::store_matrix_sync(&c_buf[warp_id][0][0], acc[i][j], WN, wmma::mem_row_major);
            __syncwarp();
            const int gm0 = block_row + warp_row + i * WM;
            const int gn0 = block_col + warp_col + j * WN;
            for (int e = lane; e < WM * WN; e += 32) {
                const int r = e / WN, col = e % WN;
                if (gm0 + r < M && gn0 + col < N)
                    c[(gm0 + r) * stride_cm + (gn0 + col) * stride_cn] =
                        from_float<scalar_t>(c_buf[warp_id][r][col]);
            }
            __syncwarp();  // reuse c_buf for the next fragment
        }
    }
}

// Tensor-core GEMM for fp16/bf16 with K a multiple of 16. The Python wrapper
// routes fp32 / K-unaligned shapes to the naive tier's universal fallback.
void gemm_opt(
    torch::Tensor out,
    torch::Tensor a,
    torch::Tensor b
){
    const int64_t M = a.size(0);
    const int64_t N = b.size(1);
    const int64_t K = a.size(1);

    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
    constexpr int WARPS_M = 2, WARPS_N = 4;
    constexpr int WARP_FRAGS_M = 4, WARP_FRAGS_N = 2;  // 64x32 per warp, load:MMA = 0.75

    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    const dim3 block(WARPS_M * WARPS_N * 32);

    const bool ok = DISPATCH_HALF_TYPES(a.scalar_type(), c_type, [&] {
        gemm_wmma_pipe_kernel<c_type, BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, WARP_FRAGS_M, WARP_FRAGS_N>
            <<<grid, block>>>(
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
    TORCH_CHECK(ok, "[gemm_opt] tensor-core path supports fp16/bf16 only; got ", a.scalar_type());
}
