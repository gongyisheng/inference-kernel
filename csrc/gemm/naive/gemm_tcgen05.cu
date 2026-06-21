#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Educative GEMM #3: Blackwell 5th-gen tensor cores via `tcgen05.mma` (sm_100a)
// ============================================================================
//
// This is the Blackwell programming model, and it is a clean break from every
// prior generation. Compare the three tiers in this directory:
//
//   wmma  (Volta..Ampere) : a *warp* (32 threads) cooperatively issues an MMA;
//                           operands + accumulator live in registers.
//   wgmma (Hopper)        : a *warpgroup* (128 threads) issues one async MMA;
//                           operands come from shared memory via descriptors,
//                           accumulator still in registers. (see gemm_wgmma.cu)
//   tcgen05 (Blackwell)   : a *single thread* issues the MMA for the whole CTA;
//                           operands come from shared memory via descriptors,
//                           and the accumulator lives in a brand-new on-chip
//                           space called Tensor Memory (TMEM) — NOT registers.
//
// The single-thread issue + TMEM accumulator are the headline changes. Warps no
// longer have a role in *issuing* the MMA; they only show up again in the
// epilogue to drain TMEM back to registers with `tcgen05.ld`. Nothing from the
// old world (ldmatrix, ld.shared, wmma loads, even cp.async) can touch TMEM —
// you must use the new tcgen05.ld/st/cp family.
//
// Minimal flow (one CTA, one 128xBLOCK_N output tile, cta_group::1):
//   1. TMA (cp.async.bulk.tensor) streams an A and B K-strip into shared memory
//      in the "core matrix" layout the MMA descriptors expect.
//   2. tcgen05.alloc reserves BLOCK_N columns of TMEM for the accumulator.
//   3. One elected thread issues tcgen05.mma over the K-strip, accumulating into
//      TMEM; completion is signalled through an mbarrier.
//   4. tcgen05.ld drains the fp32 accumulator from TMEM to registers; we cast to
//      bf16 and store C. tcgen05.dealloc releases the TMEM.
//
// Scope (kept deliberately small for teaching, not for peak FLOPs):
//   * bf16 in / bf16 out, fp32 accumulate.
//   * Tile-aligned shapes only (M%128, N%BLOCK_N, K%BLOCK_K). The Python wrapper
//     routes everything else to the naive `gemm` fallback.
//   * Single-buffered K loop (no software pipelining — that lives in the opt tier).
//   * B is transposed to [N, K] host-side so both operands are K-contiguous,
//     matching the layout tcgen05.mma wants. (A real kernel would encode the
//     [K, N] layout directly in the TMA tensor map instead of paying a copy.)
//
// Adapted, with thanks, from gau-nernst's plain-CUDA+PTX reference:
//   https://github.com/gau-nernst/learn-cuda  (02e_matmul_sm100/matmul_v1.cu)
//   https://gau-nernst.github.io/tcgen05/
// PTX reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
// ============================================================================

#include "dispatch.h"

// tcgen05 + the TMA bulk-tensor / mbarrier PTX below only exist for sm_100a and
// only assemble on CUDA >= 12.8. Everything device-side is gated on __CUDA_ARCH__
// so the other arch slices of the fatbin (e.g. sm_90a, when present) compile to
// empty bodies; the host launcher is gated on the toolkit version.
#define IK_TCGEN05_DEVICE_OK (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
#define IK_TCGEN05_HOST_OK   (CUDA_VERSION >= 12080)

#if IK_TCGEN05_HOST_OK
#include <cudaTypedefs.h>
#endif

namespace {

#if IK_TCGEN05_DEVICE_OK

// Elect a single leader lane within the warp (returns 1 on exactly one lane).
__device__ inline uint32_t elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFFu));
    return pred;
}

__device__ inline void mbarrier_init(int mbar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar), "r"(count));
}

// Spin until the mbarrier reaches `phase` (try_wait loops because a single
// try_wait can spuriously return before the phase flips).
__device__ inline void mbarrier_wait(int mbar, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar), "r"(phase), "r"(ticks));
}

// TMA bulk copy of one 2D (height x 8) bf16 tile from global to shared.
__device__ inline void tma_2d_g2s(int dst, const void* tmap, int x, int y, int mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// Encode a byte offset / address into the 14-bit descriptor field format.
__device__ inline uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }

// tcgen05.mma, both operands in shared memory (SS form). `enable_d` selects
// whether to accumulate onto the existing TMEM value (0 on the first K step).
__device__ inline void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc,
                                       uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_d));
}

#endif  // IK_TCGEN05_DEVICE_OK

template <int BLOCK_N, int BLOCK_K>
__global__ void gemm_tcgen05_kernel(
#if IK_TCGEN05_HOST_OK
    const __grid_constant__ CUtensorMap a_tmap,
    const __grid_constant__ CUtensorMap b_tmap,
#endif
    __nv_bfloat16* __restrict__ c,
    int M, int N, int K
) {
#if IK_TCGEN05_DEVICE_OK
    constexpr int BLOCK_M = 128;
    constexpr int MMA_K = 16;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int grid_n = N / BLOCK_N;
    const int off_m = (blockIdx.x / grid_n) * BLOCK_M;
    const int off_n = (blockIdx.x % grid_n) * BLOCK_N;

    // Shared memory: A tile then B tile, in the core-matrix layout TMA produces.
    extern __shared__ __align__(1024) char smem[];
    const int a_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int b_smem = a_smem + BLOCK_M * BLOCK_K * (int)sizeof(__nv_bfloat16);

    // One mbarrier (reused for TMA-done then MMA-done) and the TMEM base address.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbar_storage[1];
    __shared__ int tmem_addr[1];
    const int mbar = static_cast<int>(__cvta_generic_to_shared(mbar_storage));

    if (warp_id == 0 && elect_one()) {
        mbarrier_init(mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(BLOCK_N));
    }
    __syncthreads();
    const int taddr = tmem_addr[0];
    int phase = 0;

    // 32-bit MMA instruction descriptor: fp32 accumulate, bf16 inputs, tile MxN.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    constexpr uint32_t i_desc = (1U << 4U)                            // D dtype = FP32
                              | (1U << 7U)                            // A dtype = BF16
                              | (1U << 10U)                           // B dtype = BF16
                              | ((uint32_t)BLOCK_N >> 3U << 17U)      // MMA N
                              | ((uint32_t)BLOCK_M >> 4U << 24U);     // MMA M

    auto make_desc = [](int addr, int height) -> uint64_t {
        const int LBO = height * 16;  // leading byte offset between K core matrices
        const int SBO = 8 * 16;       // stride byte offset between M/N core matrices
        return desc_encode(addr) | (desc_encode(LBO) << 16ULL)
             | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
    };

    const int num_iters = K / BLOCK_K;
    for (int it = 0; it < num_iters; ++it) {
        // 1. TMA the next A and B K-strip into shared memory (8 cols per issue).
        if (warp_id == 0 && elect_one()) {
            for (int k = 0; k < BLOCK_K / 8; ++k) {
                const int off_k = it * BLOCK_K + k * 8;
                tma_2d_g2s(a_smem + k * BLOCK_M * 16, &a_tmap, off_k, off_m, mbar);
                tma_2d_g2s(b_smem + k * BLOCK_N * 16, &b_tmap, off_k, off_n, mbar);
            }
            constexpr int cp_bytes = (BLOCK_M + BLOCK_N) * BLOCK_K * (int)sizeof(__nv_bfloat16);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(mbar), "r"(cp_bytes) : "memory");
        }
        mbarrier_wait(mbar, phase);
        asm volatile("tcgen05.fence::after_thread_sync;");
        phase ^= 1;

        // 2. One thread issues the K-strip of MMAs into the TMEM accumulator.
        if (warp_id == 0 && elect_one()) {
            // First K step (it==0, k==0) must overwrite, not accumulate.
            tcgen05_mma_f16(taddr, make_desc(a_smem, BLOCK_M),
                            make_desc(b_smem, BLOCK_N), i_desc, it);
            for (int k = 1; k < BLOCK_K / MMA_K; ++k)
                tcgen05_mma_f16(taddr, make_desc(a_smem + k * BLOCK_M * 32, BLOCK_M),
                                make_desc(b_smem + k * BLOCK_N * 32, BLOCK_N), i_desc, 1);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                         :: "r"(mbar) : "memory");
        }
        mbarrier_wait(mbar, phase);
        phase ^= 1;
    }

    // 3. Epilogue: drain fp32 accumulator from TMEM, cast to bf16, store C.
    asm volatile("tcgen05.fence::after_thread_sync;");
    for (int n = 0; n < BLOCK_N / 8; ++n) {
        float tmp[8];
        const int addr = taddr + ((warp_id * 32) << 16) + (n * 8);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                       "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                     : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        __nv_bfloat162 out[4];
        for (int i = 0; i < 4; ++i)
            out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});
        __nv_bfloat16* out_ptr = c + (off_m + tid) * N + (off_n + n * 8);
        reinterpret_cast<int4*>(out_ptr)[0] = reinterpret_cast<int4*>(out)[0];
    }
    __syncthreads();
    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(BLOCK_N));
#endif  // IK_TCGEN05_DEVICE_OK
}

#if IK_TCGEN05_HOST_OK
inline void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char* msg = nullptr;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unknown CUDA driver error";
    TORCH_CHECK(false, "[gemm_tcgen05] ", msg);
}

// Tensor map for a [height, K] row-major bf16 matrix, tiled into shared as
// (shared_height x 8) boxes (8 = one bf16 core-matrix column group).
inline void init_tmap(CUtensorMap* tmap, const __nv_bfloat16* ptr,
                      uint64_t height, uint64_t K, uint32_t shared_height) {
    constexpr uint32_t rank = 2;
    uint64_t global_dim[rank]      = {K, height};
    uint64_t global_strides[rank-1]= {K * sizeof(__nv_bfloat16)};
    uint32_t box_dim[rank]         = {8, shared_height};
    uint32_t elem_strides[rank]    = {1, 1};
    check_cu(cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        global_dim, global_strides, box_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
#endif  // IK_TCGEN05_HOST_OK

}  // namespace

// Host entry. `out`, `a` are [M,K]/[M,N]; `b` is [K,N] row-major (standard
// A@B). Requires bf16 and tile-aligned dims; the Python wrapper guarantees both
// and routes anything else to the naive fallback.
void gemm_tcgen05(torch::Tensor out, torch::Tensor a, torch::Tensor b) {
#if IK_TCGEN05_HOST_OK
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    TORCH_CHECK(prop.major >= 10,
                "[gemm_tcgen05] requires a Blackwell (sm_100+) GPU; got sm_",
                prop.major, prop.minor);
    TORCH_CHECK(a.scalar_type() == at::ScalarType::BFloat16,
                "[gemm_tcgen05] this educative kernel supports bf16 only");

    constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % BLOCK_K == 0,
                "[gemm_tcgen05] dims must be tile-aligned (M%", BLOCK_M, ", N%",
                BLOCK_N, ", K%", BLOCK_K, "); got ", M, "x", N, "x", K);

    // Transpose B to [N, K] so both operands are K-contiguous (see header note).
    auto bt = b.t().contiguous();

    CUtensorMap a_tmap, b_tmap;
    init_tmap(&a_tmap, static_cast<const __nv_bfloat16*>(a.data_ptr()), M, K, BLOCK_M);
    init_tmap(&b_tmap, static_cast<const __nv_bfloat16*>(bt.data_ptr()), N, K, BLOCK_N);

    const int grid = (M / BLOCK_M) * (N / BLOCK_N);
    const int smem = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__nv_bfloat16);
    auto kernel = gemm_tcgen05_kernel<BLOCK_N, BLOCK_K>;
    if (smem > 48000)
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<grid, 128, smem>>>(
        a_tmap, b_tmap, static_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K);
    C10_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "[gemm_tcgen05] requires CUDA toolkit >= 12.8 to compile tcgen05");
#endif
}
