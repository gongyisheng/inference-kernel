#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Educative GEMM #2: Hopper warpgroup tensor cores via `wgmma` (sm_90a)
// ============================================================================
//
// The Hopper programming model sits between the warp-level wmma (gemm.cu) and
// the single-thread Blackwell tcgen05 (gemm_tcgen05.cu):
//
//   * A whole *warpgroup* (4 warps = 128 threads) cooperatively issues one
//     asynchronous MMA: `wgmma.mma_async`.
//   * Operands A and B are read directly from shared memory through 64-bit
//     "matrix descriptors" (no ldmatrix into registers first).
//   * The fp32 accumulator still lives in registers, distributed across the
//     128 threads (32 floats/thread for a 64x64 tile).
//   * Issue is async + batched: `wgmma.fence` makes the register/SMEM state
//     visible, you issue a batch of wgmma, `commit_group` closes the batch, and
//     `wait_group` blocks until it retires.
//
// IMPORTANT — this kernel cannot run on the Blackwell B200 in this repo.
//   wgmma is an sm_90a-only instruction; Blackwell (sm_100) *removed* it in
//   favour of tcgen05. So this file is compiled into the fatbin as an sm_90a
//   slice and is correct-by-construction against the PTX ISA + CUTLASS, but it
//   is runtime-validated on Hopper only. The Python wrapper never auto-selects
//   it on Blackwell, and the host entry raises a clear error there. That
//   wgmma->tcgen05 generational break is itself the lesson this file encodes.
//
// Design (deliberately minimal, mirrors gemm_tcgen05.cu so the diff is the
// instruction family, not the scaffolding):
//   * bf16 in / bf16 out, fp32 accumulate; one warpgroup -> one 64x64 C tile.
//   * TMA (cp.async.bulk.tensor, also a Hopper feature) stages A and B K-strips
//     into shared memory in the canonical no-swizzle "core matrix" layout the
//     wgmma descriptors expect — the same layout the tcgen05 kernel uses.
//   * B is transposed to [N, K] host-side so it lands K-contiguous (col-major
//     B), which is what wgmma's B operand wants with trans-B = 0.
//   * Tile-aligned shapes only; single-buffered K loop.
//
// References:
//   wgmma PTX: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions
//   descriptor + fragment layout follows CUTLASS cute/arch/mma_sm90_gmma.hpp
//   Colfax WGMMA tutorial: https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
// ============================================================================

#include "dispatch.h"

// wgmma exists only on sm_90a (NOT sm_80, NOT sm_100). The device guard pins it
// to Hopper so the sm_100a slice of the fatbin compiles to an empty body.
#define IK_WGMMA_DEVICE_OK (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000)
#define IK_WGMMA_HOST_OK   (CUDA_VERSION >= 12000)

#if IK_WGMMA_HOST_OK
#include <cudaTypedefs.h>
#endif

namespace {

#if IK_WGMMA_DEVICE_OK

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

__device__ inline void tma_2d_g2s(int dst, const void* tmap, int x, int y, int mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

__device__ inline uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }

// wgmma shared-memory matrix descriptor (no swizzle, base offset 0):
//   bits 0-13 start addr, 16-29 LBO, 32-45 SBO, 62-63 swizzle mode (=0).
__device__ inline uint64_t make_desc(int addr, int height) {
    const int LBO = height * 16;  // bytes between K-adjacent 8x8 core matrices
    const int SBO = 8 * 16;       // bytes between M/N-adjacent core matrices
    return desc_encode(addr) | (desc_encode(LBO) << 16ULL) | (desc_encode(SBO) << 32ULL);
}

// One m64n64k16 warpgroup MMA, SS form (both operands in SMEM), 32 fp32 accums.
// scale_d = 0 overwrites the accumulator (first K step); 1 accumulates.
__device__ inline void wgmma_m64n64k16(float d[32], uint64_t a_desc, uint64_t b_desc, int scale_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %34, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0;\n\t"
        "}"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),
          "+f"(d[6]),  "+f"(d[7]),  "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
          "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31])
        : "l"(a_desc), "l"(b_desc), "r"(scale_d));
}

#endif  // IK_WGMMA_DEVICE_OK

template <int BLOCK_K>
__global__ void gemm_wgmma_kernel(
#if IK_WGMMA_HOST_OK
    const __grid_constant__ CUtensorMap a_tmap,
    const __grid_constant__ CUtensorMap b_tmap,
#endif
    __nv_bfloat16* __restrict__ c,
    int M, int N, int K
) {
#if IK_WGMMA_DEVICE_OK
    constexpr int BLOCK_M = 64, BLOCK_N = 64, MMA_K = 16;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int grid_n = N / BLOCK_N;
    const int off_m = (blockIdx.x / grid_n) * BLOCK_M;
    const int off_n = (blockIdx.x % grid_n) * BLOCK_N;

    extern __shared__ __align__(1024) char smem[];
    const int a_smem = static_cast<int>(__cvta_generic_to_shared(smem));
    const int b_smem = a_smem + BLOCK_M * BLOCK_K * (int)sizeof(__nv_bfloat16);

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbar_storage[1];
    const int mbar = static_cast<int>(__cvta_generic_to_shared(mbar_storage));
    if (warp_id == 0 && elect_one()) {
        mbarrier_init(mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    float acc[32] = {};
    int phase = 0;
    const int num_iters = K / BLOCK_K;

    for (int it = 0; it < num_iters; ++it) {
        // 1. TMA the next A and B K-strip into shared memory (8 K-cols/issue).
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
        phase ^= 1;

        // 2. Warpgroup issues the K-strip of wgmma into the register accumulator.
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        for (int k = 0; k < BLOCK_K / MMA_K; ++k) {
            uint64_t a_desc = make_desc(a_smem + k * BLOCK_M * 32, BLOCK_M);
            uint64_t b_desc = make_desc(b_smem + k * BLOCK_N * 32, BLOCK_N);
            wgmma_m64n64k16(acc, a_desc, b_desc, (it == 0 && k == 0) ? 0 : 1);
        }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        __syncthreads();  // SMEM consumed; safe for the next TMA to overwrite
    }

    // 3. Epilogue: the m64n64 fp32 fragment layout (PTX 9.7.16.5.1). Each thread
    //    owns 32 floats = 8 n-groups x {(r0,c0),(r0,c1),(r8,c0),(r8,c1)}.
    const int row0 = warp_id * 16 + lane / 4;
    const int col0 = (lane % 4) * 2;
    for (int ng = 0; ng < BLOCK_N / 8; ++ng) {
        const float* d = &acc[ng * 4];
        const int r = off_m + row0, cc = off_n + ng * 8 + col0;
        c[(r + 0) * N + cc + 0] = __float2bfloat16(d[0]);
        c[(r + 0) * N + cc + 1] = __float2bfloat16(d[1]);
        c[(r + 8) * N + cc + 0] = __float2bfloat16(d[2]);
        c[(r + 8) * N + cc + 1] = __float2bfloat16(d[3]);
    }
#endif  // IK_WGMMA_DEVICE_OK
}

#if IK_WGMMA_HOST_OK
inline void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char* msg = nullptr;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unknown CUDA driver error";
    TORCH_CHECK(false, "[gemm_wgmma] ", msg);
}

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
#endif  // IK_WGMMA_HOST_OK

}  // namespace

// Host entry. `b` is [K,N] row-major (standard A@B). Requires bf16 + tile-aligned
// dims; the Python wrapper guarantees both and only reaches here on Hopper.
void gemm_wgmma(torch::Tensor out, torch::Tensor a, torch::Tensor b) {
#if IK_WGMMA_HOST_OK
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    TORCH_CHECK(prop.major == 9,
                "[gemm_wgmma] wgmma is a Hopper (sm_90) instruction; this GPU is sm_",
                prop.major, prop.minor,
                ". On Blackwell (sm_100) use gemm_tcgen05 instead.");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::BFloat16,
                "[gemm_wgmma] this educative kernel supports bf16 only");

    constexpr int BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 64;
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % BLOCK_K == 0,
                "[gemm_wgmma] dims must be tile-aligned (M%", BLOCK_M, ", N%",
                BLOCK_N, ", K%", BLOCK_K, "); got ", M, "x", N, "x", K);

    auto bt = b.t().contiguous();  // [N, K], K-contiguous (col-major B)

    CUtensorMap a_tmap, b_tmap;
    init_tmap(&a_tmap, static_cast<const __nv_bfloat16*>(a.data_ptr()), M, K, BLOCK_M);
    init_tmap(&b_tmap, static_cast<const __nv_bfloat16*>(bt.data_ptr()), N, K, BLOCK_N);

    const int grid = (M / BLOCK_M) * (N / BLOCK_N);
    const int smem = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__nv_bfloat16);
    auto kernel = gemm_wgmma_kernel<BLOCK_K>;
    if (smem > 48000)
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<grid, 128, smem>>>(
        a_tmap, b_tmap, static_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K);
    C10_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "[gemm_wgmma] requires CUDA toolkit >= 12.0 to compile wgmma");
#endif
}
