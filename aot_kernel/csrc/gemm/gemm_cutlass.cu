#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

// out = a @ b, all row-major, fp32 accumulate.
//
// CUTLASS 2.x device::Gemm (not the 3.x CuTe CollectiveBuilder): on GeForce
// Blackwell (sm_120) the 3.x builders are fp8/fp6/fp4 only, so 16-bit GEMM goes
// through the SM80 tensor-op path, which sm_120 runs via backward compatibility.
// The fp8/fp4 Sm120-native CuTe kernel is a separate op to be added later.
//
// device::Gemm supports row-major C directly, so unlike cuBLASLt there is no
// operand-swap trick. Shape/device/dtype validation lives in gemm.py; out is
// allocated there, so we trust the inputs here.

namespace {

// SM80 tensor-core GEMM (align 8), fp32 accumulate. Tile 128x128x32 / 3 stages
// = ~48 KiB smem: the default SM80 config (128x256x64) needs ~144 KiB, which
// exceeds sm_120's 100 KiB/block limit and fails to launch.
template <typename Element>
using TensorOpGemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 128 / cutlass::sizeof_bits<Element>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

// SIMT GEMM (align 1): any shape, fp32 accumulate. Used for K/N not divisible
// by 8 and for fp32 (exact, no tensor cores).
template <typename Element>
using SimtGemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80>;

template <typename Gemm>
void run(torch::Tensor out, torch::Tensor a, torch::Tensor b, int M, int N, int K) {
    using ElementA = typename Gemm::ElementA;
    using ElementB = typename Gemm::ElementB;
    using ElementC = typename Gemm::ElementC;

    typename Gemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<const ElementA*>(a.data_ptr()), K},   // A row-major, lda=K
        {reinterpret_cast<const ElementB*>(b.data_ptr()), N},   // B row-major, ldb=N
        {reinterpret_cast<ElementC*>(out.data_ptr()), N},       // C row-major, ldc=N
        {reinterpret_cast<ElementC*>(out.data_ptr()), N},       // D == C
        {1.0f, 0.0f});                                          // alpha, beta

    Gemm op;
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
                "gemm_cutlass: problem unsupported by this config (M=", M, " N=", N, " K=", K, ")");

    auto workspace = torch::empty(
        {static_cast<int64_t>(Gemm::get_workspace_size(args))},
        torch::dtype(torch::kUInt8).device(a.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(op.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
                "gemm_cutlass: initialize failed");
    TORCH_CHECK(op(stream) == cutlass::Status::kSuccess, "gemm_cutlass: launch failed");
}

}  // namespace

void gemm_cutlass(torch::Tensor out, torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0), K = a.size(1), N = b.size(1);
    const bool aligned = (K % 8 == 0) && (N % 8 == 0);  // 128-bit vector loads for tensor op

    switch (a.scalar_type()) {
        case at::kHalf:
            if (aligned) run<TensorOpGemm<cutlass::half_t>>(out, a, b, M, N, K);
            else         run<SimtGemm<cutlass::half_t>>(out, a, b, M, N, K);
            break;
        case at::kBFloat16:
            if (aligned) run<TensorOpGemm<cutlass::bfloat16_t>>(out, a, b, M, N, K);
            else         run<SimtGemm<cutlass::bfloat16_t>>(out, a, b, M, N, K);
            break;
        case at::kFloat:
            run<SimtGemm<float>>(out, a, b, M, N, K);
            break;
        default:
            TORCH_CHECK(false, "gemm_cutlass: unsupported dtype ", a.scalar_type());
    }
}
