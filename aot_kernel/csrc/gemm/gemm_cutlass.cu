#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

// Two ops, all row-major, fp32 accumulate: gemm_cutlass (out = a @ b) and
// gemm_cutlass_fused_act (out = act(a @ b), act = ReLU/SiLU). They share one
// dispatch + templated runner; only the epilogue functor differs.
//
// CUTLASS 2.x device::Gemm (not the 3.x CuTe CollectiveBuilder): on GeForce
// Blackwell (sm_120) the 3.x builders are fp8/fp6/fp4 only, so 16-bit GEMM goes
// through the SM80 tensor-op path, which sm_120 runs via backward compatibility.
// The fp8/fp4 Sm120-native CuTe kernel is a separate op to be added later.
//
// Activation is fused into the epilogue functor (LinearCombination[Relu/Silu]).
// With beta=0 the epilogue computes act(alpha*acc) = act(a@b) and never reads C.
// It applies on every path (tensor-op, SIMT, fp32) so behaviour is independent
// of shape/dtype.
//
// device::Gemm supports row-major C directly, so unlike cuBLASLt there is no
// operand-swap trick. Shape/device/dtype validation lives in gemm.py; out is
// allocated there, so we trust the inputs here.

namespace {

enum class Act { None = 0, ReLU = 1, SiLU = 2 };

// Epilogue functor for a given (activation, element, vector width). All three
// share the <ElementOutput, Count, ElementAccumulator, ElementCompute> signature
// and take {alpha, beta} params, so run() is identical across them.
template <Act A, typename Element, int Count> struct EpiSel;
template <typename Element, int Count>
struct EpiSel<Act::None, Element, Count> {
    using type = cutlass::epilogue::thread::LinearCombination<Element, Count, float, float>;
};
template <typename Element, int Count>
struct EpiSel<Act::ReLU, Element, Count> {
    using type = cutlass::epilogue::thread::LinearCombinationRelu<Element, Count, float, float>;
};
template <typename Element, int Count>
struct EpiSel<Act::SiLU, Element, Count> {
    using type = cutlass::epilogue::thread::LinearCombinationSilu<Element, Count, float, float>;
};

// SM80 tensor-core GEMM (align 8), fp32 accumulate. Tile 128x128x32 / 3 stages
// = ~48 KiB smem: the default SM80 config (128x256x64) needs ~144 KiB, which
// exceeds sm_120's 100 KiB/block limit and fails to launch.
template <typename Element, Act A>
using TensorOpGemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    typename EpiSel<A, Element, 128 / cutlass::sizeof_bits<Element>::value>::type,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

// SIMT GEMM (align 1): any shape, fp32 accumulate. Used for K/N not divisible
// by 8 and for fp32 (exact, no tensor cores). Tile/warp/stages come from the
// SIMT default config; only the epilogue is swapped for the activation (Count=1).
template <typename Element>
using SimtConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80, Element, Element, Element, float>;

template <typename Element, Act A>
using SimtGemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    Element, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    typename SimtConfig<Element>::ThreadblockShape,
    typename SimtConfig<Element>::WarpShape,
    typename SimtConfig<Element>::InstructionShape,
    typename EpiSel<A, Element, 1>::type,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    SimtConfig<Element>::kStages>;

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
        {1.0f, 0.0f});                                          // alpha, beta: D = act(alpha*(A@B) + beta*C)

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

// Bridge the runtime activation code to the compile-time Act template parameter.
template <template <typename, Act> class Gemm, typename Element>
void dispatch_act(torch::Tensor out, torch::Tensor a, torch::Tensor b,
                  int M, int N, int K, int64_t activation) {
    switch (activation) {
        case 0: run<Gemm<Element, Act::None>>(out, a, b, M, N, K); break;
        case 1: run<Gemm<Element, Act::ReLU>>(out, a, b, M, N, K); break;
        case 2: run<Gemm<Element, Act::SiLU>>(out, a, b, M, N, K); break;
        default: TORCH_CHECK(false, "gemm_cutlass: bad activation code ", activation);
    }
}

// Shared dtype/alignment dispatch for both the plain and fused entry points.
void run_gemm(torch::Tensor out, torch::Tensor a, torch::Tensor b, int64_t activation) {
    const int M = a.size(0), K = a.size(1), N = b.size(1);
    const bool aligned = (K % 8 == 0) && (N % 8 == 0);  // 128-bit vector loads for tensor op

    switch (a.scalar_type()) {
        case at::kHalf:
            if (aligned) dispatch_act<TensorOpGemm, cutlass::half_t>(out, a, b, M, N, K, activation);
            else         dispatch_act<SimtGemm, cutlass::half_t>(out, a, b, M, N, K, activation);
            break;
        case at::kBFloat16:
            if (aligned) dispatch_act<TensorOpGemm, cutlass::bfloat16_t>(out, a, b, M, N, K, activation);
            else         dispatch_act<SimtGemm, cutlass::bfloat16_t>(out, a, b, M, N, K, activation);
            break;
        case at::kFloat:
            dispatch_act<SimtGemm, float>(out, a, b, M, N, K, activation);
            break;
        default:
            TORCH_CHECK(false, "gemm_cutlass: unsupported dtype ", a.scalar_type());
    }
}

}  // namespace

// Plain out = a @ b.
void gemm_cutlass(torch::Tensor out, torch::Tensor a, torch::Tensor b) {
    run_gemm(out, a, b, static_cast<int64_t>(Act::None));
}

// Fused out = act(a @ b); activation is 1=ReLU, 2=SiLU (validated in gemm.py).
void gemm_cutlass_fused_act(torch::Tensor out, torch::Tensor a, torch::Tensor b, int64_t activation) {
    run_gemm(out, a, b, activation);
}
