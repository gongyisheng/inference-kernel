#!/usr/bin/env bash
# Profile a single GEMM kernel launch with Nsight Compute (ncu).
#
# Usage:
#   profile/gemm/profile.sh [BACKEND] [M] [N] [K] [DTYPE]
#
# Examples:
#   profile/gemm/profile.sh cuda 1024 1024 1024 fp16
#   profile/gemm/profile.sh triton 2048 2048 2048 bf16
#   BACKEND=cuda M=4096 N=4096 K=4096 DTYPE=fp16 profile/gemm/profile.sh
#
# Output: profile/reports/gemm/<backend>_<M>x<N>x<K>_<dtype>.ncu-rep
# Open with:  ncu-ui profile/reports/gemm/<file>.ncu-rep
set -euo pipefail

BACKEND="${1:-${BACKEND:-cuda}}"
M="${2:-${M:-1024}}"
N="${3:-${N:-1024}}"
K="${4:-${K:-1024}}"
DTYPE="${5:-${DTYPE:-fp16}}"

# Tunables (override via env):
WARMUP="${WARMUP:-5}"
TRITON_IMPL="${TRITON_IMPL:-tile}"
KERNEL_REGEX="${KERNEL_REGEX:-gemm}"      # ncu kernel-name filter
NCU_SET="${NCU_SET:-full}"                # full | detailed | roofline | basic
DEVICE="${DEVICE:-cuda:0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="$REPO_ROOT/profile/reports/gemm"
mkdir -p "$REPORT_DIR"

REPORT_NAME="${BACKEND}_${M}x${N}x${K}_${DTYPE}"
REPORT_PATH="$REPORT_DIR/$REPORT_NAME"

NCU="${NCU:-ncu}"
command -v "$NCU" >/dev/null 2>&1 || {
  echo "error: '$NCU' not found on PATH. Install Nsight Compute or set NCU=/path/to/ncu." >&2
  exit 1
}

echo "[profile_gemm] backend=$BACKEND  shape=${M}x${N}x${K}  dtype=$DTYPE  set=$NCU_SET"
echo "[profile_gemm] report -> ${REPORT_PATH}.ncu-rep"

# --launch-skip skips the warmup launches; --launch-count limits the captured ones.
# --replay-mode kernel: ncu replays the targeted kernel to gather all metrics.
"$NCU" \
  --set "$NCU_SET" \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name-base mangled \
  -k "regex:$KERNEL_REGEX" \
  --launch-skip "$WARMUP" \
  --launch-count 1 \
  -f \
  -o "$REPORT_PATH" \
  -- \
  uv run python "$REPO_ROOT/profile/gemm/_run.py" \
    --backend "$BACKEND" \
    --triton-impl "$TRITON_IMPL" \
    --m "$M" --n "$N" --k "$K" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --warmup "$WARMUP" \
    --iters 1

echo "[profile_gemm] done. View with:  ncu-ui ${REPORT_PATH}.ncu-rep"
echo "[profile_gemm] or CLI summary:   ncu --import ${REPORT_PATH}.ncu-rep --page details"
