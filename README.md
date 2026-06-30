# inference-kernel

GPU inference kernels, split into three importable packages by backend so they
can be reused independently in other projects. All three expose the **same
operation names and signatures**, so they are drop-in swappable:

- **`ref`** — torch/eager reference implementations; the correctness oracle.
- **`jit_kernel`** — Triton kernels, compiled at runtime (first call).
- **`aot_kernel`** — CUDA/C++ kernels, ahead-of-time compiled via CMake into a
  prebuilt `_C` extension (no nvcc needed at import). Ops are registered under
  the `torch.ops.aot_kernel` namespace.

```
ref/         <op>.py per category            # torch oracle
jit_kernel/  <op>.py per category            # triton
aot_kernel/  <op>.py + csrc/<category>/*.cu  # cuda; csrc/registration.cc binds all ops
```

`attention` and `math` exist only in `ref`/`jit_kernel` for now (no `.cu` yet).
The few validation helpers are duplicated into each package's `_utils.py`.

## Install

```bash
uv venv
uv pip install -e ".[dev]" --no-build-isolation
```

`--no-build-isolation` builds `aot_kernel._C` against the torch already in the
venv (avoids re-downloading the CUDA torch wheel and ABI mismatches). The CUDA
extension is compiled once at install time via CMake (scikit-build-core). Set
the target arch with `-DCMAKE_CUDA_ARCHITECTURES=...` if the default (`120`,
Blackwell) doesn't match your GPU — e.g. `80` (A100), `89` (L4/4090), `90`
(H100). Pass it through scikit-build-core:

```bash
uv pip install -e . --no-build-isolation \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

## Use

```python
from ref import silu          # torch oracle
from jit_kernel import silu   # triton
from aot_kernel import silu   # cuda (prebuilt)
```

## Test

```bash
uv run pytest                          # current CUDA device
uv run pytest --device cuda:0          # specific GPU
```

Note: Triton launches against the *current* CUDA device, so run on the device
that is current (default `cuda` = current device). Pinning `--device cuda:1`
while device 0 is current makes Triton reject the inputs. CUDA / Triton tests
skip cleanly when no GPU is available.

## Benchmark

```bash
uv run python -m benchmarks.activation.bench_silu --device cuda:0
uv run python scripts/run_all_benches.py --device cuda:0
```

CSV + PNG output goes to `benchmarks/results/`.

## Adding a kernel

1. Reference first: add the op to `ref/<category>.py` — the oracle every
   backend is tested against.
2. Triton: add it to `jit_kernel/<category>.py`, same function name/signature.
3. CUDA:
   - Drop `aot_kernel/csrc/<category>/<name>.cu`.
   - Declare its forward in `aot_kernel/csrc/include/ops.h` and add a
     `m.def` / `m.impl` line to `aot_kernel/csrc/registration.cc` (one
     `TORCH_LIBRARY(aot_kernel, ...)` block binds every op). Use distinct
     symbol names for variants (e.g. `gemm_opt`).
   - Add a thin wrapper in `aot_kernel/<category>.py` calling
     `torch.ops.aot_kernel.<op>`.
   - Rebuild: `uv pip install -e . --no-build-isolation`. CMake globs
     `aot_kernel/csrc/**/*.cu` automatically — no source list to update.
4. `tests/<category>/test_{triton,cuda}.py`: validate against `ref`.
5. `benchmarks/<category>/bench_<name>.py`: overlays every backend on one chart.
```
