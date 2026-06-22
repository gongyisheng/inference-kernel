# inference-kernel

GPU inference kernels with `torch`, `triton`, and `cuda` backends side-by-side.

Python wrappers and native sources are split at the language level, one
backend file per category:

- `python/inference_kernel/kernels/<category>/` — one file per backend:
  `torch_impl.py` (the correctness oracle + fused PyTorch ceiling),
  `triton_impl.py`, and `cuda_impl.py`. Each file holds every kernel for the
  category; a single file may expose several variants of one kernel as
  separate functions (e.g. gemm `cuda_impl.py` exports `gemm` and
  `gemm_naive`).
- `csrc/<category>/` — C++/CUDA sources, optionally grouped into `naive/` and
  `opt/` subfolders, with one shared `binding.cpp` at the category root. One
  compiled extension per category registers all of the category's kernels.

Backends are imported explicitly; there is no auto-dispatch.

## Install

```bash
uv venv
uv pip install -e ".[dev]"
```

CUDA backends compile JIT on first use (cached under `~/.cache/torch_extensions`).
For an AOT install with prebuilt extensions: `uv pip install .` (or `pip install .`).

## Use

```python
from inference_kernel.kernels.activation.torch_impl  import silu as silu_torch   # reference
from inference_kernel.kernels.activation.triton_impl import silu as silu_triton
from inference_kernel.kernels.activation.cuda_impl   import silu as silu_cuda
```

## Test

```bash
uv run pytest tests/                       # all
uv run pytest tests/ --device cuda:1       # specific GPU
```

CUDA / Triton tests skip cleanly when no GPU is available.

## Benchmark

```bash
uv run python -m benchmarks.activation.bench_silu --device cuda:0
uv run python scripts/run_all_benches.py --device cuda:0
```

CSV output goes to `benchmarks/results/`.

## Adding a kernel

1. Add the function to the category's `triton_impl.py` / `cuda_impl.py` under
   `python/inference_kernel/kernels/<category>/`. The `torch_impl.py`
   reference is written once per kernel and is the oracle every backend is
   tested against. For a second variant of an existing kernel, expose it as a
   distinct function in the same file (e.g. `gemm` vs `gemm_naive`).
2. `csrc/<category>/`: drop a `<name>.cu` (in a `naive/` or `opt/` subfolder
   if you keep that split) and declare its forward in the category's shared
   `csrc/<category>/binding.cpp`. Use distinct symbol names (e.g. `gemm_opt`)
   so variants don't collide. Point the category's `cuda_impl.py`
   `sources=[...]` at the new `.cu`.
3. `tests/<category>/test_{torch,triton,cuda}.py`: validate the new kernel
   against `torch_impl` — the same oracle and tolerances every backend is
   held to.
4. `benchmarks/<category>/bench_<name>.py`: the benchmark overlays every
   backend that implements the kernel on one chart.

`setup.py` auto-discovers `csrc/<category>/` folders and the JIT loader maps
the package name to the matching csrc dir by convention. No central registry
to update.
