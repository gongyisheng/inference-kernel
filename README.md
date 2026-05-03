# inference-kernel

GPU inference kernels with `torch`, `triton`, and `cuda` backends side-by-side.

Each kernel lives under `src/inference_kernel/kernels/<category>/<name>/`
with three implementations:

- `torch_impl.py` — eager reference (correctness oracle for the others)
- `triton_impl.py` — Triton block kernel
- `cuda_impl.py` — custom CUDA kernel (+ `csrc/`)

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
from inference_kernel.kernels.activation.silu.torch_impl  import silu as silu_torch
from inference_kernel.kernels.activation.silu.triton_impl import silu as silu_triton
from inference_kernel.kernels.activation.silu.cuda_impl   import silu as silu_cuda
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

1. `src/inference_kernel/kernels/<category>/<name>/`:
   - `torch_impl.py` (reference) + `triton_impl.py` + `cuda_impl.py`
   - `csrc/<name>.cu` + `csrc/binding.cpp`
   - `README.md` describing the math / shape contract
2. `tests/<category>/<name>/`: `test_torch.py`, `test_triton.py`, `test_cuda.py`
3. `benchmarks/<category>/bench_<name>.py`

`setup.py` auto-discovers `csrc/` folders; no central registry to update.
