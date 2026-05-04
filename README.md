# inference-kernel

GPU inference kernels with `torch`, `triton`, and `cuda` backends side-by-side.

Python wrappers and native sources are split at the language level:

- `python/inference_kernel/kernels/<category>/<name>/` — three Python
  backends side-by-side: `torch_impl.py` (eager reference, correctness
  oracle), `triton_impl.py`, and `cuda_impl.py` (thin wrapper over the
  built `_ext`).
- `csrc/<category>/<name>/` — the C++/CUDA sources (`.cu`, `binding.cpp`).

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

1. `python/inference_kernel/kernels/<category>/<name>/`:
   - `torch_impl.py` (reference) + `triton_impl.py` + `cuda_impl.py`
   - `README.md` describing the math / shape contract
2. `csrc/<category>/<name>/`:
   - `<name>.cu` + `binding.cpp`
3. `tests/<category>/<name>/`: `test_torch.py`, `test_triton.py`, `test_cuda.py`
4. `benchmarks/<category>/bench_<name>.py`

`setup.py` auto-discovers `csrc/<category>/<name>/` folders and the JIT
loader maps the package name to the matching csrc dir by convention. No
central registry to update.
