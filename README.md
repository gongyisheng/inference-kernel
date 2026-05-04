# inference-kernel

GPU inference kernels with `torch`, `triton`, and `cuda` backends side-by-side.

Python wrappers and native sources are split at the language level, one
backend file per category:

- `python/inference_kernel/kernels/<category>/` — Python backends
  side-by-side: `eager_impl.py` (slow correctness oracle), `torch_impl.py`
  (fast torch using fused PyTorch ops), `triton_impl.py`, and `cuda_impl.py`
  (thin wrapper over the built `_ext`). Each backend file holds every
  kernel function for the category.
- `csrc/<category>/` — the C++/CUDA sources (`.cu`, `binding.cpp`); one
  compiled extension per category, registering all the category's kernels.

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
from inference_kernel.kernels.activation.eager_impl  import silu as silu_eager   # slow oracle
from inference_kernel.kernels.activation.torch_impl  import silu as silu_torch   # F.silu
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

1. `python/inference_kernel/kernels/<category>/`: add a function to each
   of `eager_impl.py`, `torch_impl.py`, `triton_impl.py`, `cuda_impl.py`,
   and document the new kernel in the category's `README.md`.
2. `csrc/<category>/`: drop a `<name>.cu` and add the function to the
   shared `binding.cpp` so it joins the per-category extension.
3. `tests/<category>/test_{torch,triton,cuda}.py`: add tests alongside
   the existing kernels in each backend file.
4. `benchmarks/<category>/bench_<name>.py`: one benchmark per kernel.

`setup.py` auto-discovers `csrc/<category>/` folders and the JIT loader
maps the package name to the matching csrc dir by convention. No central
registry to update.
