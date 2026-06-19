# inference-kernel

GPU inference kernels with `torch`, `triton`, and `cuda` backends side-by-side.

Python wrappers and native sources are split at the language level, one
backend file per category:

- `python/inference_kernel/kernels/<category>/` — kernels grouped into
  tier subfolders: `reference/` (`eager_impl.py` slow correctness oracle +
  `torch_impl.py` fused PyTorch ceiling), `naive/` (readable teaching
  `triton_impl.py` / `cuda_impl.py`), and `opt/` (production-grade
  hand-written kernels). Each tier file holds every kernel for the category.
- `csrc/<category>/` — C++/CUDA sources in matching `naive/` and `opt/`
  subfolders, with one shared `binding.cpp` at the category root. One
  compiled extension per category registers all tiers' kernels.

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
from inference_kernel.kernels.activation.reference.eager_impl  import silu as silu_eager   # slow oracle
from inference_kernel.kernels.activation.reference.torch_impl  import silu as silu_torch   # F.silu
from inference_kernel.kernels.activation.naive.triton_impl     import silu as silu_triton
from inference_kernel.kernels.activation.naive.cuda_impl       import silu as silu_cuda
# from inference_kernel.kernels.activation.opt.cuda_impl       import silu as silu_cuda_opt  # when an opt kernel exists
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

1. Pick a tier. Educational/first version → `naive/`. Production version →
   `opt/`. Add the function to that tier's `triton_impl.py` / `cuda_impl.py`
   under `python/inference_kernel/kernels/<category>/<tier>/`. Reference
   impls (`reference/eager_impl.py`, `reference/torch_impl.py`) are written
   once per kernel and shared by every tier.
2. `csrc/<category>/<tier>/`: drop a `<name>.cu` and declare its forward in
   the category's shared `csrc/<category>/binding.cpp`. Use an `_opt` suffix
   for opt symbols (e.g. `silu_opt_forward`) so they don't collide with the
   naive symbols. Point the tier's `cuda_impl.py` `sources=[...]` at the new
   `<tier>/<name>.cu`.
3. `tests/<category>/test_{torch,triton,cuda}.py`: validate the new kernel
   against `reference/eager_impl` — the same oracle and tolerances every
   tier is held to.
4. `benchmarks/<category>/bench_<name>.py`: the benchmark overlays every
   tier that implements the kernel on one chart (opt rows appear
   automatically once the opt import resolves).

`setup.py` auto-discovers `csrc/<category>/` folders and the JIT loader
maps the package name to the matching csrc dir by convention. No central
registry to update.
