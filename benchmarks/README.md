# Benchmarks

Standalone scripts (one per kernel) that sweep shapes, dtypes, and backends
and write CSV to `results/`.

## Run a single benchmark

```bash
uv run python -m benchmarks.activation.bench_silu --device cuda
```

## Run all benchmarks

```bash
uv run python scripts/run_all_benches.py --device cuda
```

## CSV schema

`kernel, backend, shape, dtype, device, ms, tflops, git_sha`

`results/*.csv` is gitignored. Commit a snapshot manually if you want to
track regressions over time.
