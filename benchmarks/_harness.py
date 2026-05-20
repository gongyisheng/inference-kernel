"""Shared utilities for kernel benchmarks.

Each bench_*.py script defines (a) a list of shapes, (b) a dict of
backend name -> callable taking a tensor, (c) optional per-backend
"available" predicates. Then it calls run_bench(...) which times every
combination and writes a CSV row per measurement.

Timing prefers triton.testing.do_bench (handles warmup + CUDA events)
and falls back to torch.utils.benchmark.Timer for CPU paths.
"""

import csv
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class BenchRow:
    kernel: str
    backend: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    ms: float
    tflops: float | None
    git_sha: str


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _time_gpu(fn: Callable[[], torch.Tensor]) -> float:
    """Median ms via triton.testing.do_bench."""
    import triton.testing  # type: ignore

    return float(triton.testing.do_bench(fn, warmup=25, rep=100))


def _time_cpu(fn: Callable[[], torch.Tensor]) -> float:
    """Median ms via torch.utils.benchmark."""
    from torch.utils.benchmark import Timer

    t = Timer(stmt="fn()", globals={"fn": fn})
    measurement = t.blocked_autorange(min_run_time=1.0)
    return measurement.median * 1000.0


def time_callable(fn: Callable[[], torch.Tensor], device: torch.device) -> float:
    return _time_gpu(fn) if device.type == "cuda" else _time_cpu(fn)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def plot_results(
    rows: list[BenchRow],
    output_png: Path,
    x_axis: Callable[[tuple[int, ...]], int] = _numel,
    x_label: str = "numel",
) -> None:
    """Plot ms vs. x_axis(shape), one subplot per dtype, one line per backend.

    Skipped (no-op) if rows is empty.
    """
    if not rows:
        return
    import matplotlib.pyplot as plt  # lazy import — keeps non-plotting paths fast

    dtypes = sorted({r.dtype for r in rows})
    backends = sorted({r.backend for r in rows})
    kernel = rows[0].kernel
    device = rows[0].device

    fig, axes = plt.subplots(1, len(dtypes), figsize=(5 * len(dtypes), 4), squeeze=False)
    for i, dtype in enumerate(dtypes):
        ax = axes[0, i]
        for backend in backends:
            series = sorted(
                (r for r in rows if r.dtype == dtype and r.backend == backend),
                key=lambda r: x_axis(r.shape),
            )
            if not series:
                continue
            xs = [x_axis(r.shape) for r in series]
            ys = [r.ms for r in series]
            ax.plot(xs, ys, marker="o", label=backend)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("ms (median)")
        ax.set_title(f"dtype={dtype}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(f"{kernel} — backend comparison ({device})")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=120)
    plt.close(fig)


def run_bench(
    *,
    kernel: str,
    backends: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    make_input: Callable[[tuple[int, ...], torch.dtype, torch.device], torch.Tensor],
    shapes: list[tuple[int, ...]],
    dtypes: list[torch.dtype],
    device: torch.device,
    flops_per_element: float | None = None,
    output_csv: Path | None = None,
    output_png: Path | None = None,
    x_axis: Callable[[tuple[int, ...]], int] = _numel,
    x_label: str = "numel",
) -> list[BenchRow]:
    """Sweep (shape, dtype, backend) and write a CSV + PNG plot.

    Args:
        kernel: kernel name (e.g. "silu") — written to the CSV.
        backends: mapping of backend name -> callable. Skipped silently
            if the callable raises ImportError or RuntimeError on its
            first invocation (e.g. triton/cuda on a CPU-only host).
        make_input: builds an input tensor for a given (shape, dtype, device).
        flops_per_element: if provided, used to compute TFLOPS; else None.
        output_csv: defaults to results/<kernel>.csv.
        output_png: defaults to results/<kernel>.png.
    """
    rows: list[BenchRow] = []
    sha = _git_sha()
    output_csv = output_csv or (RESULTS_DIR / f"{kernel}.csv")
    output_png = output_png or (RESULTS_DIR / f"{kernel}.png")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    for shape in shapes:
        for dtype in dtypes:
            x = make_input(shape, dtype, device)
            if x.device != device:
                raise RuntimeError(
                    f"make_input returned tensor on {x.device}, expected {device}"
                )
            for backend_name, fn in backends.items():
                try:
                    y = fn(x)  # warmup + availability check
                except (ImportError, RuntimeError, ValueError) as e:
                    print(f"  [skip] {backend_name} on {shape} {dtype}: {e}")
                    continue
                if isinstance(y, torch.Tensor) and y.device != device:
                    raise RuntimeError(
                        f"backend {backend_name!r} produced output on {y.device}, "
                        f"expected {device}"
                    )
                ms = time_callable(lambda f=fn, x=x: f(x), device)
                tflops = None
                if flops_per_element is not None:
                    tflops = (flops_per_element * _numel(shape)) / (ms * 1e-3) / 1e12
                row = BenchRow(
                    kernel=kernel,
                    backend=backend_name,
                    shape=shape,
                    dtype=str(dtype).replace("torch.", ""),
                    device=str(device),
                    ms=ms,
                    tflops=tflops,
                    git_sha=sha,
                )
                rows.append(row)
                tflops_str = f"{tflops:.2f}" if tflops is not None else ""
                print(
                    f"  {kernel:10s} {backend_name:14s} shape={shape!s:20s} "
                    f"dtype={row.dtype:8s} ms={ms:8.4f} tflops={tflops_str}"
                )

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["kernel", "backend", "shape", "dtype", "device", "ms", "tflops", "git_sha"]
        )
        for r in rows:
            writer.writerow(
                [r.kernel, r.backend, str(r.shape), r.dtype, r.device, f"{r.ms:.6f}",
                 "" if r.tflops is None else f"{r.tflops:.6f}", r.git_sha]
            )
    print(f"wrote {len(rows)} rows to {output_csv}")

    plot_results(rows, output_png, x_axis=x_axis, x_label=x_label)
    print(f"wrote plot to {output_png}")

    return rows
