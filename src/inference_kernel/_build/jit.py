"""CUDA extension loader: AOT first, JIT fallback.

Each kernel's cuda_impl.py calls load_kernel(...) at module import time.
If the package was installed via `pip install .` (or `pip install -e .`),
setup.py built and placed an AOT extension named
`inference_kernel.kernels.<category>.<name>._ext`; we import it directly.

Otherwise we JIT-compile sources from the kernel's csrc/ directory using
torch.utils.cpp_extension.load. The compiled object is cached under
~/.cache/torch_extensions/ keyed by source contents, so subsequent
imports are fast.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType


def load_kernel(
    *,
    package: str,
    csrc_dir: Path,
    sources: list[str],
    extra_cuda_cflags: list[str] | None = None,
    extra_cflags: list[str] | None = None,
) -> ModuleType:
    """Return the compiled extension module for a kernel.

    Args:
        package: dotted name of the kernel package, e.g.
            "inference_kernel.kernels.activation.silu". The AOT extension
            is expected at f"{package}._ext".
        csrc_dir: absolute path to the kernel's csrc/ directory (for JIT).
        sources: filenames inside csrc_dir, e.g. ["silu.cu", "binding.cpp"].
        extra_cuda_cflags: extra nvcc flags for JIT mode.
        extra_cflags: extra C++ flags for JIT mode.

    Returns:
        Module exposing the kernel's C++ functions (e.g. ext.silu_forward).
    """
    aot_name = f"{package}._ext"
    try:
        return importlib.import_module(aot_name)
    except ModuleNotFoundError:
        pass

    # JIT fallback. Imported lazily so import doesn't pay the cost
    # unless we actually need to compile.
    from torch.utils.cpp_extension import load

    full_sources = [str(csrc_dir / s) for s in sources]
    # Unique JIT module name: replace dots with underscores so the cache
    # key doesn't collide across kernels.
    jit_name = package.replace(".", "_") + "_ext"
    return load(
        name=jit_name,
        sources=full_sources,
        extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math"],
        extra_cflags=extra_cflags or ["-O3", "-std=c++17"],
        verbose=False,
    )
