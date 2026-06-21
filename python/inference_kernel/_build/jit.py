"""CUDA extension loader: AOT first, JIT fallback.

Each category's cuda_impl.py calls load_kernel(package=..., sources=...) at
module import time.

If the package was installed via `pip install .` (or `pip install -e .`),
setup.py built and placed an AOT extension named
`inference_kernel.kernels.<category>._ext`; we import it directly.

Otherwise we JIT-compile sources from the category's csrc/<category>/
directory at the repo root, using torch.utils.cpp_extension.load. The
compiled object is cached under ~/.cache/torch_extensions/, keyed by source
contents, so subsequent imports are fast.

The csrc directory is derived from the package name by convention:
    inference_kernel.kernels.<category>  →  <repo_root>/csrc/<category>/
"""

import functools
import importlib
from pathlib import Path
from types import ModuleType

_PACKAGE_PREFIX = "inference_kernel.kernels."


@functools.lru_cache(maxsize=1)
def _repo_root() -> Path:
    """Find the repo root by walking up to the nearest pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "could not locate repo root (no pyproject.toml found above "
        f"{Path(__file__)})"
    )


def _csrc_dir_for(package: str) -> Path:
    """Map inference_kernel.kernels.<cat> → <repo_root>/csrc/<cat>."""
    if not package.startswith(_PACKAGE_PREFIX):
        raise ValueError(
            f"package must start with {_PACKAGE_PREFIX!r}, got {package!r}"
        )
    rest = package[len(_PACKAGE_PREFIX):].split(".")
    if len(rest) != 1:
        raise ValueError(
            f"expected package=inference_kernel.kernels.<category>, got {package!r}"
        )
    (category,) = rest
    return _repo_root() / "csrc" / category


def load_kernel(
    *,
    package: str,
    sources: list[str],
    extra_cuda_cflags: list[str] | None = None,
    extra_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
) -> ModuleType:
    """Return the compiled extension module for a kernel.

    Args:
        package: dotted name of the category package, e.g.
            "inference_kernel.kernels.activation". The AOT extension
            is expected at f"{package}._ext"; the JIT fallback compiles
            from <repo_root>/csrc/<category>/.
        sources: filenames inside the csrc dir, e.g. ["silu.cu", "binding.cpp"].
        extra_cuda_cflags: extra nvcc flags for JIT mode.
        extra_cflags: extra C++ flags for JIT mode.
        extra_ldflags: extra linker flags for JIT mode (e.g. ["-lcuda"] for
            kernels that call the CUDA driver API such as cuTensorMapEncodeTiled).
    """
    aot_name = f"{package}._ext"
    try:
        return importlib.import_module(aot_name)
    except ModuleNotFoundError:
        pass

    # JIT fallback. Lazy torch import keeps the AOT path fast.
    from torch.utils.cpp_extension import load

    csrc_dir = _csrc_dir_for(package)
    if not csrc_dir.is_dir():
        raise FileNotFoundError(
            f"csrc directory not found at {csrc_dir}; cannot JIT-compile {package}. "
            "If installed from a wheel, the AOT extension should have been built — "
            "check that setup.py was run during install."
        )
    full_sources = [str(csrc_dir / s) for s in sources]
    jit_name = package.replace(".", "_") + "_ext"
    return load(
        name=jit_name,
        sources=full_sources,
        extra_include_paths=[str(csrc_dir.parent / "include")],
        extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math"],
        extra_cflags=extra_cflags or ["-O3", "-std=c++17"],
        extra_ldflags=extra_ldflags,
        verbose=False,
    )
