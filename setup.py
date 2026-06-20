"""Build CUDA extensions for inference_kernel.

JIT is the default: kernels compile lazily via load_kernel and are cached
under ~/.cache/torch_extensions/, auto-recompiling when a .cu changes. So a
normal `pip`/`uv` install does NOT compile anything — it stays instant.

AOT compilation is opt-in via the IK_BUILD_EXT=1 env var (e.g. for a prod
wheel). When set, we auto-discover csrc/<category>/ and register one
extension per category, named `inference_kernel.kernels.<category>._ext`,
which the category's cuda_impl.py imports (else it falls back to JIT):
    IK_BUILD_EXT=1 python setup.py build_ext --inplace

All other packaging metadata lives in pyproject.toml.
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
CSRC_ROOT = ROOT / "csrc"


def discover_extensions() -> list[CUDAExtension]:
    """Find every csrc/<category>/ and build it as one extension."""
    extensions: list[CUDAExtension] = []
    for category_dir in sorted(CSRC_ROOT.glob("*")):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        sources = sorted(
            p.relative_to(ROOT).as_posix()
            for p in category_dir.rglob("*")
            if p.is_file() and p.suffix in {".cu", ".cpp", ".cc"}
        )
        if not sources:
            continue
        ext_name = f"inference_kernel.kernels.{category}._ext"
        extensions.append(
            CUDAExtension(
                name=ext_name,
                sources=sources,
                include_dirs=[str(CSRC_ROOT / "include")],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            )
        )
    return extensions


if __name__ == "__main__":
    build_aot = os.environ.get("IK_BUILD_EXT") == "1"
    setup(
        ext_modules=discover_extensions() if build_aot else [],
        cmdclass={"build_ext": BuildExtension},
    )
