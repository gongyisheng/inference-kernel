"""Build CUDA extensions for inference_kernel.

All other packaging metadata lives in pyproject.toml. This file exists
solely because torch.utils.cpp_extension.CUDAExtension requires
distutils-style setup() to compile .cu sources AOT.

Each category drops its sources at csrc/<category>/. We auto-discover
those and register one extension per category. The extension module name
is `inference_kernel.kernels.<category>._ext`, and the category's
cuda_impl.py imports it (with a JIT fallback if not built).
"""

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
            for p in category_dir.iterdir()
            if p.suffix in {".cu", ".cpp", ".cc"}
        )
        if not sources:
            continue
        ext_name = f"inference_kernel.kernels.{category}._ext"
        extensions.append(
            CUDAExtension(
                name=ext_name,
                sources=sources,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            )
        )
    return extensions


if __name__ == "__main__":
    setup(
        ext_modules=discover_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )
