cmake_minimum_required(VERSION 3.26)

# Torch ships CMake scripts that still declare old minimums; allow them under
# CMake 4.x.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(aot_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

# CUTLASS: header-only. Fetch a pinned release and use only its include tree
# (Populate, not MakeAvailable, so we skip building CUTLASS's own tests/tools).
include(FetchContent)
FetchContent_Declare(cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v4.5.3
  GIT_SHALLOW TRUE)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()

# Locate the Torch CMake package shipped with the installed torch wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch.utils;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# Torch's required C++ ABI flags (e.g. _GLIBCXX_USE_CXX11_ABI).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Default to this machine's arch; override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

file(GLOB_RECURSE AOT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/aot_kernel/csrc/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/aot_kernel/csrc/*.cc")

python_add_library(_C MODULE ${AOT_SOURCES} WITH_SOABI)
target_include_directories(_C PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/aot_kernel/csrc/include"
  "${cutlass_SOURCE_DIR}/include")
target_link_libraries(_C PRIVATE "${TORCH_LIBRARIES}" Python::Module)
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)
target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# Installed next to the python sources so `import aot_kernel._C` resolves.
install(TARGETS _C LIBRARY DESTINATION aot_kernel)
