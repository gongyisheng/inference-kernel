#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Scalar float<->native conversions. torch compiles nvcc with
// __CUDA_NO_HALF_CONVERSIONS__, so __half/__nv_bfloat16 have no static_cast to
// or from float; use the intrinsics instead. This lets a dtype-generic kernel
// load as c_type, compute in float, and store back as c_type.
__device__ inline float to_float(float v)         { return v; }
__device__ inline float to_float(__half v)        { return __half2float(v); }
__device__ inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ inline T from_float(float v);
template <> __device__ inline float         from_float<float>(float v)         { return v; }
template <> __device__ inline __half        from_float<__half>(float v)        { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
