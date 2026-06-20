#pragma once

#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#define _IK_DISPATCH_CASE(enum_t, cpp_t, c_type, ...) \
  case at::ScalarType::enum_t: {                      \
    using c_type = cpp_t;                             \
    return __VA_ARGS__();                             \
  }

#define DISPATCH_FLOATING_TYPES(scalar_type, c_type, ...)             \
  [&]() -> bool {                                                     \
    switch (scalar_type) {                                            \
      _IK_DISPATCH_CASE(Float, float, c_type, __VA_ARGS__)            \
      _IK_DISPATCH_CASE(Half, __half, c_type, __VA_ARGS__)            \
      _IK_DISPATCH_CASE(BFloat16, __nv_bfloat16, c_type, __VA_ARGS__) \
      default:                                                        \
        return false;                                                 \
    }                                                                 \
  }()
