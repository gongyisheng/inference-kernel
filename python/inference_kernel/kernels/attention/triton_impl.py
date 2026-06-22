import triton
import triton.language as tl


# @triton.jit
# def flash_attention(
#     q_ptr,
#     k_ptr,
#     v_ptr,
#     q_stride,
#     k_stride,
#     v_stride,
#     BLOCK_SIZE: tl.constexpr
# ):
#     Q = tl.load(q_ptr + )