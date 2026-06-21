"""Shared utilities for kernel implementations.

Currently exposes minimal helpers; expand as kernels reveal common needs.
"""

import torch


def assert_same_device(*tensors: torch.Tensor) -> torch.device:
    """All tensors must share the same device. Returns that device."""
    if not tensors:
        raise ValueError("assert_same_device requires at least one tensor")
    device = tensors[0].device
    for t in tensors[1:]:
        if t.device != device:
            raise ValueError(f"device mismatch: {device} vs {t.device}")
    return device


def assert_same_dtype(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise ValueError("assert_same_dtype requires at least one tensor")
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        if t.dtype != dtype:
            raise ValueError(f"dtype mismatch: {dtype} vs {t.dtype}")
    return dtype


def assert_contiguous(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not t.is_contiguous():
            raise ValueError("tensor must be contiguous")


def assert_is_cuda(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not t.is_cuda:
            raise RuntimeError("tensor must be cuda tensor")


def cuda_capability(device: torch.device | None = None) -> tuple[int, int]:
    """(major, minor) compute capability of `device` (current CUDA device if None).

    Compare as a tuple, e.g. ``cuda_capability() >= (9, 0)`` for Hopper+,
    ``>= (10, 0)`` for Blackwell+.
    """
    return torch.cuda.get_device_capability(device)
