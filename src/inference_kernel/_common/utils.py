"""Shared utilities for kernel implementations.

Currently exposes minimal helpers; expand as kernels reveal common needs.
"""
from __future__ import annotations

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


def assert_contiguous(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not t.is_contiguous():
            raise ValueError("tensor must be contiguous")
