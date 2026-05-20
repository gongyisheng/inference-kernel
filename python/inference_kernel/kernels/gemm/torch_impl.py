import torch

def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b