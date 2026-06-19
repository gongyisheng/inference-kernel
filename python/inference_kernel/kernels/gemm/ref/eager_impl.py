import torch

def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)
    out = (a_fp32.unsqueeze(1) * b_fp32.t().unsqueeze(0)).sum(dim=-1)
    return out.to(a.dtype)