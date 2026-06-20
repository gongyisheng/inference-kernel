import math

import torch

def attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False
):
    head_dim = Q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    Q32, K32, V32 = Q.float(), K.float(), V.float()
    score = (Q32 @ K32.transpose(-2, -1)) * scale

    if is_causal:
        N = score.shape[-1]
        mask = torch.ones(N, N, dtype=torch.bool, device=score.device).tril()
        score = score.masked_fill(~mask, float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            score = score.masked_fill(~attn_mask, float("-inf"))
        else:
            score = score + attn_mask.float()

    out = torch.softmax(score, dim=-1) @ V32
    return out.to(Q.dtype)
