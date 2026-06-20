import torch
import torch.nn.functional as F


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    return F.scaled_dot_product_attention(
        Q, K, V, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
