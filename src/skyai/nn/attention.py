"""Causal self-attention used by every GPT layer"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyai.nn.layers import ResidualProjection


class CausalSelfAttention(nn.Module):
    """Multi=head causal self-attention with fused QKV prjection"""

    def __init__(self, n_embed: int, n_head: int) -> None:
        super().__init__()
        if n_embed % n_head != 0:
            raise ValueError(f'n_embed ({n_embed}) must be divisible by n_head ({n_head})')
        
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_size = n_embed // n_head

        # Fused projection: one matmul produces Q, K, V concatenated
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)

        # ResidualProjection applies the residual-path scaling
        self.c_proj = ResidualProjection(n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Fused projection split into Q, K, V (each B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # Reshape to (B, n_head, T, head_size) so attension sees the head dim
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Flash attentions path
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Back to (B, T, C) before output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)