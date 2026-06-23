"""Causal self-attention with RoPE, GQA, QK-Norm"""

from __future__ import annotations

import torch
import torch.nn as nn

from skyai.flash import attention
from skyai.layers import Linear, ResidualProjection, RMSNorm


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the last dim of x"""
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    out = torch.cat([y1, y2], dim=-1)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi=head causal self-attention with fused QKV prjection"""

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int | None = None,
        use_qk_norm: bool = True,
        qk_sharpen: float = 1.2,
    ) -> None:
        super().__init__()
        if n_embed % n_head != 0:
            raise ValueError(f"n_embed ({n_embed}) must be divisible by n_head ({n_head})")

        n_kv_head = n_kv_head if n_kv_head is not None else n_head
        if n_kv_head > n_head:
            raise ValueError(f"n_kv_head ({n_kv_head}) must be <= n_head ({n_head})")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embed // n_head
        self.qk_sharpen = qk_sharpen

        self.c_q = Linear(n_embed, n_head * self.head_dim, bias=False)
        self.c_k = Linear(n_embed, n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(n_embed, n_kv_head * self.head_dim, bias=False)
        self.c_proj = ResidualProjection(n_embed, n_embed, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = self.q_norm(q) * self.qk_sharpen
        k = self.k_norm(k) * self.qk_sharpen

        y = attention(q, k, v, is_causal=True)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)
