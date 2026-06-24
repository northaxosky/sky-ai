from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from gpt.model import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # one fused projection produces q, k, v together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection back into the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj._is_residual_projection = True
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # C == n_embd

        qkv = self.c_attn(x)  # (B, T, 3C) — one matmul
        q, k, v = qkv.split(self.n_embd, dim=2)  # three (B, T, C)

        head_dim = C // self.n_head  # 768 / 12 = 64
        # (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # flash attention via SDPA; is_causal applies the lower-triangular mask
        # internally without ever building a (T, T) tensor
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hd)

        # (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C): concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
