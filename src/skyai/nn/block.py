"""Transformer block: one attention sublayer + one MLP sublayer with pre-norm"""

from __future__ import annotations

import torch
import torch.nn as nn

from skyai.nn.attention import CausalSelfAttention
from skyai.nn.mlp import MLP


class Block(nn.Module):
    """Pre-normalization transformer block"""

    def __init__(self, n_embed: int, n_head: int, hidden_multiple: int = 4) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed=n_embed, n_head=n_head)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed=n_embed, hidden_multiple=hidden_multiple)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x