"""Transformer block: pre-norm attention sublayer + pre-norm MLP sublayer"""

from __future__ import annotations

import torch
import torch.nn as nn

from skyai.attention import CausalSelfAttention
from skyai.layers import RMSNorm
from skyai.mlp import MLP


class Block(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, n_kv_head: int | None = None, hidden_multiple: int = 4
    ) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, hidden_multiple=hidden_multiple)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x
