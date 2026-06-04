"""GPT language model with modern architecture"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyai.nn.block import Block
from skyai.nn.init import init_weights
from skyai.nn.layers import Linear, RMSNorm


def _pad_to_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None
    n_embed: int = 768
    hidden_multiple: int = 4
    rope_theta: float = 100_000.0
    vocab_pad_multiple: int = 128
    tie_weights: bool = False
    logit_softcap: float | None = 15.0

    @property
    def vocab_size_padded(self) -> int:
        return _pad_to_multiple(self.vocab_size, self.vocab_pad_multiple)

    @property
    def head_dim(self) -> int:
        return self.n_embed // self.n_head


class _Transformer(nn.Module):
    """Encoder stack: embeddings, transformer blocks, final layernorm"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size_padded, config.n_embed)
        self.h = nn.ModuleList([
            Block(
                n_embed=config.n_embed,
                n_head=config.n_head,
                n_kv_head=config.n_kv_head,
                hidden_multiple=config.hidden_multiple,
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embed)


class GPT(nn.Module):
    """GPT-2 Language Model"""
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = _Transformer(config)
        self.lm_head = Linear(config.n_embed, config.vocab_size_padded, bias=False)

        if config.tie_weights:
            # Weight tying: token embedding shares weights with output projection
            self.transformer.wte.weight = self.lm_head.weight

        cos, sin = self._build_rotary_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(lambda m: init_weights(m, n_layer=config.n_layer))

    def _build_rotary_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.config.head_dim
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(self.config.block_size).float()
        angles = torch.outer(pos, inv_freq)
        cos = angles.cos()[None, :, None, :]
        sin = angles.sin()[None, :, None, :]
        return cos, sin

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.size()
        if self.config.block_size < T:
            raise ValueError(f'Sequence length {T} exceeds block_size {self.config.block_size}')
        
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos, sin)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits.float() / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

