"""GPT-2 model definition"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyai.nn.block import Block
from skyai.nn.init import init_gpt2_weights


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    hidden_multiple: int = 4


class _Transformer(nn.Module):
    """Encoder stack: embeddings, transformer blocks, final layernorm"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.wpe = nn.Embedding(config.block_size, config.n_embed)
        self.h = nn.ModuleList([
            Block(
                n_embed=config.n_embed,
                n_head=config.n_head,
                hidden_multiple=config.hidden_multiple,
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embed)


class GPT(nn.Module):
    """GPT-2 Language Model"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = _Transformer(config)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # Weight tying: token embedding shares weights with output projection
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(lambda m: init_gpt2_weights(m, n_layer=config.n_layer))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f'Sequence length {T} exceeds block_size {self.config.block_size}')
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss