from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt.block import Block
from gpt.init import init_weights


@dataclass
class GPTConfig:
    block_size: int = 1024  # Max context length (size of wpe)
    vocab_size: int = 50257  # GPT-2 BPE vocab (padded to 50304 during training)
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Attention heads -> head_dim = n_embd / n_head = 64
    n_embd: int = 768  # Residual-stream width (the "C" in (B, T, C))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token table
                wpe=nn.Embedding(config.block_size, config.n_embd),  # learned positions
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the output head IS the input embedding (same tensor)
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 init: recurse over every submodule
        self.apply(lambda m: init_weights(m, config.n_layer))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert self.config.block_size >= T, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        tok_emb = self.transformer.wte(idx)  # (B, T, C) token -> vector
        pos_emb = self.transformer.wpe(pos)  # (T, C)    position -> vector
        x = tok_emb + pos_emb  # (B, T, C) broadcast-add positions once

        for block in self.transformer.h:
            x = block(x)  # (B, T, C) -> (B, T, C)

        x = self.transformer.ln_f(x)  # Final LayerNorm
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> GPT:
        """Load OpenAI's pre-trained GPT-2 weights from HuggingFace into our model"""
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            "gpt2-medium": dict(n_layer=12, n_head=12, n_embd=768),  # 350M
            "gpt2-large": dict(n_layer=12, n_head=12, n_embd=768),  # 774M
            "gpt2-xl": dict(n_layer=12, n_head=12, n_embd=768),  # 1558M
        }[model_type]

        config = GPTConfig(vocab_size=50257, block_size=1024, **config_args)
        model = cls(config)
        sd = model.state_dict()

        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()

        # HF carries causal-mask buffers we don't (SDPA builds it)
        hf_keys = [k for k in sd_hf if not k.endswith((".attn.bias", ".attn.masked_bias"))]

        # These four are Conv1D in HF (in, out): transpose to (out, in)
        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        assert len(hf_keys) == len(sd), f"key-count mismatch: {len(hf_keys)} vs {len(sd)}"
        for k in hf_keys:
            if k.endswith(transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
