"""Token-level text generation from autoregressive language models"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def generate(model: nn.Module, prompt_ids: torch.Tensor, *,
             max_new_tokens: int,
             max_context_len: int = 1024,
             temperature: float = 1.0,
             top_k: int | None = 50,
             generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a prompt"""
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 if specified; got {top_k}")
    
    was_training = model.training
    model.eval()
    try:
        x = prompt_ids
        for _ in range(max_new_tokens):
            context = x if x.size(1) <= max_context_len else x[:, -max_context_len:]
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = _apply_top_k(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            x = torch.cat([x, next_id], dim=1)
        return x
    finally:
        if was_training:
            model.train()

def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k >= logits.size(-1):
        return logits
    topk_vals, _ = torch.topk(logits, k=k, dim=-1)
    threshold = topk_vals[:, -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))