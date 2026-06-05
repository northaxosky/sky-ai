"""Multi-sample text generation from a trained model"""

from __future__ import annotations

import tiktoken
import torch
from torch import nn

from skyai.generate import generate


def sample(
    model: nn.Module,
    encoder: tiktoken.Encoding,
    prompt: str,
    *,
    n_samples: int,
    max_length: int,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = 50,
    generator: torch.Generator | None = None,
    max_context_len: int | None = None,
) -> list[str]:
    """Generate n_samples completions of `prompt`, decoded to strings.

    max_length is the total output length cap (prompt + new tokens). The
    returned strings include the prompt as a prefix. max_context_len
    defaults to model.config.block_size when None.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1; got {max_length}")

    prompt_ids = encoder.encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    x = x.repeat(n_samples, 1)

    if max_context_len is None:
        max_context_len = model.config.block_size  # pyright: ignore

    max_new_tokens = max(1, max_length - x.size(1))
    out = generate(
        model,
        x,
        max_new_tokens=max_new_tokens,
        max_context_len=max_context_len,  # pyright: ignore
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    return [encoder.decode(out[i].tolist()) for i in range(n_samples)]
