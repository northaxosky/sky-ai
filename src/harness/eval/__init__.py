"""Pluggable eval suite: dispatch from schema names to eval functions"""

from __future__ import annotations

import tiktoken
import torch
from torch import nn

from harness.eval.hellaswag import evaluate_hellaswag
from harness.eval.lambada import evaluate_lambada
from harness.eval.result import EvalFn, EvalResult

EVALS: dict[str, EvalFn] = {
    "hellaswag": evaluate_hellaswag,
    "lambada": evaluate_lambada,
}


def run_evals(
    names: list[str],
    model: nn.Module,
    *,
    encoder: tiktoken.Encoding,
    device: str | torch.device,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, EvalResult]:
    """Run a sequence of evals by name, preserve input order"""
    results: dict[str, EvalResult] = {}
    for name in names:
        if name not in EVALS:
            raise KeyError(f"Unknown eval {name!r}. Available: {sorted(EVALS.keys())}")

        fn = EVALS[name]
        results[name] = fn(
            model, encoder=encoder, device=device, rank=rank, world_size=world_size, dtype=dtype
        )

    return results
