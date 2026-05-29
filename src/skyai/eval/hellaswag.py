"""
HellaSwag eval helpers, importable by both the training loop and standalone CLI.

The wire format is documented at https://github.com/rowanz/hellaswag. Each example
has a context and 4 candidate completions; the task is to pick the right one.
We score by the average per-token NLL of each completion and pick argmin
(this is the "acc_norm" variant; length-normalized).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
import tiktoken
import torch
from torch.nn import functional as F  # noqa: N812
from tqdm import tqdm

from skyai.log import get_logger

logger = get_logger(__name__)

DATA_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "hellaswag"

HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: Path, chunk_size: int = 1024) -> None:
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as f,
        tqdm(
            desc=str(fname),
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)


def download(split: str) -> None:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data_filename = DATA_CACHE_DIR / f"hellaswag_{split}.jsonl"
    if not data_filename.exists():
        logger.info("Downloading %s to %s", HELLASWAG_URLS[split], data_filename)
        download_file(HELLASWAG_URLS[split], data_filename)


def render_example(
    example: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, int]:
    """
    Render a HellaSwag example as three tensors:
    - tokens: (4, N) token ids for context + each candidate completion
    - mask:   (4, N) 1 in the completion region (where likelihood is scored)
    - label:  index (0..3) of the correct completion
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    data: dict[str, Any] = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "ending_tokens": [],
    }

    tok_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # leading space because GPT-2 BPE
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # Rows differ in length; pad to max with zeros (mask is 0 there, so padding
    # contributes nothing to the loss).
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows, strict=True)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split: str) -> Iterator[dict[str, Any]]:
    """Yields the 10,042 examples in val (or whichever split is requested)."""
    download(split)
    with open(DATA_CACHE_DIR / f"hellaswag_{split}.jsonl") as f:
        for line in f:
            yield json.loads(line)


def compute_completion_losses(
    tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-row sum and average of cross-entropy loss inside the completion region.
    Inputs are (4, N) for tokens/mask and (4, N, vocab) for logits.
    Returns (sum_loss, avg_loss), each (4,).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)
    losses = F.cross_entropy(flat_logits, flat_tokens, reduction="none").view(tokens.size(0), -1)

    # Mask shifts with logits so the first scored position is the last prompt token
    shift_mask = mask[..., 1:].contiguous()
    masked = losses * shift_mask
    sum_loss = masked.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return sum_loss, avg_loss


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """Return the candidate index (0..3) with the lowest length-normalized loss (acc_norm prediction)."""
    _, avg_loss = compute_completion_losses(tokens, mask, logits)
    return int(avg_loss.argmin().item())
