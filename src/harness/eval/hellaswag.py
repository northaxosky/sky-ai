"""HellaSwag eval helpers, importable by both the training loop and standalone CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
import tiktoken
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa: N812
from tqdm import tqdm

from harness.eval.result import EvalResult
from harness.log import get_logger

logger = get_logger(__name__)

DATA_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "hellaswag"

HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download_file(url: str, fname: Path, chunk_size: int = 1024) -> None:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
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
        logger.info(f"Downloading {HELLASWAG_URLS[split]} to {data_filename}")
        download_file(HELLASWAG_URLS[split], data_filename)


def _download_on_rank_zero(split: str, rank: int) -> None:
    """Rank 0 fetches the split, others wait at a barrier; prevents N ranks
    on a cold node from racing to write the same cache file."""
    is_distributed = dist.is_available() and dist.is_initialized()
    if rank == 0:
        download(split)
    if is_distributed:
        dist.barrier()
    if rank != 0:
        # Sanity-check that rank 0 actually produced the file; if it didn't,
        # iterate_examples below would try to download from a non-zero rank
        # and we'd race again.
        data_filename = DATA_CACHE_DIR / f"hellaswag_{split}.jsonl"
        if not data_filename.exists():
            raise RuntimeError(
                f"HellaSwag cache {data_filename} missing on rank {rank} "
                f"after rank-0 download + barrier"
            )


def iterate_examples(split: str) -> Iterator[dict[str, Any]]:
    """Yields the 10042 examples in val (or whatever split)"""
    download(split)
    with open(DATA_CACHE_DIR / f"hellaswag_{split}.jsonl") as f:
        for line in f:
            yield json.loads(line)


def render_example(
    example: dict[str, Any], *, encoder: tiktoken.Encoding
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

    ctx_tokens = encoder.encode(ctx)
    data: dict[str, Any] = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "ending_tokens": [],
    }

    tok_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for end in endings:
        end_tokens = encoder.encode(" " + end)  # leading space because GPT-2 BPE
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


def evaluate_hellaswag(
    model: nn.Module,
    *,
    encoder: tiktoken.Encoding,
    device: str | torch.device,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    split: str = "val",
) -> EvalResult:
    """Score HellaSwag accuracy on a model, sharded across DDP ranks"""
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else str(device)

    # Race-safe cache fetch before any rank starts iterating.
    _download_on_rank_zero(split, rank)

    # Accumulate on-device so we don't sync GPU->CPU on every example.
    device_t = torch.device(device)
    correct = torch.zeros((), dtype=torch.long, device=device_t)
    correct_norm = torch.zeros((), dtype=torch.long, device=device_t)
    total = torch.zeros((), dtype=torch.long, device=device_t)

    for i, example in enumerate(iterate_examples(split)):
        if i % world_size != rank:
            continue

        _, tokens, mask, label = render_example(example, encoder=encoder)
        tokens = tokens.to(device)
        mask = mask.to(device)
        label_t = torch.tensor(label, dtype=torch.long, device=device_t)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
            logits, _ = model(tokens)

        sum_loss, avg_loss = compute_completion_losses(tokens, mask, logits)
        correct += (sum_loss.argmin() == label_t).long()
        correct_norm += (avg_loss.argmin() == label_t).long()
        total += 1

    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_norm, op=dist.ReduceOp.SUM)

    num_total = int(total.item())
    num_correct = int(correct.item())
    num_correct_norm = int(correct_norm.item())

    acc = num_correct / num_total if num_total > 0 else 0.0
    acc_norm = num_correct_norm / num_total if num_total > 0 else 0.0

    return EvalResult(
        name="hellaswag", metrics={"acc": acc, "acc_norm": acc_norm}, num_examples=num_total
    )
