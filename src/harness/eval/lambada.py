"""Lambada OpenAI Eval: teacher forced last word prediction over a discourse passage"""

from __future__ import annotations

import math

import tiktoken
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F

from harness.eval.result import EvalResult
from harness.log import get_logger

logger = get_logger(__name__)

LAMBADA_DATASET_ID = "EleutherAI/lambada_openai"
LAMBADA_SPLIT = "test"
LAMBADA_REVISION = "900124bf3b8235c6daf21033af9948b3f07346c4"


def _load_examples() -> list[str]:
    """Load LAMBADA OpenAI test passages as raw strings"""
    ds = load_dataset(LAMBADA_DATASET_ID, split=LAMBADA_SPLIT, revision=LAMBADA_REVISION)
    return [ex["text"] for ex in ds]  # pyright: ignore


def _render_lambada(
    text: str, encoder: tiktoken.Encoding, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode a LAMBADA passage and split into model input + ground truth target"""
    _, sep, target = text.rpartition(" ")
    if not sep or not target:
        raise ValueError(f"LAMBADA passage has no whitespace boundary: {text!r}")

    full_ids = encoder.encode(text)
    target_len = len(encoder.encode(" " + target))
    if target_len >= len(full_ids):
        raise ValueError(
            f"target_len ({target_len}) >= len(full_ids) ({len(full_ids)}) for {text!r}"
        )

    if len(full_ids) > block_size:
        full_ids = full_ids[-block_size:]
        if target_len >= len(full_ids):
            raise ValueError(
                f"target_len ({target_len}) >= block_size ({block_size}) after truncation"
            )

    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long).unsqueeze(0)
    gt_target_ids = torch.tensor(full_ids[-target_len:], dtype=torch.long)
    return input_ids, gt_target_ids, target_len


def _score_lambada_logits(
    logits: torch.Tensor, gt_target_ids: torch.Tensor, target_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score model logits against the ground truth target span.

    Returns (is_correct, sum_nll) as 0-d tensors on the same device as logits.
    Keeping the result on-device lets the caller accumulate without per-example
    GPU->CPU syncs.
    """
    target_logits = logits[0, -target_len:, :]  # (target_len, vocab)

    preds = target_logits.argmax(dim=-1)
    is_correct = (preds == gt_target_ids).all().long()

    # fp32 for the softmax/log/gather
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    gt_log_probs = log_probs.gather(1, gt_target_ids.unsqueeze(1)).squeeze(1)
    sum_nll = -gt_log_probs.sum()
    return is_correct, sum_nll


def evaluate_lambada(
    model: nn.Module,
    *,
    encoder: tiktoken.Encoding,
    device: str | torch.device,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 1024,
) -> EvalResult:
    """Score LAMBADA OpenAI accuracy + perplexity on a model"""
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else str(device)
    examples = _load_examples()

    device_t = torch.device(device)
    correct = torch.zeros((), dtype=torch.long, device=device_t)
    total_nll = torch.zeros((), dtype=torch.float64, device=device_t)
    total_tokens = torch.zeros((), dtype=torch.long, device=device_t)
    total_examples = torch.zeros((), dtype=torch.long, device=device_t)

    for i, text in enumerate(examples):
        if i % world_size != rank:
            continue

        input_ids, gt_target_ids, target_len = _render_lambada(text, encoder, block_size)
        input_ids = input_ids.to(device)
        gt_target_ids = gt_target_ids.to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
            logits, _ = model(input_ids)

        is_correct, sum_nll = _score_lambada_logits(logits, gt_target_ids, target_len)
        correct += is_correct
        total_nll += sum_nll.to(torch.float64)
        total_tokens += target_len
        total_examples += 1

    if world_size > 1:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)

    num_correct = int(correct.item())
    num_total = int(total_examples.item())
    n_tokens = int(total_tokens.item())
    nll_sum = float(total_nll.item())

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    perplexity = math.exp(nll_sum / n_tokens) if n_tokens > 0 else float("inf")

    return EvalResult(
        name="lambada",
        metrics={"accuracy": accuracy, "perplexity": perplexity},
        num_examples=num_total,
    )
