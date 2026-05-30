"""Lambada OpenAI Eval: teacher forced last word prediction over a discourse passage"""

from __future__ import annotations

import math

import tiktoken
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F

from skyai.eval.result import EvalResult
from skyai.log import get_logger

logger = get_logger(__name__)

LAMBADA_DATASET_ID = "EleutherAI/lambada_openai"
LAMBADA_SPLIT = "test"
LAMBADA_REVISION = "900124bf3b8235c6daf21033af9948b3f07346c4"


def _load_examples() -> list[str]:
    """Load LAMBADA OpenAI test passages as raw strings"""
    ds = load_dataset(
        LAMBADA_DATASET_ID,
        split=LAMBADA_SPLIT,
        revision=LAMBADA_REVISION
    )
    return [ex["text"] for ex in ds] # pyright: ignore

def _render_lambada(text: str, encoder: tiktoken.Encoding, block_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode a LAMBADA passage and split into model input + ground truth target"""
    _, sep, target = text.rpartition(" ")
    if not sep or not target:
        raise ValueError(f"LAMBADA passage has no whitespace boundary: {text!r}")
    
    full_ids = encoder.encode(text)
    target_len = len(encoder.encode(" " + target))
    if target_len >= len(full_ids):
        raise ValueError(f"target_len ({target_len}) >= len(full_ids) ({len(full_ids)}) for {text!r}")
    
    if len(full_ids) > block_size:
        full_ids = full_ids[-block_size:]
        if target_len >= len(full_ids):
            raise ValueError(f"target_len ({target_len}) >= block_size ({block_size}) after truncation")
        
    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long).unsqueeze(0)
    gt_target_ids = torch.tensor(full_ids[-target_len:], dtype=torch.long)
    return input_ids, gt_target_ids, target_len

def _score_lambada_logits(logits: torch.Tensor, gt_target_ids: torch.Tensor, target_len: int) -> tuple[bool, float]:
    """Score model logits against the ground truth target span"""
    target_logits = logits[0, -target_len:, :] # (target_len, vocab)

    preds = target_logits.argmax(dim=-1)
    is_correct = bool(torch.equal(preds, gt_target_ids))

    # fp32 for the softmax/log/gather
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    gt_log_probs = log_probs.gather(1, gt_target_ids.unsqueeze(1)).squeeze(1)
    sum_nll = float(-gt_log_probs.sum().item())
    return is_correct, sum_nll

def evaluate_lambada(model:nn.Module, *,
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

    num_correct = 0
    total_nll = 0.0
    total_tokens = 0
    num_total = 0

    for i, text in enumerate(examples):
        if i % world_size != rank:
            continue

        input_ids, gt_target_ids, target_len = _render_lambada(text, encoder, block_size)
        input_ids = input_ids.to(device)
        gt_target_ids = gt_target_ids.to(device)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=dtype):
            logits, _ = model(input_ids)

        is_correct, sum_nll = _score_lambada_logits(logits, gt_target_ids, target_len)
        num_correct += int(is_correct)
        total_nll += sum_nll
        total_tokens += target_len
        num_total += 1

    if world_size > 1:
        t_int = torch.tensor([num_correct, total_tokens, num_total], dtype=torch.long, device=device)
        t_nll = torch.tensor(total_nll, dtype=torch.float64, device=device)
        dist.all_reduce(t_int, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_nll, op=dist.ReduceOp.SUM)
        num_correct = int(t_int[0].item())
        total_tokens = int(t_int[1].item())
        num_total = int(t_int[2].item())
        total_nll = float(t_nll.item())

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")

    return EvalResult(name="lambada", metrics={"accuracy": accuracy, "perplexity": perplexity}, num_examples=num_total)