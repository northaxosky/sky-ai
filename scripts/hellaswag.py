"""
Standalone HellaSwag eval CLI.

Loads a HuggingFace pretrained GPT-2 variant and evaluates it on the HellaSwag
val set, reporting both raw accuracy (sum-loss argmin) and length-normalized
accuracy (avg-loss argmin).

Reference scores against this scoring scheme:
    gpt2 (124M)     : acc 0.2859, acc_norm 0.2955
    gpt2-xl (1558M) : acc 0.3842, acc_norm 0.4893

Usage:
    uv run python scripts/hellaswag.py                 # gpt2 on cuda
    uv run python scripts/hellaswag.py -m gpt2-xl      # bigger model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
import torch
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel

from harness.eval.hellaswag import compute_completion_losses, iterate_examples, render_example

# Load .env from repo root (regardless of cwd)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@torch.no_grad()
def evaluate(model_type: str, device: str) -> None:
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)  # pyright: ignore
    enc = tiktoken.get_encoding("gpt2")

    num_correct_norm = 0
    num_correct = 0
    for num_total, example in enumerate(iterate_examples("val"), start=1):
        data, tokens, mask, label = render_example(example, encoder=enc)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits
        sum_loss, avg_loss = compute_completion_losses(tokens, mask, logits)

        pred = int(sum_loss.argmin().item())
        pred_norm = int(avg_loss.argmin().item())

        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(
            f"{num_total} acc: {num_correct}/{num_total}={num_correct / num_total:.4f} "
            f"acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}"
        )

        if num_total <= 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a HuggingFace GPT-2 against HellaSwag val."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default="gpt2", help="HF model identifier (default: gpt2)"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="torch device (default: cuda)"
    )
    args = parser.parse_args()
    evaluate(args.model_type, args.device)


if __name__ == "__main__":
    main()
