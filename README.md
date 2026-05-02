# SkyAI

A from-scratch reproduction of GPT-2 (124M), built to deeply understand transformer language models from first principles — every layer, every optimizer step, every tokenization decision implemented and explained.

This project follows Andrej Karpathy's *["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU)* tutorial, with the goal of training a 124M-parameter language model end-to-end on a single RTX 4090.

## Status

Phase 0 — setup. See [Roadmap](#roadmap).

## Why this repo exists

Calling an API is one thing. Building the thing the API calls is another. The point of this project is the second kind of understanding: writing the attention mechanism by hand, watching gradients flow, debugging shape mismatches, profiling kernel launches, and — at the end — having a model I trained from random weights to coherent English.

The companion [`journal/`](./journal/) directory captures the reasoning behind each module. Read those alongside the code to follow the thinking, not just the result.

## Structure

```
src/skyai/      model, tokenizer, training loop, eval
notebooks/      exploration and experimentation
journal/        module-by-module learning notes
tests/          unit tests (shapes, gradients, end-to-end smoke)
scripts/        train.py, sample.py, eval.py
data/           training data (gitignored)
checkpoints/    saved models (gitignored)
```

## Roadmap

| Phase | What |
|-------|------|
| 0 | Project scaffold, env, PyTorch warm-up |
| 1 | GPT-2 architecture from scratch; verify by loading HF pretrained weights |
| 2 | Tokenizer, DataLoader, overfit on TinyShakespeare |
| 3 | LR schedule, weight decay, gradient clipping, gradient accumulation, FineWeb-Edu prep |
| 4 | bf16, Flash Attention, `torch.compile`, profiling |
| 5 | Full training run on FineWeb-Edu 10B tokens (~3-5 days on a 4090) |
| 6 | HellaSwag eval, sample CLI, polish |
| 7 | *(Optional)* C++ inference engine — port to llm.c style |

## Hardware

NVIDIA RTX 4090 (24GB VRAM), 64GB system RAM. bf16, Flash Attention 2, and `torch.compile` all in play.

## Setup

```powershell
uv sync           # install dependencies
uv run pytest     # smoke test (verifies CUDA is wired up)
```

## References

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Karpathy's video
- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — companion repo
- [llm.c](https://github.com/karpathy/llm.c) — pure C/CUDA reference for the optional Phase 7
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 paper
