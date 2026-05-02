# SkyAI

A from-scratch reproduction of GPT-2 (124M), built to deeply understand transformer language models from first principles - every layer, every optimizer step, every tokenization decision implemented and explained.

This project follows the goal of training a 124M-parameter language model end-to-end on a single RTX 4090.

## Why this repo exists

Calling an API is one thing. Building the thing the API calls is another. The point of this project is the second kind of understanding: writing the attention mechanism by hand, watching gradients flow, debugging shape mismatches, profiling kernel launches, and having a model I trained from random weights to coherent(ish) English.

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
