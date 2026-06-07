# SkyAI

A from-scratch reproduction of GPT-2 (124M), built to deeply understand transformer language models from first principles: every layer, every optimizer step, every tokenization decision implemented and explained.

This project follows the goal of training a 124M-parameter language model end-to-end on a single RTX 4090.

## Why this repo exists

Calling an API is one thing. Building the thing the API calls is another. The point of this project is the second kind of understanding: writing the attention mechanism by hand, watching gradients flow, debugging shape mismatches, profiling kernel launches, and having a model I trained from random weights to coherent(ish) English.

The companion [`journal/`](./journal/) directory captures the reasoning behind each module. Read those alongside the code to follow the thinking, not just the result.

## Results

Trained the 124M model on 10B tokens of FineWeb-Edu on 8x A100 (Lambda Cloud), two runs total, total spend $111.

| Run | Val loss | HellaSwag | Notes |
|-----|---------:|----------:|-------|
| V1  | 3.402    | 26.9%     | three bugs in flight (DataLoader rank offset, autocast device_type, load_tokens dtype) |
| V2  | **3.397** | **27.3%** | bugs fixed, otherwise identical |
| Karpathy 10B baseline | ~3.29 | ~28% | his llm.c stated number for the same token budget |
| Karpathy 40B headline | 3.05 | 30.5% | 4 epochs over the same shards, not the released script |

Wandb: [V1](https://wandb.ai/kuz-skyai/skyai/runs/zh522npo) - [V2](https://wandb.ai/kuz-skyai/skyai/runs/b591tcsx).

V2 lands ~0.11 nat above Karpathy's 10B baseline. The three bug fixes between V1 and V2 only moved val loss by 0.005, so the bugs weren't the gap. A local-vs-cloud A/B at 200 steps narrowed the suspect to `torch.compile` and/or DDP numerical ordering: local eager runs at 0.18 nat ahead of cloud compile at step 199. That's the same direction and roughly the same magnitude as the full-run gap. Karpathy's `torch.compile` has been broken in `build-nanogpt` since 2024-06-09 (see his `# TODO fix` comment) so this is a known landmine, not a new one.

### Recipe

Standard nanoGPT recipe with no exotic tricks: AdamW with weight decay (0.1 on 2D params, 0 elsewhere), cosine LR schedule with linear warmup, gradient clipping at 1.0, gradient accumulation to a 0.5M-token effective batch, BF16 mixed precision via autocast, Flash Attention 2 through PyTorch SDPA, fused AdamW, `torch.compile` on, weight tying between input embedding and output head. Karpathy's video is the spec.

## Project layout

```
src/skyai/      model, tokenizer, harness (cli/, config/, data/, eval/, training/, sample.py)
notebooks/      prereq explorations and post-train sanity checks
journal/        module-by-module learning notes
tests/          shape + gradient unit tests, end-to-end smoke, golden numerics fixture
scripts/        shard_text.py + hellaswag.py (one-time data prep)
configs/        run configs (base.yaml is the V2 production config)
data/           token shards (gitignored)
checkpoints/    saved models (gitignored)
```

## Hardware

NVIDIA RTX 4090 (24GB VRAM), 64GB RAM, WSL Ubuntu on Windows. Cloud runs were 8x A100 on Lambda. `bf16`, Flash Attention 2, and `torch.compile` are all in play; `torch.compile` requires Linux (Triton has no Windows wheels).

## Setup

```bash
uv sync                 # install dependencies (resolves the CUDA 12.8 torch wheel)
uv run pytest           # full test suite (asserts CUDA + 4090 + bf16; fails on other hardware by design)
uv run skyai doctor     # one-shot env + project sanity check
```

The test suite's environment assertions will fail anywhere that isn't a 4090 box. That's intentional, not a bug to relax.

## Using the harness

Four subcommands cover the lifecycle:

```bash
uv run skyai train --config configs/base.yaml         # train from scratch
uv run skyai train --config configs/base.yaml --resume # resume from latest checkpoint
uv run skyai eval --checkpoint checkpoints/best.pt    # rerun eval suite on a saved checkpoint
uv run skyai sample --checkpoint checkpoints/best.pt --prompt "Hello"
uv run skyai doctor --config configs/base.yaml        # env + config sanity
```

Configs are YAML and validated through pydantic. Any field can be overridden on the CLI:

```bash
uv run skyai train --config configs/base.yaml --override schedule.max_steps=1000
```

Multi-GPU is just `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m skyai.cli.main train --config configs/base.yaml
```

`configs/base.yaml` is the production config that produced V2. The harness pins those numerics: a golden-fixture short-run test (`tests/test_golden.py`) catches accidental numerical drift on refactor.

## References

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): Karpathy's video
- [build-nanogpt](https://github.com/karpathy/build-nanogpt): companion repo
- [llm.c](https://github.com/karpathy/llm.c): pure C/CUDA reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 paper
