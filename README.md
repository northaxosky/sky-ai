# SkyAI

A from-scratch language model project in two acts. First, reproduce GPT-2
(124M) from the ground up. Then modernize the architecture and scale toward a
~1.5B model. Every layer, every optimizer step, and every tokenization decision
is implemented and explained rather than imported.

## Why this repo exists

Calling an API is one thing. Building the thing the API calls is another. The
point of this project is the second kind of understanding: writing the attention
mechanism by hand, watching gradients flow, debugging shape mismatches,
profiling kernel launches, and having a model I trained from random weights to
coherent(ish) English.

The companion [`journal/`](./journal/) directory captures the reasoning behind
each module. Read those alongside the code to follow the thinking, not just the
result.

## Act 1: reproduce GPT-2 (124M)

The first milestone is the textbook GPT-2 small, built from scratch and trained
end-to-end until it reproduces Karpathy's published baseline. Trained on 10B
tokens of FineWeb-Edu on 8x A100 (Lambda Cloud), two runs total, total spend
$111.

| Run | Val loss | HellaSwag | Notes |
|-----|---------:|----------:|-------|
| V1  | 3.402    | 26.9%     | three bugs in flight (DataLoader rank offset, autocast device_type, load_tokens dtype) |
| V2  | **3.397** | **27.3%** | bugs fixed, otherwise identical |
| Karpathy 10B baseline | ~3.29 | ~28% | his llm.c stated number for the same token budget |
| Karpathy 40B headline | 3.05 | 30.5% | 4 epochs over the same shards, not the released script |

Wandb: [V1](https://wandb.ai/kuz-skyai/skyai/runs/zh522npo), [V2](https://wandb.ai/kuz-skyai/skyai/runs/b591tcsx).

V2 lands ~0.11 nat above Karpathy's 10B baseline. The catch worth writing down:
the famous 3.05 figure is a 40B-token (4-epoch) result, not the released
`max_steps=19073` script, which is exactly one epoch over FineWeb-Edu
sample-10BT. That reframing came out of a multi-source audit after the bug fixes
between V1 and V2 only moved val loss by 0.005. The full story is in
[`journal/01-reproduce-gpt2.md`](./journal/01-reproduce-gpt2.md).

Recipe: standard nanoGPT, no exotic tricks. AdamW with weight decay (0.1 on 2D
params, 0 elsewhere), cosine LR with linear warmup, gradient clipping at 1.0,
gradient accumulation to a 0.5M-token effective batch, BF16 mixed precision via
autocast, Flash Attention through PyTorch SDPA, fused AdamW, `torch.compile`,
and weight tying between the input embedding and output head.

## Act 2: modern architecture, scaling toward ~1.5B

The second act rebuilds the model around the current decoder-LM stack and a
modern training recipe, then scales it up.

Architecture changes from the GPT-2 baseline:

- RMSNorm in place of LayerNorm, plus a post-embedding norm
- Rotary position embeddings (RoPE) instead of learned positional embeddings
- SwiGLU feed-forward instead of the GELU MLP
- Grouped-query attention (GQA)
- QK-normalization with q/k sharpening
- Untied input and output embeddings
- Logit soft-capping
- cl100k tokenizer instead of GPT-2 BPE, with internal vocab padding kept
  separate from the logical tokenizer vocab

Training recipe changes:

- Muon optimizer for the transformer matrix parameters, AdamW for the
  embeddings and the output head
- Warmup-stable-decay LR schedule instead of cosine
- Width- and batch-scaled per-group learning rates

Status: the modern stack is implemented, unit-tested, and validated locally on a
single 4090, where it trains cleanly at small scale. A ~1.5B configuration
([`configs/skyai-xl.yaml`](./configs/skyai-xl.yaml): 48 layers, 1536 hidden, 32
heads, 8 KV heads, 2048 context) is staged for an 8x H100 run. That run has not
happened yet. Results will be added here when it does.

## Project layout

```
src/skyai/      model, tokenizer, harness (cli/, config/, data/, eval/, nn/, training/) plus checkpoint.py, ablation.py, generate.py, sample.py
notebooks/      prereq explorations and post-train sanity checks
journal/        module-by-module learning notes
tests/          shape + gradient unit tests, end-to-end smoke, golden numerics fixture
scripts/        shard_text.py + hellaswag.py (one-time data prep)
configs/        run configs (base.yaml is the GPT-2 V2 recipe; skyai-xl.yaml is the modern ~1.5B config)
data/           token shards (gitignored)
checkpoints/    saved models (gitignored)
```

## Hardware

NVIDIA RTX 4090 (24GB VRAM), 64GB RAM, WSL Ubuntu on Windows for local
development. Cloud runs are 8x A100 (GPT-2 reproduction) and 8x H100 (the modern
scale-up) on Lambda. `bf16`, Flash Attention, and `torch.compile` are all in
play; `torch.compile` requires Linux (Triton has no Windows wheels).

## Setup

```bash
uv sync                 # install dependencies (resolves the CUDA 12.8 torch wheel)
uv run pytest           # full test suite (asserts CUDA + 4090 + bf16; fails on other hardware by design)
uv run skyai doctor     # one-shot env + project sanity check
```

The test suite's environment assertions will fail anywhere that isn't a 4090
box. That's intentional, not a bug to relax.

## Using the harness

Four subcommands cover the lifecycle:

```bash
uv run skyai train --config configs/base.yaml          # train from scratch
uv run skyai train --config configs/base.yaml --resume # resume from latest checkpoint
uv run skyai eval --config configs/base.yaml --checkpoint checkpoints/best.pt
uv run skyai sample --checkpoint checkpoints/best.pt --prompt "Hello"
uv run skyai doctor --config configs/base.yaml         # env + config sanity
```

Configs are YAML and validated through pydantic. Any field can be overridden on
the CLI:

```bash
uv run skyai train --config configs/base.yaml --override schedule.max_steps=1000
```

Multi-GPU is just `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m skyai.cli.main train --config configs/base.yaml
```

The harness pins its numerics: a golden-fixture short-run test
(`tests/test_golden.py`) catches accidental numerical drift on refactor.

## References

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU): Karpathy's video
- [build-nanogpt](https://github.com/karpathy/build-nanogpt): companion repo
- [nanochat](https://github.com/karpathy/nanochat): modern speedrun harness, reference for the Act 2 architecture and recipe
- [llm.c](https://github.com/karpathy/llm.c): pure C/CUDA reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 paper
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864), [GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202), [GQA](https://arxiv.org/abs/2305.13245), [Muon](https://kellerjordan.github.io/posts/muon/): Act 2 architecture and optimizer
