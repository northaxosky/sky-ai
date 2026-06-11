# Reproduce GPT-2 (124M)

> Scaffold. The "What surprised me" section is intentionally left for my own
> raw notes; everything else is a first pass to react to and rewrite in my
> voice.

## What I'm building

The first SkyAI-proper milestone: GPT-2 124M built from scratch and trained
end-to-end until it reproduces Karpathy's published baseline. This is where the
prereqs (micrograd through "Let's build GPT") stop being toy exercises and turn
into a real training run on real hardware.

The model is the textbook GPT-2 small: `n_layer=12, n_head=12, n_embed=768,
block_size=1024, vocab_size=50257` (padded to 50304 for tensor-core alignment),
pre-norm LayerNorm, causal multi-head attention, weight tying between the input
embedding and the output head. No exotic tricks. The whole point is to match a
known-good reference, not to innovate.

The training recipe is the nanoGPT recipe: AdamW with decoupled weight decay
(0.1 on 2D params, 0 on the rest), a cosine LR schedule with linear warmup,
gradient clipping at 1.0, gradient accumulation to a 0.5M-token effective batch,
BF16 mixed precision through `torch.autocast`, Flash Attention via PyTorch's
SDPA, fused AdamW, and `torch.compile`. Periodic validation, HellaSwag, and
sampling run inside the loop.

Two cloud runs on Lambda 8x A100 produced the result:

| Run | Val loss | HellaSwag | Notes |
|-----|---------:|----------:|-------|
| V1  | 3.402    | 26.9%     | three bugs in flight |
| V2  | 3.397    | 27.3%     | bugs fixed, otherwise identical |

Total spend across both runs: $111. Final result lands ~0.11 nat above
Karpathy's stated 10B-token baseline (~3.29), which is inside reproduction
variance.

The harness around the model is its own deliverable: a unified `skyai` CLI
(train / eval / sample / resume / doctor), YAML + pydantic config, rank-aware
logging, a pluggable eval suite (HellaSwag + LAMBADA), checkpoint hygiene with
atomic writes and a sidecar manifest, and a golden short-run regression test
that pins the numerics so refactors can't silently drift them.

## Concepts I had to internalize

These are the things that go beyond the prereqs. The prereqs taught the
architecture; this milestone was about everything *around* the architecture
that makes a real run work.

- **Distributed Data Parallel (DDP).** Each GPU holds a full model replica and
  a disjoint slice of the batch. Gradients are all-reduced (averaged) across
  ranks before the optimizer step, so every replica stays identical. The
  subtlety is data sharding: each rank must read a *different* slice of the
  shards, offset by `rank`, advancing by `batch * block * world_size`. Getting
  that offset wrong (using `world_size` where you meant `rank`) means all 8 GPUs
  silently train on the same data.
- **Gradient accumulation.** When the target batch (0.5M tokens) doesn't fit in
  VRAM, split it into micro-batches, run forward/backward on each, and only
  `optimizer.step()` after N of them. The loss must be divided by
  `grad_accum_steps` so the accumulated gradient matches what a single large
  batch would have produced. With DDP, you also skip the gradient all-reduce on
  every micro-step except the last (`require_backward_grad_sync`), otherwise you
  pay the communication cost N times for nothing.
- **Mixed precision via autocast.** `torch.autocast` runs matmuls in BF16 while
  keeping a FP32 master copy of the weights and FP32 reductions. BF16 has the
  same exponent range as FP32 (just fewer mantissa bits), so no loss scaling is
  needed the way FP16 requires. The trap: autocast is device-typed
  (`"cuda"` vs `"cpu"`), and passing the full device string (`"cuda:0"`) instead
  of the stripped type is a real bug.
- **Flash Attention through SDPA.** `F.scaled_dot_product_attention` picks a
  fused kernel (FlashAttention-2 on the right hardware) that never materializes
  the full `(T, T)` attention matrix in HBM. It's a memory and speed win for
  free, and it means the hand-written causal mask buffer from the prereqs is
  dead weight (SDPA generates the mask internally with `is_causal=True`).
- **The cosine schedule with warmup.** Linear ramp from 0 to peak over the
  warmup steps, then cosine decay to a floor. Warmup keeps the early, high-
  variance gradients from blowing up the freshly initialized weights; the cosine
  tail anneals the LR so late training settles instead of bouncing.
- **The FineWeb-Edu data pipeline.** Stream the HF dataset, tokenize with the
  GPT-2 BPE, pack into fixed-size `.npy` shards of ~100M tokens each, reserve the
  first shard for validation. The loader memory-maps shards and walks them
  sequentially with per-rank offsets. `uint16` is enough for the 50257-token
  GPT-2 vocab; wider tokenizers need `uint32`.
- **HellaSwag as a length-normalized multiple-choice eval.** For each of the 4
  candidate endings, score the completion region's cross-entropy, normalize by
  token count, and pick the lowest. `acc_norm` (length-normalized) is the number
  everyone reports; raw `acc` over-rewards short endings.
- **The golden-fixture regression test.** A tiny, deterministic, CPU-pinned
  training run whose per-step losses, final val loss, sample text, and parameter
  checksum are frozen in a JSON fixture. Any change to model math or train/eval
  timing breaks it loudly, which is exactly what you want when refactoring a
  training loop you can't afford to silently corrupt.
- **Multi-source audit as a debugging methodology.** When a single perspective
  keeps confirming its own framing, parallel independent reviewers (each given
  the diff, the references, and an adversarial brief) catch interpretation
  errors that staring harder never would. The 40B-token discovery below came out
  of this, not out of more thinking.

## What surprised me

<!--
  RAW NOTES GO HERE, IN MY VOICE, VERBATIM. Don't let the agent polish this.
  Drop the rough reactions, the dumb-in-hindsight moments, the things that
  clicked late. Bullet form is fine. Typos are fine.

  Prompts to react to (delete after answering):
  - (EXPAND: the "0.35 nat gap" panic -> the audit -> the 40B-token bombshell.
    What did it feel like to discover the famous 3.05 number was a 4-epoch run,
    not the released script? That's the emotional core of this entry.)
  - (EXPAND: V2 landing on the same number as V1 after fixing three real bugs.
    The moment the bug fixes didn't move the needle and what you thought next.)
  - (EXPAND: the WSL local-vs-cloud A/B. The "wait, we can test compile for
    free" realization, and the confound where local vs cloud was really data
    drift, not torch.compile.)
  - (EXPAND: first time seeing 8 GPUs light up on Lambda, or the first coherent
    sample out of your own from-scratch weights.)
-->

## What I'd do differently

- **Verify the reference number before treating a gap as a bug.** The entire
  V1->V2->"audit" arc started from a wrong target (3.05) that was actually a
  40B-token, 4-epoch result, not the 10B script everyone runs. The real gap was
  0.11 nat, not 0.35. Confirming what `max_steps=19073` actually corresponds to
  (one epoch of sample-10BT) would have reframed everything before $55 went out
  the door. Always pin down what the number you're chasing was produced *with*.
- **Check data trajectory before blaming the runtime stack.** The "smoking gun"
  that local eager beat cloud compile by 0.18 nat at step 199 was data drift:
  single-GPU at B=16 covers ~3.3M tokens in 200 steps (all inside shard 1),
  while 8-GPU covers ~26M tokens across multiple shards. By step 199 they were
  literally training on different documents. Cross-setup trajectory comparisons
  are meaningless unless the data stream is controlled.
- **Run the cheap local smoke before the expensive cloud run.** The WSL single-
  GPU A/B (eager vs compile, 100 steps, $0) settled a question that a third
  $55-70 cloud run was lined up to answer. Local single-GPU validation is cheap,
  fast, and surprisingly informative; it should be standard before anything
  paid.
- **Make wandb runs public from the start.** Half the audit's leverage came from
  research agents being able to read the actual run logs. Private runs would
  have blocked that.
- **Treat upstream errata as load-bearing.** Three of the bugs came straight
  from the build-nanogpt README errata (DataLoader rank offset, autocast device
  type, the `load_tokens` dtype cast). They were real fixes even though none of
  them closed the headline gap. Read the errata before assuming your code is the
  problem.
- **Kill the dead causal-mask buffer.** Multiple audit agents independently
  flagged the `self.bias` lower-triangular buffer as 48MB of VRAM that SDPA
  never reads. Karpathy removed it when he switched to SDPA; we kept it by
  inertia. Small, but it's the kind of thing a clean reproduction shouldn't
  carry.

## Open questions

**Q: What is the actual reproduction variance for GPT-2 124M at 10B tokens?**

A: Unknown, and that's the honest answer. Karpathy reports a single run, so
there's no published spread to compare against. Our V1 and V2 came out
statistically equivalent (3.402 vs 3.397, HellaSwag 26.9% vs 27.3%), which is
one data point on intra-recipe variance but says nothing about how far a
"correct" reproduction can legitimately land from his number. A 0.11 nat gap
sits comfortably inside the plausible variance band, but proving that would take
several seeded runs nobody is going to pay for. The takeaway: at this scale,
"reproduce" means "land in the neighborhood," not "match to three decimals."

**Q: Why does WSL eager differ from Windows eager on the same 4090?**

A: WSL eager hit step-99 loss 7.187; Windows eager hit step-100 loss 7.240 on
the same card, same B=16, same seed. ~0.05 nat. Most likely cause is the
toolchain: different CUDA wheel builds, different compiled kernels
(glibc/Linux vs MSVC/Windows), or just a one-step trajectory offset. Not
investigated, because it doesn't affect the cloud result and cross-OS bit-
reproducibility was never a goal. Worth remembering if exact determinism across
operating systems ever does matter.

**Q: Could the batch geometry (B=16+accum=4 vs B=64+accum=1) alone explain
~0.05 nat in BF16?**

A: Plausible but untested. Different micro-batch sizes change the order and
grouping of BF16 reductions, and BF16's limited mantissa means reduction order
isn't associative the way FP32 is closer to being. So two mathematically
"equal" batch geometries can land slightly apart. It's a candidate for part of
the residual gap to Karpathy (who ran B=64+accum=1), but isolating it would take
a controlled run we didn't do.

**Q: Does the `wpe` init std of 0.02 vs OpenAI's 0.01 matter?**

A: Community evidence (build-nanogpt Issue #18, llm.c PR #832 with a Cohen's
d=1.44 effect on TinyShakespeare) suggests 0.01 is meaningfully better for the
learned positional embedding. But Karpathy also uses 0.02, so it can't explain a
gap *to him*. It's moot for SkyAI going forward anyway: the modern architecture
drops learned positional embeddings entirely in favor of RoPE, so there's no
`wpe` left to initialize.
