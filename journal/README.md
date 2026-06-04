# Journal

Module-by-module notes capturing what I learned, where I got stuck, and the moments where things clicked, while building SkyAI.

## Why this exists

The code in this repo is one artifact of the learning process. The journal is the other. It's meant to be read alongside the code, in module order, to follow the reasoning that shaped each design choice.

If you came here to evaluate this project, start with the journal. The code shows what I built; the journal shows what I understood.

## Structure

One markdown per module, numbered by build order. Entries prefixed `00-prereq-*` cover the foundations I built up *before* SkyAI proper — Karpathy's Zero to Hero series, watched in order, journaled as I went:

```
prereq-micrograd.md
prereq-makemore-bigram.md
prereq-makemore-mlp.md
prereq-makemore-activations.md
prereq-makemore-backprop-ninja.md
prereq-makemore-wavenet.md
prereq-lets-build-gpt.md
prereq-gpt-tokenizer.md
tokenizer.md
attention.md
transformer-block.md
positional-encoding.md
training-loop.md
optimizer-and-schedules.md
mixed-precision-and-flash-attention.md
data-pipeline.md
full-training-run.md
evaluation.md
```

(Numbering is approximate — it'll shift as the work unfolds. Prereq entries are intentionally short — capture-the-reaction quick notes, not polished essays.)

## What's in each entry

Loose template, not strict:

- **What I'm building** — the module under construction
- **Concepts I had to internalize** — the math/intuition that took real work
- **What surprised me** — bugs, unintuitive behavior, "wait, why does this work?"
- **What I'd do differently** — design choices I might revisit
- **Open questions** — things to come back to later
