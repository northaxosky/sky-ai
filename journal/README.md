# Journal

Module-by-module notes capturing what I learned, where I got stuck, and the moments where things clicked, while building SkyAI.

## Why this exists

The code in this repo is one artifact of the learning process. The journal is the other. It's meant to be read alongside the code, in module order, to follow the reasoning that shaped each design choice.

If you came here to evaluate this project, start with the journal. The code shows what I built; the journal shows what I understood.

## Structure

One markdown per module, numbered by build order:

```
01-tokenizer.md
02-attention.md
03-transformer-block.md
04-positional-encoding.md
05-training-loop.md
06-optimizer-and-schedules.md
07-mixed-precision-and-flash-attention.md
08-data-pipeline.md
09-full-training-run.md
10-evaluation.md
```

(Numbering is approximate — it'll shift as the work unfolds.)

## What's in each entry

Loose template, not strict:

- **What I'm building** — the module under construction
- **Concepts I had to internalize** — the math/intuition that took real work
- **What surprised me** — bugs, unintuitive behavior, "wait, why does this work?"
- **What I'd do differently** — design choices I might revisit
- **Open questions** — things to come back to later
