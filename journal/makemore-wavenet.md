# Makemore: Wavenet

## What I'm building

Two things at once. **(1) Refactor**: rewrite the makemore codebase as a stack of PyTorch-style module classes (`Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `FlattenConsecutive`, `Sequential`) that mirror how real PyTorch code is structured. **(2) New architecture**: instead of flattening the whole context at the input, build a WaveNet-style hierarchy where consecutive character embeddings are paired layer by layer (8 chars → 4 pairs → 2 pairs-of-pairs → 1 root). The tree structure lets the model build up context gradually and use parameters more efficiently. Context bumped from 3 to 8 chars, and with tuned hyperparameters (`n_embed=24`, `n_hidden=128`) the dev loss drops to **~1.99** — the first time in the series it breaks below 2.0.

## Concepts I had to internalize

- **PyTorch-style module pattern.** Every layer class has the same shape: `__init__` stores parameters, `__call__` runs the forward and saves output to `self.out`, `parameters()` returns the list of trainable tensors. This is essentially `torch.nn.Module` rebuilt from scratch in ~10 lines per class.
- **The `self.out` attribute pattern.** Each layer stashes its forward output on itself so downstream inspection code (`for layer in model.layers: print(layer.out.shape)`) works without any extra plumbing. Real `nn.Module` does the same thing via forward hooks; this is the simpler version.
- **`Sequential` as a chain of `__call__`s.** Forward pass walks the list of layers calling each one on the previous output. `Sequential.parameters()` flattens parameters from all sub-layers so the optimizer sees a single flat list. Same API as `torch.nn.Sequential`.
- **`FlattenConsecutive(n)`.** Given `(B, T, C)`, reshape to `(B, T//n, C*n)`. This "fuses" n consecutive timesteps into a single timestep with n times the channels. With `n=2`, repeated application builds the wavenet binary tree: `(B, 8, C) → (B, 4, 2C) → (B, 2, 4C) → (B, 1, 8C)`.
- **Hierarchical processing > flat processing for sequences.** A vanilla MLP at block_size=8 has to digest all 8 chars at once in the first hidden layer. The wavenet splits this up: pairs of adjacent chars first, then pairs of pairs, etc. Better inductive bias for sequence data, fewer parameters needed.
- **BatchNorm1d on 3D tensors.** With 2D input `(B, C)`, you reduce over `dim=0` for stats. With 3D input `(B, T, C)`, you reduce over `dim=(0, 1)` — both batch and time get averaged for the per-feature mean/std. Per-feature running stats still have shape `(C,)`. The class needs to dispatch on `x.ndim` to pick the right reduction dim, which is exactly what Karpathy adds in this video. Subtle but important: forgetting to handle T correctly silently produces wrong gradients.
- **Training mode vs eval mode for BatchNorm.** During training, BN uses per-batch mean/std. During eval (sampling, val loss computation), BN uses the running_mean / running_var EMA tracked during training. The `for layer in model.layers: layer.training = False` cell is what flips the switch. Forgetting it gives the BatchNorm-at-inference bug.

## What surprised me

- I did not expect changing the context size from 3 -> 8 to make this much of a difference on the loss function. What's the point of no return on increasing context size? I assume you wouldnt wnat a bigger context size then the avg length of words in the dataset? What happens? Overfitting?
- I am realizing alot of "proficiency" comes from understanding and knowing PyTorch well. PyTorch seems to be one of the most thoroughly made libraries I have seen and being good with PyTorch seems like the #1 skill.
- I wonder how validation is handled on extremely large networks. You dont have the luxury of training multiple different times I assume.
- Is this the best it gets without going into transformer models? Arent there more inbetweens? Or are they not really applicable for this sort of task
- I really want to learn more about the "experimental harness" karpathy is talking about. Seems like it is critical for hardening/validating a neural network.
- Forgot that layer.out only exists after a forward pass, and since I didnt run that cell I ran into issues. Once again making me more of a fan of using jupyter as a scratch notebook and actual python files for complete implementations

## What I should be doing differently

- **Mirror `torch.nn.Module` when writing real models.** Even if you're not subclassing `nn.Module` directly, the pattern (store params in `__init__`, run forward in `__call__`/`forward`, expose `parameters()`) makes your code immediately understandable to anyone who knows PyTorch. SkyAI's actual model will inherit from `nn.Module`, but the mental model is the same one you built here.
- **Always switch BatchNorm to eval mode before inference / val loss.** The `layer.training = False` pattern (or `model.eval()` in real PyTorch) is non-optional. Loss looks fine in training but samples look like garbage = you forgot this. It's the most common BN bug; we hit it once in the activations notebook and once in backprop. Build the muscle memory now.
- **Run one forward pass before any inspection code.** Anything that reads `layer.out` (shape inspection, activation histograms, gradient diagnostics) depends on `__call__` having run at least once. If your inspection cell is in the notebook before training, throw `model(Xtr[:1])` at the top.
- **Pick context length based on the task, not blindly bigger.** For names (avg length ~7), block_size=8 is well-matched. For real LLMs (GPT-2 uses 1024, GPT-4 uses 8k-128k), the choice is driven by the typical document/conversation length you want to handle. Bigger context = more compute per token + more chance for the model to memorize irrelevant noise.
- **Look at `model.layers[i].out.shape` after the first forward pass for any new architecture.** Sanity-checks that your dimension arithmetic is right. Way faster than running training and finding out 100 steps later that the final layer collapses to the wrong shape.

## Open questions

**Q: What's the point of diminishing returns on increasing context size?**

A: For *this* dataset (32K names, avg word length ~7), the answer is roughly **right at block_size=8**. Beyond that you'd see:

- **Mostly-padding inputs.** With block_size=10 and avg word length 7, most of the context is `.` tokens (padding from before the word started). These don't carry information; you're just feeding noise.
- **Diminishing returns on what's there.** The first few chars of a name predict the rest reasonably well; the *first* char alone gets you most of the way. Adding more left context past 5-6 chars buys very little for names specifically.
- **Risk of overfitting** with more parameters chasing the same training signal.

For *real* LLMs, the answer is much higher because the task is harder:
- **GPT-2**: 1024 tokens
- **GPT-3**: 2048 tokens
- **GPT-4**: 8k-128k tokens (depending on variant)
- **Claude 4.7 / Gemini 2.5 / GPT-5**: 200k-1M+ tokens

The "right" context length depends on the typical *meaningful dependency length* in your data: how many tokens back does information you need to predict the next token typically live? For names, ~7. For news articles, ~500. For code with cross-file dependencies, 10k+. For long conversations with memory, 100k+.

Going past the meaningful dependency length wastes compute (attention is O(n²) in context length) but doesn't really hurt accuracy until you hit overfitting. The compute cost is usually what stops you, not loss degradation.

**Q: Is being good at PyTorch the #1 ML skill?**

A: It's *a* top skill, especially for the current era. Honestly though, the more nuanced answer:

- **PyTorch dominates** research (~90%+ of ML papers use PyTorch) and increasingly production. Knowing it well is genuinely high-leverage.
- **But the people who go beyond average PyTorch usage know what's underneath**: CUDA kernels, memory hierarchy, profiler output, distributed training primitives. Karpathy's whole career is built on "I can use PyTorch *and* I know what each line is doing at the hardware level".
- **The complementary skill is the layer above PyTorch**: HuggingFace transformers, Lightning, FSDP / DeepSpeed, Triton, Weights & Biases. Production ML is usually orchestration of higher-level libraries.
- **What's genuinely scarce**: people who can debug *why* training is slow, *why* gradients are exploding, *why* a particular kernel is the bottleneck. That requires understanding multiple layers of the stack.

So: yes, deep PyTorch knowledge is very valuable. But the engineers who command serious salaries (or get into top PhDs) usually know PyTorch *and* one layer above *and* one layer below.

**Q: How is validation handled on extremely large networks where you can't retrain multiple times?**

A: You're right that you can't just retrain. Several patterns substitute:

1. **Hold-out validation tracked continuously during training.** Standard practice: every N steps, run inference on a fixed val set and log the loss. Watch the curve as training progresses; intervene if val starts diverging.
2. **Multiple downstream eval datasets.** Don't just track loss. For LLMs: HellaSwag, MMLU, ARC, GSM8K, HumanEval, etc. Different eval datasets measure different capabilities. SkyAI will use HellaSwag.
3. **Checkpoint snapshots + post-hoc evaluation.** Save model state every few hours. After training ends, evaluate each checkpoint on a battery of tasks. Pick the best one. This is what "training scaling laws" papers do.
4. **Smaller models as proxies.** Karpathy's GPT-2 124M is partly a proxy for what GPT-2 XL would do; if your tricks help at 124M, they probably help at 1.5B. Lab teams routinely train 10-100M models to test recipes before committing to 100B+.
5. **Eval on subsets of training data ("training probes").** Take 1000 examples from train, hold them out, evaluate periodically. Tells you if the model is converging on the *kind* of thing it's supposed to learn.
6. **Linear probes.** Freeze the model, train a tiny linear classifier on its representations for a downstream task. Measures representation quality without full fine-tuning.
7. **Internal benchmarks before public release.** OpenAI, Anthropic, DeepMind all run extensive internal eval suites that are never published. Public benchmarks are the visible tip.

The bigger the model, the more you rely on (1), (2), and (4). You don't get many shots; you make each shot count.

**Q: Is wavenet the best non-transformer architecture? Are there in-betweens?**

A: Wavenet is *one* non-transformer architecture, not "the" best. The landscape:

- **MLPs**: what we built. Simple, works for tiny tasks.
- **RNNs (vanilla, LSTM, GRU)**: process tokens one at a time, maintain a hidden state. Dominated sequence modeling pre-2017. Slow because they can't parallelize the time dimension during training.
- **CNNs (1D, dilated, WaveNet)**: process sequences with convolutions. Faster than RNNs because they parallelize. Limited receptive field.
- **Transformers**: introduced 2017, won most sequence tasks by 2019. Attention lets every token look at every other token. Quadratic cost in context length is the downside.
- **State-space models (Mamba, RWKV, S4)**: 2022+ research, trying to combine RNN-like O(n) cost with transformer-like quality. Promising but not yet displacing transformers at scale.
- **Hybrid architectures**: mixing attention with state-space (e.g. Mamba-Transformer hybrids, Jamba). Active area.

For name generation specifically, **transformers are overkill**. WaveNet is actually a great fit because the task has fixed-length context and local dependencies. Transformers shine when long-range dependencies matter (essays, code).

**Q: What is the "experimental harness" Karpathy mentions?**

A: It's the surrounding infrastructure that lets you run many experiments efficiently, compare them, and not lose your mind. Components:

1. **Config-driven runs.** A YAML or Python config file declares every hyperparameter. `python train.py --config configs/exp42.yaml`. No magic numbers in the training script.
2. **Experiment tracking.** Weights & Biases (or Tensorboard, or MLFlow). Every run logs its config + loss curves + metrics + checkpoints. You can compare runs side-by-side in a UI without rewriting code.
3. **Reproducibility primitives.** Seeded random number generators, deterministic kernel choices, pinned package versions. Same config + same code = same loss curve.
4. **Hyperparameter sweep tools.** W&B Sweeps, Optuna, or custom shell scripts. Define a search space ({lr: [1e-4, 1e-3, 1e-2], n_hidden: [200, 400, 800]}) and the tool runs all combinations.
5. **Automated eval.** When a training run finishes, automatically run it through HellaSwag, sample generations, and post the results to your dashboard.
6. **Cluster orchestration.** For multi-machine training: Slurm, Kubernetes, or cloud-specific tools (AWS SageMaker, GCP Vertex). Mostly relevant past single-node scale.

For SkyAI specifically, we'll build a lightweight version of (1)-(4) during Phases 3-5:
- `train.py` driven by a config dict
- W&B logging from day one
- Seeded reproducibility
- Maybe one or two grid-search runs near Phase 5

We don't need cluster orchestration. The 4090 is one machine and Lambda 8xA100 is one cloud machine; orchestration is overkill at that scale. But the muscle memory of "configure, run, log, compare" transfers directly to any larger setup.

This is genuinely a high-value skill — most ML practitioners are mediocre at this. Getting good at experiment management is what separates "I trained a model" from "I systematically explored a model's design space and made principled choices".
