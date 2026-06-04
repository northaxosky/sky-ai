# Let's Build GPT

## What I'm building

Following Karpathy's *"Let's build GPT: from scratch, in code, spelled out"* (Zero to Hero, video 7). A character-level GPT trained on Tiny Shakespeare. By the end of the video the model is a full transformer: multi-head causal self-attention, residual connections, LayerNorm, dropout, and a feed-forward network. Architecture matches GPT-2 in shape but is much smaller (`n_embed=384, n_head=6, n_layer=6`, ~10M parameters, `block_size=256`). Train loss ~1.1, val loss ~1.5 after 5000 steps on a 4090. This is the prereq closest to SkyAI Phase 1 in structure: a clean nn.Module-based transformer trained on a real dataset.

Also the first prereq where I dropped Jupyter for actual `.py` files. Workflow shifted to: `model.py` (defines the architecture, trains, saves a checkpoint when run directly) + `generate.py` (loads the checkpoint, samples without retraining).

## Concepts I had to internalize

- **Self-attention as "let every position decide how much to listen to every other position".** Three learned linear projections of the input (`Q`, `K`, `V`). `Q @ K.T` produces a `(T, T)` matrix of "affinities" between every query position and every key position. `V` provides the actual content to pull. The mechanism is just three matrices and a softmax.
- **Scaled dot-product (`/ sqrt(d_k)`).** Without this, the dot products grow with `d_k`, the softmax saturates (sharply peaked), and gradients vanish. Dividing by `sqrt(d_k)` keeps the pre-softmax variance ~1 regardless of embedding size. Non-obvious but load-bearing.
- **Causal masking.** Lower-triangular mask of `-inf` on the upper triangle before softmax. Ensures position `i` can only attend to positions `0..i`, never to future positions. Without it, training-time loss is meaningless because the model trivially sees the answer.
- **Multi-head attention.** Run `N` independent attention heads in parallel (each with `head_size = n_embd // N`), then concatenate their outputs and project back with a final linear. Lets the model attend to different things at once (one head attends to nearby chars, another to vowel patterns, another to long-range references, etc).
- **Residual connections.** `x = x + sublayer(x)` (not `x += sublayer(x)`, which breaks autograd). Lets the network train deeper without gradients vanishing. Information has a "skip path" around every sublayer.
- **LayerNorm (vs BatchNorm).** Normalizes per-token across the embedding dim, not per-feature across the batch. No running stats needed (so no train/eval mode required for it). Simpler in practice. Karpathy uses *pre-norm* (`x = x + sa(ln(x))`) which is the current best practice; the original Attention Is All You Need paper used post-norm.
- **Position embeddings (learned).** `nn.Embedding(block_size, n_embd)`, indexed by token position. Just another lookup table. Sinusoidal encodings (the original transformer paper) work too but learned wins in practice.
- **Dropout.** Randomly zero out activations during training; do nothing during eval. The `model.eval()` switch is what makes the difference, so you have to remember to call it before sampling.
- **The bigram baseline vs the full transformer.** Karpathy starts with a pure bigram model (just an embedding table) at ~2.5 loss, then iteratively adds attention, residuals, LayerNorm, and dropout. Each piece moves the dev loss down by ~0.2-0.4. This is a great mental model for "what does each transformer component buy you".
- **The `if __name__ == '__main__':` guard.** Lets a Python file double as both a runnable script and an importable module. Run it directly → trains. Import it from elsewhere → just exposes the class definitions, no training. Pattern is everywhere in real ML codebases.
- **Checkpoint save/load via `torch.save` + `state_dict`.** State dicts are the standard way to serialize a model's parameters (not the full Python object). `torch.save({'model_state': model.state_dict(), 'vocab_size': ..., 'stoi': ...}, path)` is the typical pattern. To load: instantiate the model, then `model.load_state_dict(ckpt['model_state'])`.

## What surprised me

- Oh my god, it really is just attention, attention really is all you need...
- Im not sure I understand how query and key work, as in how does it know how much weight to assign to each piece of context?
- It could honestly be real nice if the tensor outputs were color coded -> the softmax values brightness is determined by how probabilistic it is. Would help when inspecting a certain tensor and understanding its contents
- Softmaxing along the wrong dim broke causal masking in a way that didn't crash but let the model see the future. Loss looked too good; sampling looked too bad.
- What does karpathy mean by: "the tokens want to know the positions of other vowels, constants, etc...:? How does the query and key do that?
- Debugging errors on these larger less intuitive models seem very difficult, I can only imagine how impossible it is for 1B+ models...
- Dropouts seem counter intuitive; you train them as subnets so when you merge them wouldnt they perform worse since you didnt train them together?
- Need to set up an ideal IDE environment for neural network development - dont really have a python dev environment so keep running into little issues

*(EXPAND: the "transition from notebook to .py file" was a workflow shift in this video specifically. The interactive feel of Jupyter is gone, but you get separation of concerns (train vs generate vs model definition). Worth one sentence on whether the trade felt worth it.)*

## What I should be doing differently

- **Always softmax with `dim=-1` for attention scores.** The information-leak bug in this video (softmax along the query axis instead of the key axis) was silent. Loss looked unrealistically good; samples looked terrible. The rule from earlier prereqs ("use `dim=-1` whenever you mean the last/feature dim") compounds harder here because failure isn't a crash, it's silent wrongness.
- **Always write residual connections as `x = x + sublayer(x)`, never `x += sublayer(x)`.** The in-place version breaks autograd because sublayer saved a reference to the original `x` and the `+=` mutates it underneath. Burned an hour debugging "one of the variables needed for gradient computation has been modified by an inplace operation".
- **Always call `model.eval()` before sampling or running val loss.** Disables dropout. For BatchNorm models, also switches to running stats. Forgetting this is the #1 cause of "loss looks fine but samples are garbage".
- **Pre-norm LayerNorm (`x = x + sa(ln(x))`), not post-norm.** Karpathy uses this and it's what every modern transformer uses (GPT-2, GPT-3, Llama, etc). The original transformer paper used post-norm but it's harder to train deeply.
- **Anchor file paths to the script's directory using `Path(__file__).parent`.** Saves you when you run scripts from different working directories. Same fix applies to the Shakespeare data load and the checkpoint save.
- **For attention specifically, trace shapes by hand on paper with tiny dims** (T=4, head_size=2) before writing code. The `(B, T, T)` attention matrix is fiddly; getting the wrong `dim` for softmax or the wrong axis for `transpose` is easy. A 4x4 toy example takes 2 minutes to verify and prevents an hour of debugging.

## Open questions

**Q: How do query and key learn to assign the right attention weights?**

A: They aren't programmed; they emerge from training. Both `query = self.query(x)` and `key = self.key(x)` are *learned* linear projections. The model learns the weights of these projections through gradient descent on the next-token prediction objective.

Concretely: if predicting the next character at position 5 benefits from looking at position 2 (e.g., a `q` at position 2 implies `u` is likely next), training will nudge `query(x_5)` and `key(x_2)` toward vectors that have a high dot product. Specifically, the gradient flows back through the softmax and the dot product, telling each projection: "make these vectors more similar". Over thousands of training steps, the projections learn to align Q and K for *useful* relationships.

What's neat: different attention heads end up specializing in different relationships, completely unprompted. Heads might independently learn:
- "Attend to the previous N tokens" (positional pattern)
- "Attend to the matching open-bracket / quote" (syntactic pattern)
- "Attend to consonants that follow vowels" (phonetic pattern)
- "Attend to uppercase letters" (style pattern)

Nobody tells the model what to look for. The objective (next-token prediction) is the only signal. **This is the wild part of training transformers**: the attention patterns that emerge are interpretable and often map onto human concepts that nobody designed in.

**Q: Color-coded tensor visualization — does that exist?**

A: Yes! Several existing tools and a few quick recipes:

- **BertViz** (https://github.com/jessevig/bertviz) — the canonical attention-visualization tool for transformers. Renders attention weights as heatmaps, head-by-head, layer-by-layer. Open it in a Jupyter notebook with `from bertviz import model_view; model_view(model, ...)`. Works on any HuggingFace-compatible model.
- **`plt.imshow(weights, cmap='Blues')` for any 2D tensor.** Drop it in a notebook cell after computing attention weights:
  ```python
  import matplotlib.pyplot as plt
  plt.imshow(weights[0, 0].detach().cpu(), cmap='Blues')
  plt.colorbar()
  ```
  Brightness = magnitude. Works on any tensor.
- **For full-model probing**: TransformerLens (https://github.com/TransformerLensOrg/TransformerLens) lets you inspect every intermediate activation and attention pattern, with built-in visualization helpers.

**Q: What does Karpathy mean by "tokens want to know the positions of other vowels, consonants, etc"?**

A: He's describing what the network learns to do, in informal "anthropomorphizing the math" language.

Concretely: when predicting the next character, the current token (e.g., `'h'` at position 5) often needs to know where similar or related characters are in its context. Maybe it needs to attend to the previous vowel to figure out a likely word ending. Maybe it needs to attend to the last whitespace to know it's at the start of a word.

The query at position 5 is essentially asking *"give me anything in my past that satisfies this learned criterion"*. The keys at past positions answer *"I am this kind of thing"*. The dot product is highest when the query's criterion matches the key's "kind of thing". So "tokens want to know positions of other vowels" is shorthand for "the model learned a query/key pattern where the query at this position has high dot product with keys at vowel-character positions".

The learning isn't given the rule explicitly; the cross-entropy loss on next-character prediction implicitly pushes the projections toward whatever patterns help predict accurately. Vowel-tracking happens to be one such useful pattern.

**Q: Debugging at 1B+ scale — how is it even possible?**

A: It's hard, and there's a structured approach. The standard practice:

1. **Develop and debug at small scale (~10M to ~125M params).** Anything you don't understand at small scale will be 10x harder to understand at 1B+. Get the math, the optimizer config, the data pipeline, and the eval suite working at small scale first.
2. **Scale by trusting recipes that worked.** Once a 125M model trains cleanly to expected loss, scale it up via known scaling laws (Chinchilla, etc.). Don't experiment with new ideas at 1B+; experiment at 125M and scale only after validating.
3. **Heavy instrumentation.** Log gradient norms per parameter group, parameter norms, loss curves, learning rate, hardware utilization, etc. to W&B. When something goes wrong (NaN gradients, training stalls), one of these metrics usually catches it first.
4. **Snapshot frequently.** Save checkpoints every N hours. If training crashes or diverges, resume from the last good checkpoint rather than restarting.
5. **Small-batch sanity checks.** Periodically pause training, evaluate on a held-out probe set, sample some generations. If quality regresses, intervene.
6. **Pre-allocate budget for retries.** Real ML labs budget 1.5-2x the "ideal" cost because runs fail. The cost of a failed run is huge, so the planning explicitly accounts for it.

Frontier labs (OpenAI, Anthropic, DeepMind) have specialized infrastructure engineers whose entire job is making large training runs not silently break. It's a real specialty.

**Q: Why does dropout work? Don't the subnets fail to generalize when reassembled?**

A: Great question, and the "ensemble of subnets" framing is useful but a bit misleading. The actual mechanism:

During training:
- Each forward pass randomly zeros out a fraction of neurons (say 20%).
- The remaining neurons have to handle the loss alone — they can't rely on any single neuron being present.
- This *forces* the network to learn redundant, distributed representations. Multiple neurons end up encoding similar features so no single one is critical.

At inference:
- All neurons fire (no zeroing).
- The output is roughly the **expected value** over all possible dropout masks the model has been trained with.
- Because the model learned to work with any of those masks, the all-neurons-active state is a kind of "average" that the model handles well.

Why this doesn't degrade quality:
- The neurons aren't really independent subnets. They share parameters; every gradient update affects multiple "subnets" at once. So when you re-enable all neurons, they've been trained *together*, just with random subsets active at any given step.
- The redundancy enforced by dropout means the network can lose any small subset of neurons without falling apart. Generalization improves because the model can't memorize via one critical pathway.

Empirically: dropout reduces overfitting consistently across architectures. The theoretical justification is hand-wavy ("approximate Bayesian inference", "implicit regularization") but the practical effect is undeniable. Modern transformers use dropout sparingly (Karpathy uses 0.2 here; large LLMs often use 0.1 or less) because the implicit regularization of large data already provides much of the same benefit.

**Q: What's the ideal IDE setup for neural network development?**

A: For your stack (VS Code + Windows + uv-managed Python), here's the setup I'd build to:

1. **VS Code + Python extension + Pylance.** Set Pylance to `basic` for general work, `strict` for production code in `src/`. We covered this earlier.
2. **Jupyter extension** (already installed if notebooks render). Useful even when working in `.py` files thanks to `# %%` cells and the interactive window.
3. **GitHub Copilot or another AI completion tool**, but **disable it for notebooks during learning sessions** (it'll write your code for you and short-circuit the learning). Re-enable for production code in Phase 1+.
4. **GPU monitoring**: keep `nvidia-smi` (or `nvitop` for a prettier TUI: `uv pip install nvitop`) running in a separate terminal during training. Watch utilization, temperature, memory.
5. **W&B (Weights & Biases)** for experiment tracking. Already in your stack. Use it from Phase 3 onward.
6. **Terminal**: use PowerShell or Windows Terminal with a tab for VS Code's terminal, a tab for `nvidia-smi`, a tab for shell commands.
7. **`.vscode/settings.json` in the project root** for project-specific config:
   ```json
   {
       "python.analysis.typeCheckingMode": "basic",
       "python.testing.pytestEnabled": true,
       "files.exclude": {"**/__pycache__": true, "**/.pytest_cache": true}
   }
   ```
8. **Optional**: Cursor or Zed if you want a more AI-native editor. Both are VS Code-compatible (Cursor is a fork). Honestly VS Code is fine.
