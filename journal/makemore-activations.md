# Makemore: Activations, Gradients, BatchNorm

## What I'm building

The same MLP from the previous video, but instrumented and tuned. Two passes through the model: first with manual fixes (lower-confidence init for the output layer, Kaiming-style scaling for the hidden layer to keep tanh out of saturation, and a hand-rolled BatchNorm with running mean/std for inference). Then a second pass that rebuilds the network as a stack of `Linear` / `BatchNorm1D` / `Tanh` modules and goes 6 layers deep, with diagnostic tools (activation histograms, gradient histograms, weight-gradient ratios, update-to-data ratios) so I can actually see whether the network is healthy. Dev loss drops from 2.17 (last video) to ~2.10. The wins are small in number but came from understanding *why* the network was stuck before.

## Concepts I had to internalize

- **The initial loss spike.** With random init, the output logits have large variance, so softmax produces over-confident predictions on random classes. Cross-entropy on a confidently wrong prediction is huge. Step 0 loss should be `-ln(1/vocab_size) = ln(27) ~= 3.30`; if you see anything much higher, your output layer is too confident at init. Fix: shrink the output layer's weights and zero its bias. Now step 0 starts where it should.
- **Tanh saturation.** When pre-activations are too large (say, std ~ 5), tanh outputs land at ±1, where its derivative is essentially 0. No gradient flows back to the weights that fed those neurons. They become "dead" and never update. Fix: scale the input weights by Kaiming gain (`5/3` for tanh) divided by `sqrt(fan_in)`, which keeps pre-activations in a reasonable range.
- **Kaiming init isn't magic, it's variance preservation.** Each layer multiplies a vector by a matrix; if you don't scale the matrix's entries by `1/sqrt(fan_in)`, the variance of the output grows by a factor of `fan_in`. After a few layers, activations explode (or vanish if you scale wrong). The scaling rule keeps variance ~= 1 layer to layer.
- **BatchNorm: normalize, then re-scale with learnable parameters.** For each feature dimension, subtract the per-batch mean and divide by the per-batch std (keeps activations centered and scaled). Then multiply by `gamma` and add `beta` (both learnable) so the network can undo the normalization if it actually wants to. This decouples weight initialization from activation health.
- **Running mean/std for inference.** During training, BatchNorm uses the current batch's stats. At inference time you don't have a batch (or your batch is size 1), so you maintain an exponential moving average of mean/std during training and use those at eval. The "calibrate at end of training" cell is doing the same thing in one shot, but the EMA is the standard approach.
- **Diagnostic plots.** Three things to plot, every time you train a deeper net:
  1. **Activation histograms per layer.** If layers near the input are saturating and layers near the output are dead-zero, you have a propagation problem.
  2. **Gradient histograms per layer.** Same idea, but for gradients on the way back. If gradients vanish at deep layers, your network isn't learning past a certain depth.
  3. **Update-to-data ratio.** `(lr * grad.std() / param.std())` per parameter, log10. Healthy is around -3 (1 part in 1000 update relative to the parameter's scale). Above -2 means LR is too high; below -4 means too low.
- **Why this all matters more for deeper networks.** A 1-layer net mostly survives bad init because there's only one layer to mess up. Stack 6 of them and small per-layer issues compound multiplicatively. The whole video is a tour of "things that don't matter at depth 1 but kill you at depth 6 (or 96, like GPT-2)".

## What surprised me

- It seems like alot of this is just tinkering around with little variables/variations. Is there no concrete mathematic/probability backed optimization to this? Or is this technology too new to come to that conclusion? I'd assume not. Perhaps it could be because at scale we can't comprehend the "math" behind every parameter (GPT2 has 124 million)?
- Im noticing a lot of "bugs" are logic bugs and these can be extremely easy to miss and extremely hard to catch. Are there ways people can catch these other than just being a god at pytorch? Seems like missing little bugs is especially easy on such a high level language like Python - The Rust compiler would find any given oppurtunity to complain lol.
- There is no way starting at a better loss has this much of an effect on the outcome? Wouldnt a high loss be fixed faster anyways so it shouldnt matter?
- dead neurons in neural networks because of activation function inefficiencies... wow so these LLMs really are just like me.
- Thank god for batch normalization and these other init optimizations because individually tinkering with all these parameters/hyperparameters are absolutely behind me.
- Im also starting to notice the same amount of steps is taking alot longer to run...

## What I should be doing differently

- **Always plot the activation distribution per layer before declaring training done.** A deep net can converge to a "fine" loss while half its neurons are dead. The histogram tells you that; the loss number doesn't.
- **Check the initial loss against `-ln(1/vocab_size)`.** If step 0 loss is much higher than this baseline, your output layer is too confident at init. This is a 30-second diagnostic that catches a real bug.
- **Suspect tanh saturation when training stalls early.** If loss plateaus high in the first 1000 steps and the activation histogram shows tanh outputs piled up at ±1, the gradient highway is blocked. Lower the input-to-tanh weight scale.
- **Track update-to-data ratio during training, not just loss.** Loss curves can be misleading. The update ratio tells you if the LR is sane; if it's not, no amount of tuning other things will help.
- **Use `torch.nn` modules in real code.** The hand-rolled BatchNorm in the first half of the video is for understanding. In production, `nn.BatchNorm1d` is the same thing (with the running-mean EMA built in) and is what we'll use in SkyAI.

## Open questions

**Q: Is all this just empirical tinkering, or is there real math backing it?**

A: Honest answer: **mostly empirical, with theoretical scaffolding around the edges.** Specifically:

- **Init scales (Kaiming, Xavier, etc.) DO have real math.** They're derived from the requirement that the variance of activations stays roughly 1 as you propagate forward, and that gradient variance stays roughly 1 as you propagate backward. The `5/3` factor for tanh, the `1/sqrt(fan_in)` scaling, all of it falls out of variance arithmetic. This is one of the few places in deep learning where the theory is rigorous and the practice matches.
- **BatchNorm has weaker theoretical justification.** The original 2015 paper claimed it fixes "internal covariate shift", but follow-up work (Santurkar et al., 2018) showed that's mostly wrong. BatchNorm works empirically; the *reason* it works is still debated. The current best guess is that it smooths the loss landscape, but this is not nailed down.
- **Architectures are mostly designed empirically.** The transformer, CNN, ResNet, etc. were all proposed because someone tried them and they worked. Math came afterward to explain *why*. This is sometimes called "the bitter lesson": models that scale tend to win, and clever theoretical priors tend to lose.
- **At LLM scale, we mostly trust empirical scaling laws** (Kaplan et al. 2020, Chinchilla 2022): given compute and data, you can predict the loss to surprising accuracy. That's a *quantitative* law, not a mechanistic explanation.
- **Mechanistic interpretability is trying to retrofit math** to specific behaviors of trained models (induction heads, in-context learning, etc.) but this is research, not standard practice.

The 124 million parameters of GPT-2 is exactly the right intuition: at that scale, no one is reasoning about every weight. They're reasoning about distributions, scaling laws, and emergent properties. The "math" is more like statistical mechanics than calculus.

**Q: How do people catch logic bugs in PyTorch other than being a wizard?**

A: Several ways, in roughly increasing order of rigor:

1. **Print everything.** Shapes, means, stds, max/min after every line until the bug surfaces. Boring but works. The earlier shape-mismatch bug we hit (BatchNorm losing the batch dim) would have been caught instantly with `print(hiddenpre.shape)` after the assignment.
2. **Sanity check known invariants at training start.** Initial loss should equal `-ln(1/vocab_size)`. Output of softmax should sum to 1. Gradient signs should be sensible (a positive logit on the wrong class should have a positive gradient). These take 30 seconds to check and catch a huge fraction of bugs.
3. **Test on tiny examples.** A 4-character vocabulary, 5 examples, batch size 1. Run forward and backward by hand on paper, then compare to code output. If you can't match by hand, the code is wrong (or your understanding is wrong).
4. **Static type checking.** Tools like `jaxtyping` or `beartype` let you annotate tensor shapes and dtypes in function signatures, and a runtime checker enforces them. Less compile-time strict than Rust but in the same spirit. Once you've written `def forward(x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch vocab"]` with these libraries, shape mismatches fail loudly.
5. **Unit tests.** Tests for: shape correctness on each module, gradient flow (backward produces non-zero grads on every parameter), known-output reproducibility (seed + input -> exact loss). The `tests/` directory in SkyAI is set up for exactly this.
6. **Weights & Biases / Tensorboard.** Plot loss, gradients, activations, weights over time. A bug often manifests as a metric going somewhere weird before it manifests as bad loss. Visual catches are fast.

The "Rust compiler" comparison is fair, and the ML community knows it. There's active work on type-safer ML frameworks (Burn in Rust, Candle in Rust, JAX with shape polymorphism in Python). Nothing has fully replaced PyTorch yet because the ecosystem is too valuable, but the dissatisfaction is real and shared.

**Q: Why does starting at a better loss matter so much? Wouldn't bad init just get fixed in a few steps?**

A: Two reasons it matters far more than intuition suggests:

1. **Bad init wastes the early steps undoing badness, not learning the data.** Imagine starting at loss 27 because your output layer is wildly over-confident on random classes. The first 1000 steps of training are the network frantically dialing back the over-confidence. By the time it reaches "uniform predictions" (~3.30 for 27 classes), you've already burned through 1000 steps without learning anything about names. Compare that to starting at loss 3.30 (correct initial state) and immediately spending those 1000 steps actually learning bigram and trigram patterns.
2. **Some bad init is unrecoverable.** Saturated tanh means dead gradients in those neurons forever. There's no force pushing them back to a useful range; the gradient is zero, so the weight doesn't move. You can train for 200k steps and those neurons are still dead. The "fix it later" intuition assumes gradients can always flow; saturation breaks that assumption.

So it's not that bad init eventually converges to a worse final loss (sometimes it does, sometimes it doesn't). It's that bad init *wastes capacity* (dead neurons) and *wastes compute* (steps spent undoing noise). At GPT-scale, both of those are catastrophic. A network with 10% dead neurons is effectively 10% smaller.

**Q: Why is training getting noticeably slower?**

A: Three things stacking up:

1. **The PyTorch-style network is bigger.** The first half of the video used 12K parameters; the 6-layer rebuild has 47K. Per-step compute scales roughly with parameter count for the linear layers, so each step is ~4x more arithmetic.
2. **BatchNorm has nontrivial overhead per layer.** Each BatchNorm forward pass computes mean and variance across the batch (two reductions), then applies scale+shift. Backward is comparable. With BatchNorm after each Linear in a 6-layer net, you're doing 6 of these per forward and 6 per backward.
3. **The diagnostic tracking.** Computing `(lr * p.grad.std() / p.data.std()).log10()` for every parameter every step is itself expensive, and the histograms over activations/gradients are fairly heavy. Karpathy includes them for pedagogy; in a production training loop you'd compute them every N steps, not every step.

If you want training back to MLP-video speed, drop the deeper network back to 1-2 hidden layers and turn off the per-step diagnostics. But "training is getting slower as my model gets more capable" is the reality of all of deep learning. Welcome to the field.
