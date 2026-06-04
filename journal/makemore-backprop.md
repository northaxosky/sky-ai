# Makemore: Backpropagation

## What I'm building

The same MLP from the activations video, but with the autograd training wheels off. The notebook walks through three exercises in order: (1) for every intermediate variable in the forward pass, manually derive `dL/d<variable>` by hand and verify against PyTorch's autograd via a `cmp` helper; (2) collapse cross-entropy and batch normalization into single-equation backward forms (the elegant closed-form expressions that real frameworks use internally); (3) train the full network end-to-end using *only* the hand-derived gradients with autograd entirely disabled (`with torch.no_grad():`). If the manual derivations are correct, the network trains identically to the autograd version.

## Concepts I had to internalize

- **Multivariate chain rule.** When a forward variable feeds *multiple* downstream consumers (e.g. `bndiff` is used to compute both `bndiff2` and `bnraw`), its gradient is the **sum** of contributions from every path. Missing one path produces a "close but wrong" answer (small nonzero `maxdiff`) which is the hardest kind of bug to catch.
- **`retain_grad()` for non-leaf tensors.** PyTorch only populates `.grad` on leaf tensors (parameters) by default. To `cmp` your manual gradient against autograd's for an intermediate variable, you have to explicitly call `retain_grad()` on it before `loss.backward()`. Forgetting one retain_grad gives a confusing warning + bogus comparison.
- **Exact vs Approximate equality.** Bit-identical results across two different orderings of floating-point operations are rare. If your manual gradient matches autograd's *up to* `maxdiff ~ 1e-9`, that's success — the small noise is from the math being computed in a slightly different order. Anything larger (e.g. `1e-3` or higher) is a real derivation bug.
- **Cross-entropy backward in one expression.** Instead of going through `logprobs -> probs -> counts -> norm_logits -> logits` step by step, the algebra collapses to: `dlogits = softmax(logits) - one_hot(targets)`, divided by batch size. Cleaner, faster, and what `F.cross_entropy` does internally.
- **BatchNorm backward in one expression.** The full BN backward (going through `bnmeani`, `bndiff`, `bndiff2`, `bnvar`, `bnvar_inv`, `bnraw`) collapses into a single formula for `dhidden_prebn`. Same trick: derive the math once, skip all the intermediate scratch variables.
- **Training with autograd disabled.** `with torch.no_grad():` shuts off autograd entirely (no graph built, no `.grad` populated). The whole training loop becomes "compute forward values, compute manual gradients, update parameters by hand". Matches autograd training behavior if and only if the manual derivations are correct.
- **Bessel's correction (n vs n-1) matters in batchnorm.** `var(unbiased=True)` (PyTorch default for `.var()`) divides by `n-1`. `nn.BatchNorm1d` uses biased variance (divides by `n`). The discrepancy is small in practice but causes `Approximate: True` rather than `Exact: True` when comparing.

## What surprised me

- I dont know how I will every be able to "manually" backpropogate and calculate derivatives of a neural net. I know PyTorch is meant to handle this, but doing it manually seems impossible, even if you know on paper how a matrix/etc's derivative is found.
- did the batchnorm1D biased vs unbiased discrepancy get fixed? This seems like a major logic issue that should be addressed.
- So does PyTorch essentially store the backwards() internally in a similar way we did for micrograd? Or does it use a different system since its much more sophisticated/extensive?
- There is no way I could have done the manual backprop by myself. Would have taken me days to calculate that.
- Instead of manually deriving every operation, constructing a mathematical expression and finding its derivative seems much more feasible and smarter. How does PyTorch actually do this?
- This is the video/concept I feel the least confident on. How is this handled in production? How critical is this?

## What I should be doing differently

- **Set up `retain_grad()` on every non-leaf intermediate before `loss.backward()`.** The list is long (logprobs, probs, counts, counts_sum, counts_sum_inv, norm_logits, logit_maxes, logits, hidden, hidden_preact, bnraw, bnvar_inv, bnvar, bndiff, bndiff2, hidden_prebn, bnmeani, embedcat, embed). Missing one breaks the cmp for that variable AND every variable downstream of it.
- **For each gradient derivation, identify ALL downstream consumers in the forward pass first.** Before writing any math, list the variables that depend on the one you're differentiating. The number of consumers tells you how many gradient paths sum into your answer.
- **Use the simplified one-go forms for cross-entropy and batchnorm in real code.** Even when autograd is doing the work, the closed-form is faster (fewer intermediate tensors allocated) and numerically more stable. PyTorch's `F.cross_entropy` and `nn.BatchNorm1d` already use these internally.
- **Always copy the BatchNorm transform from training to inference.** The most common ML production bug: training-time BN normalizes per batch, inference can't. You need stored running mean/std (or end-of-training calibration) and the same BN line in your sampling forward pass.
- **When samples look weird but loss looks fine, the bug is almost always in the inference forward pass.** Loss is computed during training where everything is correct; samples are generated separately, often with a slightly different code path that's easy to get wrong.
- **Restart kernel + run all when state feels wrong.** Stale `.grad` values, stale `bnvar` from the last training batch, stale variable bindings — Jupyter accumulates these and they're invisible until they bite. Restart-and-run-all is the seatbelt.

## Open questions

**Q: Did the BatchNorm1d biased vs unbiased discrepancy get fixed?**

A: No, it's a known quirk that's not really a bug. Two pieces:

- **`torch.var(x, 0, unbiased=True)`** (the PyTorch default for `.var()`) divides the sum of squared deviations by `n-1`. This is "Bessel's correction" and gives an unbiased estimate of the population variance from a sample.
- **`nn.BatchNorm1d`** internally uses **biased variance** (divides by `n`) for the per-batch normalization step during training. It does this for performance and because the activations themselves aren't a "sample" of anything in the statistical sense.

Karpathy's notebook uses `unbiased=True` because it's matching against the autograd of the explicit per-step formula (which uses `bndiff2.sum() / (n-1)`). When comparing against `nn.BatchNorm1d`'s output, you'd want `unbiased=False` to match exactly.

The numerical difference is `n / (n-1)` on the variance, which approaches 1 as batch size grows. At batch size 32, that's `32/31 = 1.032` — about 3% off. Negligible for training loss; would only matter in extreme edge cases. Most production code uses `nn.BatchNorm1d` and never thinks about it.

**Q: Does PyTorch store backwards() internally like micrograd does?**

A: Yes, conceptually identical. Different implementation details:

- **Micrograd**: each `Value` stored a `_backward` lambda (a closure) that knew how to compute its op's backward. `loss.backward()` did a topological sort and called each `_backward` in reverse.
- **PyTorch**: each tensor with `requires_grad=True` has a `grad_fn` attribute pointing to a `Function` object (e.g. `MulBackward0`, `AddmmBackward0`). The Function knows the op's local backward formula. `loss.backward()` walks the autograd graph backward, calling each Function's `backward()` method.

The differences vs micrograd:
1. **C++ backend.** All the heavy compute and graph traversal is in C++/CUDA. Python is just the orchestration layer.
2. **Function objects, not closures.** Each op has a separate Function class registered globally. Cleaner memory management, thread-safe, easier to extend with custom ops.
3. **Operates on tensors, not scalars.** One Function call corresponds to a tensor op (potentially millions of underlying scalar ops fused into a single CUDA kernel).
4. **Heavy optimizations**: kernel fusion, async dispatch, in-place ops, view tracking, gradient accumulation, etc.

But the algorithm is exactly micrograd: build a DAG during forward, walk it in reverse during backward, multiply local gradients via chain rule. Reading the PyTorch source for `torch.autograd.Function` is genuinely just a more sophisticated version of what you wrote in micrograd.

**Q: How is manual backprop handled in production? Is this critical to know?**

A: Almost no one writes manual backward passes in production. The exceptions:

- **Custom CUDA ops**: writing a new low-level kernel (e.g. a fused attention kernel like Flash Attention) requires you to implement both forward and backward in C++/CUDA. PyTorch's autograd can't infer the backward for your custom op.
- **Gradient debugging**: when training produces NaN gradients, exploding gradients, or dead neurons, you sometimes need to hand-trace what the gradient should be at each step to find where it goes wrong.
- **Performance-critical hot paths**: if a specific op shows up as a bottleneck, hand-fusing forward+backward into a single kernel can be a real win. Most people use Triton or `torch.compile` to do this automatically rather than writing it by hand.
- **Research where you need a non-standard gradient**: e.g. straight-through estimators, custom loss landscapes, gradient surgery. Rare.

For 99% of work, you import a model from `torchvision` or HuggingFace, define a loss, call `.backward()`, and never think about it. The reason this video exists is so you understand what's happening, which helps when (not if) something breaks. **You do NOT need to be confident at hand-deriving the backward of every op. You DO need to be confident that you know what autograd is doing at a conceptual level.** The backprop ninja video is "look how complicated it would be without autograd"; the takeaway is "thank god for autograd".

This is the video to feel least-confident-but-not-worried about. Move on.

**Q: How does PyTorch actually compute derivatives? Symbolic differentiation vs operator-level autograd?**

A: PyTorch uses **operator-level autograd**, not symbolic differentiation. Each op has a hand-written backward; the framework strings them together via the chain rule.

The alternatives:

- **Symbolic differentiation**: take a mathematical expression as input, output a new expression representing its derivative. Mathematica does this. Limited to algebraic expressions; can't handle Python control flow (if statements, loops) cleanly.
- **Forward-mode autograd**: compute derivatives during the forward pass by carrying around dual numbers. Efficient when there are few inputs and many outputs (e.g. computing Jacobians of a simple function). Inefficient for typical ML where there are many parameters and one scalar loss.
- **Reverse-mode autograd** (what PyTorch uses): build the computation graph during forward, traverse it in reverse during backward. Efficient when there are many inputs (parameters) and one output (loss). The right tradeoff for deep learning.
- **Tracing-based JIT differentiation**: trace the function once, then differentiate the trace symbolically. JAX uses this. More elegant for math-heavy code, less elegant for dynamic control flow.

So when you wonder "why doesn't PyTorch just take my function and differentiate it like calculus" — that's symbolic differentiation, and it doesn't scale to functions with loops, conditionals, and millions of parameters. Reverse-mode autograd on a per-op basis is the engineering compromise that actually works at scale. JAX is the modern attempt to combine the elegance of symbolic differentiation with the practicality of operator-level autograd, but PyTorch's per-op approach is what dominates in industry.
