# 3B1B: Intro to Neural Nets

## What I'm building

Foundation. No code yet, just watching Grant Sanderson's *"Neural Networks"* playlist (4 videos, ~1 hour total) to build visual intuition for what a neural network is, what gradient descent does, and what backpropagation computes. The series ends with a transformer video that's heavier on intuition than math, included as a teaser for what's to come.

## Concepts I had to internalize

- **A neural network is layers of weighted sums plus nonlinearities.** Each "neuron" is a dot product followed by an activation. Layers chain these together. There is no magic; it's all matrix multiplication and elementwise functions.
- **Gradient descent is "walk downhill on the loss surface".** The loss is a function of all the parameters; the gradient points uphill; you step in the negative gradient direction. Repeat until you're at a low point.
- **Backpropagation is just the chain rule, applied through the computation graph.** Every parameter contributes to the loss through some chain of operations. Backprop computes how a small wiggle in each parameter would change the loss, by walking the chain in reverse.
- **The transformer (briefly).** Self-attention is "every token decides how much to listen to every other token, then averages them". The math is matrix multiplications and softmaxes; the *intuition* is associative recall.

## What surprised me

- Overall really interesting and in-depth videos, though some of the math behind certain algorithms, especially those in transformer models, is still confusing to me.
- The visuals surrounding gradient descent were actually really helpful and informative.
- The digit predictor/recognizer was also a really engaging and interesting way to introduce neural nets.

## What I should be doing differently

- **Watch this playlist first, before anything else ML.** I came in cold to Karpathy's GPT-2 video and bounced off the math; that was the wrong starting point. The 3B1B series + Karpathy's "Let's build GPT" should be the first two things, not the last.
- **Rewatch the transformer video after building one.** A lot of the visual intuition only lands after you've manually computed a self-attention block. The video will hit differently after building one in code.
- **Don't fight the abstraction on the first pass.** Some of the math is going to feel handwavy. Trust that it will get more concrete when you implement it. The intuition layer is doing real work even when the algebra is still fuzzy.

## Open questions

**Q: What's actually going on in the math behind transformers?**

A: The core operation is **scaled dot-product self-attention**:

1. For each token, compute three vectors: query (Q), key (K), value (V). These are just three different linear projections of the token's embedding.
2. For every pair of tokens (i, j), compute `Q_i · K_j` as the "score" of how much token i should pay attention to token j.
3. Scale by `1/sqrt(d_k)` (just to keep variance reasonable) and softmax across j to turn the scores into a probability distribution.
4. Each token's new representation is a weighted average of all the V vectors, weighted by those attention probabilities.

That's it. The "intelligence" emerges from training: the Q, K, V projection matrices get learned to make attention weights pick out the right relationships for whatever task. Stack a bunch of these layers, add an MLP between each one, and you have a transformer. We'll build all of this from scratch in Phase 1 of SkyAI, and that's when the math will stop feeling magical and start feeling mechanical.
