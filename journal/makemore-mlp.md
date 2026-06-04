# Makemore: MLP

## What I'm building

A multi-layer perceptron that takes the previous N characters (`block_size` of them) and predicts the next one. Each input character is mapped through an embedding lookup into a low-dimensional vector, those vectors are concatenated and fed through a tanh hidden layer, and a linear layer produces logits over the 27-character vocabulary. Trained with cross-entropy on the same names dataset as the bigram model. Reaches ~2.17 dev loss with reasonable hyperparameters, beating the bigram baseline of 2.46.

## Concepts I had to internalize

- **Character embeddings.** Instead of one-hot vectors, each char gets a learned low-dim vector. Similar characters end up close in the embedding space (vowels cluster, consonants cluster, etc.). Way more expressive than one-hot for the same parameter budget.
- **Block size = context window.** The model sees `block_size` previous characters when predicting the next. Bigger block size = more context = more parameters = potentially better predictions, with diminishing returns past 4 or 5.
- **Why tanh.** Squashes activations to (-1, 1). Without a nonlinearity between layers, multiple linear layers collapse to one linear layer (matrix multiplication is associative). The nonlinearity is what makes the network actually multi-layer.
- **Train/dev/test split (80/10/10).** Train on the train set, tune hyperparameters on the dev set, report final numbers on the test set. Without this, lower training loss doesn't mean a better model; it might just mean overfitting.
- **Stepped LR schedule.** Train hot (LR=0.1), then cold (LR=0.01). The hot phase makes fast progress; the cold phase fine-tunes. Beats constant LR almost always.
- **Hyperparameter sensitivity has a ceiling.** Block_size, n_embed, n_hidden, and LR all change the final loss noticeably, but the architecture itself caps how low you can go (~2.05-2.10 dev loss for vanilla MLP on this dataset). No amount of HP tuning gets you below the architectural floor.

## What surprised me

- Once again the Tensor based approach is hard to visualize. Understanding what goes on behind each variable/operation is confusing.
- I also started to notice that I am getting worse at managing the jupyter notebook, wondering if there are best practices for this.
- Don't understand why karpathy prefers mini batches, wouldnt simply training on all words be best?
- So the loss function only went down a little bit and this is considered good? And also says a 0 loss would just mean we memorized the dataset? How do we measure success?
- Learning Rate decay did have a positive effect on the MLP, but I had honestly assumed finding the "perfect" learning rate and keeping it the same throughout would be the best strategy. I wonder if/how dynamic learning rates are handled.

## What I should be doing differently

- **Set up the train/dev/test split before training, not after.** Without a dev loss to look at, "lower training loss" doesn't mean "better model". The first version of this notebook trained without a split and the result was meaningless until experiment 2 added one.
- **Track val loss inside the training loop, not just at the end.** Plotting train and val together over time shows when overfitting starts. With a single end-of-training number, you can't tell if you trained for too long.
- **Restart kernel + run-all is the first move when things feel weird.** Jupyter state collisions (variables redefined, classes redefined while old instances still in memory) are the #1 source of "but my code looks right" bugs.
- **Recognize when the wall is architectural, not config.** If you've tried five hyperparameter combinations and you're stuck around the same loss, more tuning isn't the move. Move on to the next architectural idea (BatchNorm, deeper net, attention). This is the lesson at the heart of makemore part 3.

## Open questions

**Q: Are there best practices for managing Jupyter notebooks?**

A: Yes, and they hurt at first because Jupyter is seductive (cells let you skip ahead, redefine state, mix exploration and code) but bites you for it. The practices that actually work:

1. **Restart and run-all often.** Once a session, ideally before any "this looks weird" debugging. Forces you to confront whether the notebook actually runs end-to-end.
2. **Keep cells linear.** Resist the urge to "go back and fix earlier cells" without re-running everything below. Out-of-order execution is the #1 source of state bugs.
3. **Save expensive intermediate state to disk.** When loading data takes 30 seconds, save the result to a `.pt` file and load from disk on subsequent runs. Don't redo expensive ops every kernel restart (or tank it if you have a 7800x3d).
4. **Refactor mature code into `.py` files.** Once a class is settled, move it to `src/skyai/` and import it. Notebooks are for exploration, not living code. Could look into this after im done with all of makemore.
5. **One concept per notebook.** When a notebook gets longer than 30-40 cells, it's usually doing too much. Split it. I am splitting by youtube video/episode

**Q: Why mini-batches instead of training on all words?**

A: Three reasons:

1. **Memory.** A full-batch gradient on the entire dataset means computing the loss on 228K examples at once. The forward activations and backward gradients alone can blow past GPU memory for any non-trivial model.
2. **Speed per step.** A mini-batch gradient step is much faster than a full-batch step. You can do thousands of mini-batch updates in the time it takes to do one full-batch update. Total gradient signal received per second is higher with mini-batches.
3. **Noise as regularization.** Each mini-batch's gradient is noisy (it's only an estimate of the true gradient over the full dataset). That noise actually helps the optimizer escape sharp local minima and find flatter, better-generalizing optima. This is why "stochastic gradient descent" (SGD) usually works better than vanilla full-batch gradient descent in practice.

The standard practice across all of deep learning is mini-batch SGD with batch size 32-512, depending on memory and dataset. Karpathy's 32 is at the small end; production training runs often use much larger batches with proportionally larger learning rates.

**Q: How do I measure "success"? What does loss actually mean, and why is 0 bad?**

A: This is one of the most important questions in ML and worth getting clear on now.

**What loss means in absolute terms:** cross-entropy loss has a concrete interpretation. A loss of `L` means the model assigns probability `e^(-L)` to the correct next character on average. So:

- Loss 3.30 ≈ probability 1/27 ≈ uniform random guessing (you've learned nothing)
- Loss 2.46 ≈ probability 0.085 ≈ the bigram baseline (what counting alone gets you)
- Loss 2.17 ≈ probability 0.114 ≈ what your MLP achieves
- Loss 0.00 = probability 1.0 = perfect prediction

**Why 0 is bad:** the model can ONLY achieve 0 loss on the training set if it has memorized every example. That memorization doesn't generalize. The dev loss will be high while the train loss is 0, which is the textbook signature of overfitting. The model has learned the training data, not the underlying pattern.

**Why "loss only went down a little" can be good:** what matters isn't the absolute drop but where you ended up:

- Going from 2.46 → 2.17 doesn't sound like much, but it's a 30% increase in average probability assigned to the correct character. That's a real improvement in modeling ability.
- More importantly, train loss and dev loss should track each other. If train drops to 1.5 and dev stays at 2.5, you're overfitting. If both drop to 2.17, you're learning the actual structure.

**How we actually measure success:** the standard approach is:

1. **Watch dev loss, not train loss.** Train loss can be driven arbitrarily low by overfitting; dev loss measures generalization.
2. **Compare against baselines.** Bigram = 2.46 is the "you've learned the simplest possible pattern" floor. Beating it meaningfully (~2.17) means you've learned something the bigram model can't capture.
3. **Eventually, downstream task metrics.** For a name generator, the question becomes "do the generated names look name-like?" For a translation system, BLEU score. For a code model, did the test pass. Loss is a proxy for what you actually care about.
4. **Compare to known ceilings.** For language modeling, there's a theoretical lower bound (the entropy of the data itself). You can never get below that, no matter how good your model.

For SkyAI's eventual GPT-2 reproduction, we'll track dev loss on FineWeb-Edu validation set, plus HellaSwag accuracy as a downstream task metric. Multiple signals, not just one number.
