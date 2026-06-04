# Makemore: Bigrams

## What I'm building

A character-level bigram language model on ~32K names. Two parallel implementations: first by counting bigram frequencies in a 27x27 table and normalizing each row into a probability distribution, then by training a single-layer neural network (one weight matrix, no nonlinearity) that learns the same distribution via gradient descent. Both end up at roughly the same negative log-likelihood (~2.46), demonstrating that "counting" and "learning" can produce the same model for simple enough cases.

## Concepts I had to internalize

- **Bigram = P(next char | current char).** This isn't too far fetched from intro to probability. A 27x27 lookup table where row `i` is "what comes after character `i`". Every row is a probability distribution over the next character. Sort of hard to conceptualize this as probabilities.
- **One-hot encoding as input.** Each character becomes a 27-dim vector that's mostly zeros. Not 100% on if this is the most efficient way (definitely isnt for storage)?
- **Logits, softmax, probabilities.** Logits are unnormalized scores from the network. Softmax exponentiates them and divides by the row sum to produce a probability distribution.
- **Cross-entropy loss.** Average of `-log(probability assigned to the true next char)`. Lower is better. The bigram baseline gets ~2.46 NLL on this dataset.
- **Laplace smoothing (+1).** Adding 1 to every count avoids zero probabilities for unseen bigrams, which would otherwise produce `log(0) = -inf` and blow up the loss. Simple and effective trick.
- **Count-based and learned models can be equivalent.** A linear layer trained with cross-entropy on this task converges to the same distribution as the count-and-normalize approach. Gradient descent is doing the same thing as taking a closed-form average, just iteratively.

## What surprised me

- Wow this pytorch math is hard to wrap my head around. Also the "broadcastable" makes sense to me as an idea but I have no idea how that looks in practice.
- Pytorch just seems like a difficult framework to grasp - like the (T)ensor vs (t)ensor dogma.
- Im having a hard time visualizing the process of the neural net and the gradient descent. Using matrices and code instead of a literal picture makes it alot harder to understand and visualize what actually goes down behind the scenes. This is especially important for me because I feel like I don't understand something at all unless I know exactly what is going on (which is why I generally like low level languages).
- Don't know where the "intelligence" comes in because it just seems like a cool trick based on matrix multiplication. When you get a parameter count of 1 billion it just works?

## What I should be doing differently

- **Practice broadcasting in isolation before using it inside a training loop.** A 30-minute shape-prediction drill, or honestly just PyTorch practice in general
- **Always print intermediate tensor shapes.** When something feels wrong, the first move is `print(x.shape)`. The second is `print(x[0])` to look at one element by hand. Setting up good print statements is pretty helpful.
- **Always use lowercase `torch.tensor(...)` to construct.** `torch.Tensor` (capital) returns uninitialized memory. Pretend it doesn't exist outside `isinstance` checks.

## Open questions

**Q: What does broadcasting actually look like in practice?**

A: The rule: align shapes from the right. At each position, the dimensions must either match, be 1, or not exist (treated as 1). Concrete examples:

- `(3, 4) + (4,)` aligns right as `(3, 4)` and `(_, 4)`. Matches. The `(4,)` virtually expands to `(3, 4)`. Result: `(3, 4)`.
- `(3, 4) + (3,)` aligns right as `(3, 4)` and `(_, 3)`. 4 vs 3 mismatch, error.
- `(3, 1) + (1, 3)` both expand to `(3, 3)`. Famous bug source: when you wanted `(3, 1) + (3,) → (3,)` but accidentally wrote `(1, 3)` for the second operand.

The thing to watch for: broadcasting is silent. PyTorch will not warn you when shapes virtually expand, even if the expansion was not what you intended. And python being python will also not throw an explicit error, instead you will get a logic error deep down the road when your net turns out to be lobotomized.

**Q: `Tensor` vs `tensor`: what's the actual rule?**

A:
- `torch.Tensor(...)` (capital T) is the class. Calling it as a constructor gives you uninitialized garbage memory in the requested shape. Almost never what you want.
- `torch.tensor(...)` (lowercase t) is the factory function. It infers dtype from the input data and copies values in. This is what you want 99% of the time.
- `torch.Tensor` is useful only for `isinstance(x, torch.Tensor)` type checks, not for construction.

The PyTorch docs explicitly recommend `torch.tensor`. The capital version exists for backwards-compat reasons, not because it's good API. lol

**Q: How do I visualize what's happening in matrix ops?**

A: Two strategies, both worth using:

1. **Tiny toy examples.** Drop the vocab to 4 chars, the data to 5 examples, and trace one forward pass by hand on paper. Then run it in code and verify your hand-computed values match. 
2. **Print every shape, every intermediate value.** After each line, print `x.shape` and `x[0]`. When the shape doesn't match what you expected, you've found the bug. Slower than mental simulation but works in any environment.

The "I can't see what's happening" instinct from low-level languages is not a weakness for ML, it's a strength. ML practitioners also can't fully see what's happening at full scale. They just shrink the problem until they can. Honestly a solution to this could be super useful.
