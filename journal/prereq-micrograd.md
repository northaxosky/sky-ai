# Micrograd

## What I'm building

A scalar autograd engine from scratch (the `Value` class), plus a tiny multi-layer perceptron built on top of it, plus a training loop that drives gradient descent on a 4-example toy dataset. By the end, my hand-rolled autograd produces the same gradients as PyTorch's autograd on the same expressions, proving the math is right. About 150 lines total.

## Concepts I had to internalize

- **A computation graph is a DAG of operations.** Every intermediate value remembers (a) the operation that produced it and (b) the values that fed into it. Together this forms a graph from inputs/parameters through to the final loss.
- **Forward pass evaluates the graph; backward pass traverses it in reverse.** Forward computes values; backward computes gradients by applying the chain rule at each node.
- **Topological sort is the right traversal order for backprop.** You can't compute a node's gradient until all of its parents (in the forward direction) have already computed theirs. Topo sort guarantees this.
- **Each op has a local gradient rule.** `+` passes gradients through unchanged. `*` swaps the operands. `tanh` multiplies by `1 - tanh^2`. `**n` multiplies by `n * x^(n-1)`. The chain rule glues them all together.
- **Python's reverse-op protocol (`__radd__`, `__rmul__`, etc.).** When `int + Value` is evaluated, Python tries `int.__add__` first, fails, then tries `Value.__radd__`. Custom numeric classes that interact with built-in numbers must implement these.

## What surprised me

- I initially thought Neural Nets would store data as actual nodes with connections/edges. clearly not the case.
- Every time I work with python I am reminded why I love statically typed languages like C++
- The radd & add (and every other right and left operation) was suprising

## What I should be doing differently

- **When a class definition changes in Jupyter, restart the kernel.** Python objects don't update when their class is redefined. Old `MLP` instances built from the old `Layer` keep using the old broken behavior even after you fix the source. Restart-kernel-and-run-all is the safe default.
- **When a numeric type is acting weird in a builtin, suspect the reverse-op.** `int + Value`, `float * Value`, `sum([Value, ...])`, etc. all need the `__r*__` methods. The error message ("unsupported operand type") gives you exactly enough info to find it once you know what to look for.
- **Trace by hand on small examples.** When the math feels confusing, set up a graph with 3 nodes and a known answer. Compute forward and backward by hand. Then run it in code and verify they match. This is what Karpathy does on the whiteboard at the start of the video, and it works.

## Open questions

**Q: Does scalar autograd scale to real models?**

A: No, and that's the point of the next video. A scalar `Value` per number means O(N) Python objects per forward pass, and Python's interpreter overhead per object kills performance. PyTorch keeps the same chain-rule logic but operates on **tensors** (whole arrays at a time) instead of individual scalars. One tensor op = one autograd node, even if the tensor has a million elements. Same algorithm, vastly different constant factor.

**Q: Will I ever use the `Value` class directly in production code?**

A: Almost never. But understanding it is the cheapest way to understand PyTorch's `tensor.backward()`. Every `requires_grad=True` tensor is doing the same thing as the `Value` class, just on arrays. When you call `loss.backward()` in PyTorch, it's running the same topological-sort + chain-rule algorithm you wrote in micrograd, just on a bigger graph.

**Q: How do real frameworks decide which ops to implement?**

A: They implement primitives (add, multiply, exp, log, matmul, conv, etc.) and let users build everything else by composing them. Each primitive has a forward and a backward function, written in C++/CUDA. The framework's job is to track the graph and route gradients; the heavy compute is in the primitive ops. This is what we'll see when we start using PyTorch directly in makemore.
