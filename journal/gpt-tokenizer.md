# GPT Tokenizer

## What I'm building

Following Karpathy's *"Let's build the GPT Tokenizer"* (Zero to Hero, video 8). Implementing Byte-Pair Encoding (BPE) from scratch on raw UTF-8 bytes: start with 256 single-byte tokens, iteratively find the most-common adjacent pair, merge it into a new token, repeat. After 20 merges on the Bee Movie script the compression ratio hits ~1.29x. Then add GPT-2's regex-based pre-tokenization (so BPE merges within "word-like" chunks, not across them), and finally compare against `tiktoken` (the production tokenizer used by GPT-2 and GPT-4).

The final notebook has a working toy tokenizer with `encode(text) -> [token_ids]` and `decode([token_ids]) -> text`, plus side-by-side `gpt2` vs `cl100k_base` (GPT-4) comparisons showing how the design choices differ.

This is the last prereq. After this: SkyAI Phase 1.

## Concepts I had to internalize

- **UTF-8 as the universal byte-level vocabulary.** Every Unicode codepoint encodes to 1-4 bytes in UTF-8. By starting BPE on raw bytes (vocab = 0..255), we get coverage of every possible string for free — no Unicode handling needed. Inefficient at first (every emoji is 4 tokens), but BPE merges fix that quickly.
- **BPE = greedy pair-merging.** Find the most common adjacent (token_a, token_b) pair, replace every occurrence with a new token id, repeat. Each merge adds one entry to the vocab. The list of merges (ordered) IS the tokenizer.
- **The merges dict is everything.** `{(p0, p1) -> new_token_id}` stores the training output. Inverting it gives you decode; greedily replaying it (earliest merges first) gives you encode. No model, no learning rate, no gradients — just dictionary lookups.
- **Encode picks the EARLIEST applicable merge, not the most common.** When applying the trained tokenizer to a new string, you replay merges in the order they were learned. `min(stats, key=lambda p: merges.get(p, +inf))` finds the pair with the lowest merge index (= learned earliest). This is the deterministic counterpart of the greedy "most common" training step.
- **Pre-tokenization via regex (GPT-2's contribution).** Before BPE, GPT-2 splits text into chunks using a regex like `r"'s|'t|'re| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"`. This prevents merging across logical boundaries (e.g. "the" + "," never becomes a single token). BPE then only merges WITHIN each chunk.
- **GPT-2 vs GPT-4 tokenizer differences.** GPT-4's `cl100k_base` merges consecutive whitespace into single tokens; GPT-2 doesn't. Different vocab sizes (50,257 vs ~100,000). Different regex patterns. These design changes reflect lessons learned (code with indentation tokenizes way better in GPT-4).
- **`regex` (third-party) vs `re` (stdlib).** Stdlib `re` doesn't support Unicode property escapes like `\p{L}` or `\p{N}`. The third-party `regex` package does. Standard practice: `import regex as re` so existing code keeps working but with the upgraded engine.
- **Special tokens.** Sequences like `<|endoftext|>` are reserved single-token ids inserted into the vocab. They get matched in the input via separate logic (not learned merges), then injected into the output. Crucial for chat templates, tool calling, etc.
- **Why tokenization affects model behavior.** "GPT-2 is bad at Python" was largely a tokenization issue (every space wasted a token). "GPT-4 is better at math" partly comes from better number tokenization. The tokenizer is invisible to model architecture but huge for downstream quality.

## What surprised me

- Really interesting to see how tokenization has changed and evolved. Specifically the pitfalls surrounding the whitespace, "\t" tokens that caused GPT to be quite bad at Python. Does this mean theoretically we could create a new "AI First" programming language that is optimized for LLM's tokenization and context windows? Theoretically we could also create a made up language since human languages like english can also run into slight issues with tokenization? Training data also plays a role here though. Or is the future a more robust handling of tokenizers. Seems like alot of room for improvement here. Maybe cut all the extra stuff, like supporting "multiple" languages, or emojis, and try to create a word based tokenization strategy?
- I wonder what actually determines how many merges we want. How do we measure, experiment and validate this?
- Tiktoken library written in rust - what a display of excellence
- When we train a model to be better at coding, better at chatting, or whatever specific exercise, is that actually just training the tokenizer to merge more optimally for a given set of inputs? Or is it a combination of both which is what I assume. Theoretically - couldn't you get more performance out of the same model & parameters & configurations if you "focused" its scope down, ie "training" for just code? Or is there a benefit to a general training?
- Looks like there isnt a "clean"/"best" tokenizer? Whats the reason behind that? Just that its way too fragile/variable?

*(EXPAND: this is the last prereq. One sentence on what crossing the finish line of the Zero to Hero series feels like would land well.)*

## What I should be doing differently

- **Use `tiktoken` for anything production-y. Never write your own tokenizer unless that IS the project.** For SkyAI, `tiktoken.get_encoding("gpt2")` gives you GPT-2's exact tokenizer (50,257 vocab). Zero work, identical to what HuggingFace's GPT-2 uses.
- **Always `import regex as re`, not `import re`, when working with text patterns.** The stdlib's lack of Unicode property escapes (`\p{L}`, `\p{N}`) bites silently — patterns "work" but match nothing.
- **Test encoding + decoding roundtrip on edge cases.** Emojis, multi-byte chars, repeated whitespace, code with mixed indentation. `encode(decode(tokens)) == tokens` should hold for any valid token sequence; if it doesn't, your tokenizer is broken.
- **When debugging tokenization, print BOTH the byte view AND the decoded string view.** Two interpretations of the same data; comparing them catches off-by-one and encoding errors fast.
- **Don't customize tokenization unless you have a strong reason.** The marginal gains from a "better" custom tokenizer rarely justify the ecosystem fragmentation (no shared pretrained models, no shared embeddings, every downstream tool needs your custom code).

## Open questions

**Q: Could you design an "AI-first" programming language optimized for LLM tokenization? Or a made-up human language?**

A: Yes, conceptually, but it would be barking up the wrong tree. The honest version:

- **For programming languages**: The reason GPT-2 was bad at Python wasn't that Python is poorly designed for LLMs — it was that GPT-2's tokenizer didn't handle indentation well. GPT-4 fixed this by retraining the tokenizer with more code in the corpus and learning whitespace merges. Python is *fine* for LLMs; the tokenizer was the problem. So the optimization target was the tokenizer, not the language.
- **Could you design a maximally-tokenizer-friendly language?** Yes. Common features would include: no significant whitespace, consistent syntax (every statement starts the same way), uniform identifier conventions, no ambiguous operators. But this would also be a worse language for humans, and we have ~50 years of programming language evolution that says optimizing for the wrong reader (humans, not tokenizers) is the design choice that matters.
- **For human languages**: tokenizers handle major languages pretty well after training. Where they struggle is low-resource languages (e.g. Swahili, Tagalog) that didn't get enough training data. The "fix" is more multilingual training data, not a new language.
- **The actual trend**: away from tokenizers entirely. Research on **byte-level models** (e.g. ByT5, MegaByte, recent Mamba variants) skips BPE and operates on raw bytes. The model has more capacity to spend but the brittleness of tokenization disappears. Not yet competitive with tokenized models at frontier scale, but trending.

So the answer is: tokenization friction is real, but solving it via "design a new language" is harder than just training a better tokenizer (or eventually, removing it).

**Q: How do you determine the right number of merges / vocab size?**

A: Mostly empirical with some scaling guidance. Standard values:

- **GPT-2**: 50,257 tokens
- **GPT-3.5 / GPT-4 (cl100k_base)**: ~100,256 tokens
- **GPT-4o (o200k_base)**: ~200,000 tokens
- **Llama 2**: 32,000 tokens
- **Llama 3**: 128,256 tokens
- **Mistral / Claude**: similar ranges, each with their own choices

The tradeoff:

- **Bigger vocab = fewer tokens per text** (more compression, shorter sequences, more text fits in context window) **BUT bigger embedding table and bigger output head**. Each row of the embedding is `n_embd` floats; doubling vocab doubles those parameter blocks.
- **Smaller vocab = more granular tokens** (longer sequences, more compute per text) **but smaller parameter overhead**.

For a frontier model (~100B+ params), the embedding/output cost is a small fraction of total params, so they go big. For a small model, you might use a smaller vocab.

**How to measure**: train your tokenizer on a corpus and report (1) compression ratio (chars/token), (2) token coverage (how many real-world tokens are unseen), (3) bytes per token on representative downstream data. Sweep over vocab sizes (e.g. 8k, 16k, 32k, 64k) and pick the knee of the curve.

For SkyAI: don't tune this. Use `tiktoken.get_encoding("gpt2")` (50,257 tokens) and inherit GPT-2's choice. The whole point of "reproducing GPT-2" is matching its setup.

**Q: tiktoken is written in Rust — what makes it different from a pure-Python BPE?**

A: Speed. Tokenization is CPU-bound, sequential per-token, and embarrassingly parallel across documents. Pure-Python BPE has Python interpreter overhead on every character lookup; Rust eliminates that overhead and uses optimized data structures.

Concrete numbers: a pure-Python BPE tokenizer might tokenize ~100k tokens/sec. `tiktoken` does ~5-10 million tokens/sec. ~50-100x faster. At training scale (10 billion tokens through the tokenizer), this is the difference between "tokenization takes 30 hours" and "tokenization takes 20 minutes".

HuggingFace's `tokenizers` library is also Rust for the same reason. The Python wrapper is via PyO3 (Rust ↔ Python bindings). Both libraries are great examples of "tight inner loops belong in Rust, glue belongs in Python".

When you scope a Rust agent later, this is a model to study — same pattern: Rust for performance-critical work, Python (or just CLI invocation) for orchestration.

**Q: Is "training a model for code" actually just training the tokenizer better? Could you get more performance from the same params by focusing scope?**

A: No, the tokenizer is a small part of it. The main levers are:

1. **Training data composition.** A code model is trained on a corpus heavy in code (GitHub, StackOverflow, etc.) and lighter in prose. A general model balances domains. Same architecture; very different data.
2. **Fine-tuning.** After pretraining, the model is often further tuned on a specialized dataset (instruction-tuning data, code completion data, RLHF data). Same weights initially; very different post-training.
3. **Tokenizer can help but usually doesn't dominate.** A code-optimized tokenizer (e.g. one that handles whitespace well) gives ~10-20% improvement on code tasks. A code-heavy training corpus gives ~5-10x improvement on code tasks. Order of magnitude difference.

**To your "focused training" question**: yes, narrow scope helps for the same params. Code-specialized models with 7B params often outperform general 70B models on code-only benchmarks. But the cost is generality — they're bad at everything else. This is exactly why we have CodeLlama, StarCoder, DeepSeek-Coder, etc., alongside the general models.

The frontier strategy today: train one huge general model, then distill/fine-tune specialized variants from it. You get the data diversity of general training AND the focused quality of specialized fine-tuning. That's roughly what "GPT-4o code", "Claude code", and "Gemini code" all do.

**Q: Why isn't there a "clean" / "best" tokenizer that everyone agrees on?**

A: Several reasons stacked:

1. **Competing requirements.** Cover all human languages efficiently vs cover English efficiently. Be reversible for any byte sequence vs maximize compression. Handle code well vs handle prose well. Handle emojis, special chars, control sequences. Each requirement pushes the design in a different direction.
2. **Lock-in once trained.** A tokenizer is part of a pretrained model's identity. Embeddings and output layers are sized to the vocab. You can't swap tokenizers without retraining or doing complex weight surgery. So labs make their choice, train a big model, and stick with it forever.
3. **Each lab has different goals.** OpenAI wants multilingual + code. Anthropic wants high-quality reasoning. Meta wants efficient inference on cheaper hardware. These goals lead to slightly different tokenizer trade-offs.
4. **The tokenizer is also a competitive moat.** It encodes lab-specific decisions (special tokens for tools, RLHF formats, etc.) that competitors would have to replicate.
5. **Ecosystem inertia.** GPT-2's tokenizer (BPE on bytes with regex pre-tokenization) became the de-facto standard because so much downstream tooling was built around it. Llama's SentencePiece is mostly compatible. New tokenizers face a high adoption bar.

The "ideal" tokenizer for any specific use case can be designed; the universal one can't, because the requirements genuinely conflict. The long-term answer might just be eliminating tokenization (byte-level models) — but that's still a research bet.
