# learn-you-an-llm

A single-page illustrated tutorial for understanding how LLMs work, built around Karpathy's [nanochat](https://github.com/karpathy/nanochat) codebase.

## What it covers

The tutorial walks through the core concepts of building a language model from scratch:

1. **Tokenization** — BPE, tiktoken, character-level tokenizers
2. **Embeddings** — token embeddings (`wte`), positional encoding (`wpe`), `nn.Embedding`
3. **Transformer internals** — Q/K/V projections, multi-head attention, the complete dimension trace from input to output
4. **Training infrastructure** — LayerNorm, AdamW, the training loop

Every concept is annotated with what's PyTorch built-in vs. what's nanochat-custom, and includes tensor shape traces so you can follow the dimensions through each operation.

## Files

- `llm_tutorial.html` — the tutorial (open in a browser)
- `llm_tutorial.pdf` — PDF version
