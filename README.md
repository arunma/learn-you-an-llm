# learn-you-an-llm

Learning materials for understanding how LLMs work, built around Karpathy's [nanochat](https://github.com/karpathy/nanochat) codebase.

Every concept is annotated: `[PT]` = PyTorch built-in, `[NC]` = nanochat custom. Tensor shape traces follow dimensions through each operation.

## Tutorial

- `llm_tutorial.html` — illustrated single-page tutorial (open in a browser)
- `llm_tutorial.pdf` — PDF version

## Learning journal

The `journal/` directory breaks down the full nanochat build into 9 sections, from tokenisation to modern techniques:

| # | File | What it covers |
|---|------|---------------|
| 0 | [Architecture Overview](journal/00_architecture_overview.md) | Full pipeline shape trace, `wte`/`lm_head`/`n_head`, logits/softmax/cross-entropy, complete dimension trace reference tables, PyTorch quick reference |
| 1 | [Tokenisation](journal/01_tokenisation.md) | BPE algorithm, character-level tokeniser, tiktoken, data preparation (`prepare.py`, `get_batch`) |
| 2 | [Embeddings](journal/02_embeddings.md) | `idx`, `nn.Embedding`, positional encoding, broadcasting, dropout, LayerNorm, AdamW, mixed precision training, regularisation |
| 3 | [Q/K/V Projections](journal/03_qkv_projections.md) | Why three projections, fused `c_attn`, `.view()`/`.transpose()`, attention score computation, dimension trace |
| 4 | [Scaled Attention](journal/04_scaled_attention.md) | Scale by 1/sqrt(d_h), causal mask, softmax, weighted sum of V, head reassembly, attention granularity, token types vs positions, parameters vs activations |
| 5 | [Residuals, Multi-Head & MLP](journal/05_residuals_multihead_mlp.md) | Vanishing gradients, residual connections, MLP/FFN, `c_proj` synthesis, complete `CausalSelfAttention` |
| 6 | [Transformer Block](journal/06_transformer_block.md) | Residual stream, `Block` class, `GPT` model, `forward()`, `lm_head` pipeline (training vs inference), softcap, value embeddings (lookup vs transformation), end-to-end shape trace |
| 7 | [Training & Generation](journal/07_training_and_generation.md) | Autoregressive generation, temperature, top-k, training loop, LR scheduling, checkpointing |
| 8 | [Appendix: Modern Techniques](journal/08_appendix_modern_techniques.md) | FlashAttention, MQA/GQA, sliding window, RoPE, sparse/linear attention, scaling laws, value embeddings, smear, resid lambdas/backout, output stage, naive generate |

The original monolithic journal is in `nanochat_learning_journal.md`.
