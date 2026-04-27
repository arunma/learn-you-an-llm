# nanochat Learning Journal
*Building an LLM from scratch — concepts, code, and intuition*

> **How to use this document**
> Each phase maps to the nanochat build roadmap. Every code block is annotated:
> - `# [PT]` = PyTorch built-in — comes from the library, just use it
> - `# [NC]` = nanochat custom — Karpathy wrote it, understand every line
>
> Shape traces show tensor dimensions at each step: `(B, T) → (B, T, C)`
> where B = batch size, T = sequence length (time), C = channels (embedding dim)

---

## Architecture Overview — The Full nanochat Pipeline

![Basic Transformer Architecture for Language Modeling](transformer_architecture.png)

*The complete nanochat data flow. Every phase in this journal corresponds to a coloured region above.*

### Reading the diagram

| Colour | Region | Covered in |
|--------|--------|-----------|
| **Blue** | Input token IDs → `wte` embedding table → positional encoding | Phase 1 (token IDs) + Phase 2 (embeddings) |
| **Green** | N × Transformer Blocks (self-attention + layer norm + FFN + residuals) | Phase 3 + Phase 4 |
| **Red** | `lm_head` output projection → logits | Phase 5 |
| **Black** | Operations: `+` addition, cross-entropy loss | Phase 5 |

### Shape trace — left to right across the diagram

```
Input Token IDs   (B, T)           int64   ← get_batch() output, Phase 1
       ↓  wte = nn.Embedding(vocab_size, C)          [PT]
Token Embeddings  (B, T, C)        float   ← each integer → 384-dim vector
       ↓  + positional encoding    (T, C)
Embeddings        (B, T, C)        float   ← tokens now know their position
       ↓  × N Transformer Blocks
Context Vectors   (B, T, C)        float   ← same shape, much richer meaning
       ↓  lm_head = nn.Linear(C, vocab_size, bias=False)  [PT]
Output Logits     (B, T, vocab_size) float ← one score per possible next token
       ↓  F.cross_entropy(logits, targets)               [PT]
Loss              scalar           float   ← compared against Target Token IDs (B, T)

B = batch_size (12)   T = block_size (1024)   C = n_embd (384)
vocab_size = 50,257   N = n_layer (6)

Note on naming: C and n_embd mean the same thing — the embedding dimension (384).
C is used in shape comments throughout this document (shorter).
n_embd is used in GPTConfig and in nanochat's actual code.
You will see both — they are identical.
```

### The two places vocab_size appears (highlighted in diagram)

- **Place 1 — Embedding Table** `(50,257 × 384)`: `self.wte = nn.Embedding(vocab_size, n_embd)` — one learnable row per token
- **Place 2 — Output Projection** `(384 → 50,257)`: `self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)` — maps back to vocabulary scores

These are the only two places `vocab_size` touches the model. Everything in between operates at dimension `C = 384`.

### Key architectural observations

- **Shape is preserved through all transformer blocks** — input `(B, T, C)` and output `(B, T, C)` are identical. Each block refines meaning without changing dimensions.
- **Residual connections** (the curved arrows looping back in the green region) ensure gradients flow cleanly all the way back to the embedding layer during training.
- **The transformer blocks are the expensive part** — `wte` and `lm_head` are single matrix lookups. The 6 stacked blocks with multi-head attention are where the compute lives.

---

### ★ The Complete Dimension Trace — get_batch() to attention scores

*This is the single most important reference table in this journal. It traces every shape change from raw token IDs all the way to the `(B, 6, T, T)` attention score matrix inside CausalSelfAttention. Memorise this and the entire model becomes readable.*

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| 0 | `get_batch()` `[NC]` | `(B, T)` | `(12, 1024)` int64 — raw token IDs |
| 1 | `wte(idx)` `[PT]` | `(B, T, C)` | `(12, 1024, 384)` — **C=384 appears. 2D→3D** |
| 2 | `+ wpe(pos)` `[PT]` | `(B, T, C)` | `(12, 1024, 384)` — shape unchanged |
| 3 | `c_attn(x)` `[PT]` | `(B, T, 3C)` | `(12, 1024, 1152)` — Q+K+V fused. C triples |
| 4 | `.split(384, dim=2)` `[PT]` | `q,k,v: (B, T, C)` | `(12, 1024, 384)` × 3 tensors |
| 5 | `.view(B, T, 6, 64)` `[PT]` | `(B, T, n_h, d_h)` | `(12, 1024, 6, 64)` — **4D! heads labelled** |
| 6 | `.transpose(1, 2)` `[PT]` | `(B, n_h, T, d_h)` | `(12, 6, 1024, 64)` — heads → dim 1 |
| 7★ | `q @ k.T(-2,-1)` `[PT]` | `(B, n_h, T, T)` | `(12, 6, 1024, 1024)` — **75M scores!** |
| 8 | `× 1/√64` `[NC]` | `(B, n_h, T, T)` | scaled — prevents softmax collapse |
| 9 | `masked_fill(-∞)` `[PT]` | `(B, n_h, T, T)` | future positions zeroed |
| 10 | `softmax(dim=-1)` `[PT]` | `(B, n_h, T, T)` | weights sum to 1.0 per row |
| 11 | `att @ v` `[PT]` | `(B, n_h, T, d_h)` | `(12, 6, 1024, 64)` — weighted value sum |
| 12 | `.transpose(1,2)` `[PT]` | `(B, T, n_h, d_h)` | `(12, 1024, 6, 64)` — heads back to dim 2 |
| 13 | `.view(B, T, 384)` `[PT]` | `(B, T, C)` | `(12, 1024, 384)` — 6×64 concatenated |
| 14 | `c_proj` `[PT]` | `(B, T, C)` | `(12, 1024, 384)` — **heads mixed. synthesis complete ✓** |

`B=12  ·  T=1024  ·  C=n_embd=384  ·  n_h=n_head=6  ·  d_h=head_dim=64  ·  V=vocab_size=50257`

**Three moments a new dimension appears:**
1. Step 1 `wte` — C=384 appears. Tensor goes 2D → 3D.
2. Step 5 `.view()` — n_head and head_dim appear. Tensor goes 3D → 4D.
3. Step 7 `q@k.T` — second T appears. The 64 dims cancel → T×T scores.

**After Step 14, this `(B, T, 384)` flows back to the Block's residual addition:**
```python
x = x + self.attn(self.ln_1(x))   # [NC]  x: (B,T,384) throughout
x = x + self.mlp(self.ln_2(x))    # [NC]  shape never changes
# → repeat × 6 blocks
# → ln_f → lm_head → (B, T, 50257) logits
```

---

## The Entrance, Exit, and Thinking Style — `wte`, `lm_head`, `n_head`

This is where raw token IDs first meet the neural network's architecture. Three terms define the boundary between tokenisation and computation.

### `wte` — The Entrance

Computers can't do math on words, and raw token IDs (like `50256`) are just arbitrary integers — the number itself carries no meaning. The model needs *vectors* (lists of numbers where proximity encodes meaning).

- **What it is:** A giant lookup table with one learnable row per vocabulary token
- **PyTorch:** `self.wte = nn.Embedding(vocab_size, n_embd)` `# [PT]`
- **Shape:** `(50,257 rows × 384 cols)` = 19.3M learnable parameters
- **How it works:** Token ID `2746` ("model") → go to row 2746 → pull out 384 numbers that encode the *meaning* of that token
- **Why `vocab_size` here:** Every possible token from the tokeniser needs its own unique row. No row = no representation = can't process that token

```
Token ID 2746  →  wte[2746]  →  [0.21, -0.54, 0.88, ..., 0.13]
                                  ↑ 384 numbers representing "model"
```

The 384 numbers are *learned during training* — they start random and gradually shift so that tokens used in similar contexts end up with similar vectors. This is why `king - man + woman ≈ queen` falls out of the geometry.

---

### `lm_head` — The Exit

After 6 transformer blocks have finished "thinking", each token position holds a 384-dim vector of rich contextual meaning. But the model's job is to *predict the next token* — and there are 50,257 possible answers. The 384 numbers need to become 50,257 scores.

- **What it is:** A linear layer that projects 384 dims → 50,257 dims
- **PyTorch:** `self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)` `# [PT]`
- **Shape:** `(384 → 50,257)` — produces one raw score ("logit") per vocabulary token
- **How it works:** Highest logit = model's prediction for the next token
- **Why `vocab_size` here:** The model needs one "exit door" for every word it might want to say

```
Context vector  [0.82, 0.41, ...]   ← 384 numbers after 6 blocks
      ↓  lm_head
Logits          [2.1, 0.3, 8.7, 6.2, ...]   ← 50,257 raw scores
      ↓  softmax
Probabilities   ["the": 3%, "flooded": 31%, "dry": 9%, ...]
      ↓  argmax (or sample)
Prediction      "flooded"
```

> **Weight tying:** In nanochat, `wte.weight` and `lm_head.weight` share the same tensor — the entrance and exit use identical parameters. This halves the parameter count at those two layers and works because the same "meaning geometry" that helps encode tokens also helps predict them.

---

### `n_head` — The Thinking Style

If `n_embd = 384` is the total "brain power" available per layer, `n_head = 6` determines *how that power is divided*.

**The photo analogy:** Imagine looking at a photo of a cat sitting on a mat.

- **1 head:** You look at the whole image at once, trying to understand everything simultaneously
- **6 heads:** Specialist perspectives running in parallel:
  - Head 1 focuses on *colours*
  - Head 2 focuses on *shapes*
  - Head 3 focuses on *relationships* ("cat is **on** the mat")
  - Head 4 focuses on *texture*
  - Head 5 focuses on *foreground vs background*
  - Head 6 focuses on *overall scene*
  - All six combine their findings at the end

**In the model:** `n_head = 6` means the 384-dim embedding is split into 6 chunks of 64 dims each (`384 ÷ 6 = 64`). Each head runs its own independent attention computation on its 64-dim slice, then all 6 outputs are concatenated back to 384.

```
Input:  (B, T, 384)
           ↓  split into 6 heads
Head 1: (B, T, 64)  → attends to grammar relationships?
Head 2: (B, T, 64)  → attends to topic / subject matter?
Head 3: (B, T, 64)  → attends to nearby context?
Head 4: (B, T, 64)  → attends to coreference ("it" = what)?
Head 5: (B, T, 64)  → attends to long-range dependencies?
Head 6: (B, T, 64)  → attends to positional patterns?
           ↓  concatenate all heads
Output: (B, T, 384)   ← same shape as input
```

The model *learns* what each head specialises in — you don't assign roles manually. The split-concatenate structure forces specialisation by limiting each head's capacity to 64 dims instead of 384.

> **The constraint that creates specialisation:** Each head only sees 64 of the 384 dimensions. It *can't* try to track everything — it must develop a focused perspective. Forcing scarcity is what makes the heads diverge.

---

### Summary table

| Term | What it is | Analogy | Shape | PyTorch |
|------|-----------|---------|-------|---------|
| `wte` | Embedding table | Dictionary: ID → coordinates | `(vocab_size, n_embd)` = `(50257, 384)` | `nn.Embedding` `[PT]` |
| `lm_head` | Output projection | Voting booth: pick next word | `(n_embd, vocab_size)` = `(384, 50257)` | `nn.Linear` `[PT]` |
| `n_head` | Attention head count | Team of specialists | splits `n_embd` → `n_head` slices of `head_dim = n_embd // n_head = 64` | `[NC]` split logic |

### The complete shape flow

```
(B, T)         ← integer token IDs from get_batch()
   ↓ wte
(B, T, 384)    ← each ID is now a 384-dim meaning vector            [Entrance]
   ↓ + positional encoding
(B, T, 384)    ← vectors now carry position information
   ↓ × 6 transformer blocks, each with 6 attention heads of dim 64
(B, T, 384)    ← same shape, deeply contextualised meaning           [Thinking]
   ↓ lm_head
(B, T, 50257)  ← 50,257 scores per token position                   [Exit]
   ↓ cross_entropy vs targets (B, T)
scalar loss    ← single number to minimise during training
```

---

## Logits, Softmax, and Cross-Entropy Loss

These three concepts form the output pipeline — everything that happens after the transformer blocks finish processing and the model needs to make a prediction.

### Logits — raw scores before probability

A logit is a raw, unnormalised score. Think of a talent show where 50,257 contestants compete for "next token." The judges (`lm_head`) give each contestant a raw score:

```
"cat"      →   8.7   ← highest score
"sat"      →   6.2
"the"      →   2.1
"a"        →   0.3
"jumped"   →  -1.4   ← negative = unlikely, not impossible
```

These numbers — 8.7, 6.2, 2.1 — are logits. They can be any value, positive or negative. They only tell you the *ranking*, not the probability. They come directly from `lm_head`:

```python
logits = self.lm_head(x)   # [PT] nn.Linear matrix multiply
# x shape:      (B, T, 384)    ← 384-dim context vector per token
# logits shape: (B, T, 50257)  ← one raw score per vocabulary token
# Each score = dot product of context vector with one vocab row
```

---

### Softmax — turning scores into probabilities

Softmax converts logits into proper probabilities that sum to exactly 1.0. It does this in three operations:

**Step 1 — apply `exp()` to every logit.** Makes everything positive. Also amplifies differences — the gap between 8.7 and 6.2 becomes much larger.
```
exp(8.7)  = 6002.9   ← was biggest, now enormously biggest
exp(6.2)  =  492.7
exp(2.1)  =    8.2
exp(0.3)  =    1.3
exp(-1.4) =    0.25  ← was negative, now just tiny positive
```

**Step 2 — sum all exp() values.**
```
6002.9 + 492.7 + 8.2 + 1.3 + 0.25 = 6505.4
(in practice you sum all 50,257 tokens)
```

**Step 3 — divide each by the sum.** Every value is now between 0 and 1, all summing to exactly 1.0.
```
"cat"    → 6002.9 / 6505.4 = 92.3%   ← was highest, stays highest
"sat"    →  492.7 / 6505.4 =  7.6%
"the"    →    8.2 / 6505.4 =  0.1%
"jumped" →    0.25 / 6505.4 = 0.004%
```

> **Key property:** Softmax never changes the ranking — the highest logit always becomes the highest probability. It only rescales the numbers into a valid probability distribution.

```python
probs = F.softmax(logits[:, -1, :], dim=-1)   # [PT]
#                          ^^
#                    -1 = last token position (the "predict next" slot)
#       dim=-1 = apply softmax across the vocab dimension (50257)
# Result: (B, 50257) probabilities summing to 1.0
```

---

### Cross-entropy loss — measuring how wrong the model is

After softmax, the model has a probability distribution over 50,257 tokens. Cross-entropy answers: **how wrong is it?**

The formula is simpler than it sounds:
```
loss = -log(probability assigned to the correct token)
```

That's it. You ignore every other token. Just look at what probability the model gave to the word that *actually* came next in the training text, and take the negative log of it.

**Why negative?** `log()` of a number between 0 and 1 is always negative. The negative sign flips it positive so loss is always ≥ 0.

**Why log?** It punishes confident wrong answers exponentially harder than uncertain ones:

```
Model gave correct token 95% probability  →  -log(0.95) = 0.05   ← tiny loss, good
Model gave correct token 50% probability  →  -log(0.50) = 0.69   ← medium loss
Model gave correct token  1% probability  →  -log(0.01) = 4.61   ← high loss, bad
Model gave correct token  0% probability  →  -log(0.00) = ∞      ← catastrophic
```

The loss is the single number that flows backward through every layer during training. Every weight nudges itself slightly to make this number smaller. That process is backpropagation — training.

---

### Why NOT softmax before cross-entropy?

```python
# WRONG — floating point danger
probs = F.softmax(logits, dim=-1)        # softmax first
loss  = F.cross_entropy(probs, targets)  # then loss
# Problem: softmax produces numbers like 0.9999997
# log(0.9999997) loses floating point precision
# Tiny errors accumulate across B*T examples

# CORRECT — PyTorch fuses them internally
loss = F.cross_entropy(logits, targets)  # [PT] raw logits directly
# F.cross_entropy applies log-softmax in one numerically stable operation
# No intermediate rounding errors
```

---

### The full output pipeline in nanochat code

```python
# ── TRAINING ────────────────────────────────────────────────────

logits = self.lm_head(x)             # [PT] (B, T, 50257) raw scores

loss = F.cross_entropy(              # [PT] softmax + log + negate, fused
    logits.view(-1, vocab_size),     # [NC] (B*T, 50257) — flatten for PyTorch
    targets.view(-1)                 # [NC] (B*T,)       — flatten targets
)
# Returns one scalar — the average loss across all B*T token predictions

loss.backward()   # [PT] nudge every weight to reduce this number


# ── INFERENCE (generating text) ──────────────────────────────────

last_logits = logits[:, -1, :]            # [NC] (B, 50257) — last position only
probs = F.softmax(last_logits, dim=-1)    # [PT] convert to probabilities

# Option A — greedy: always take the most likely token (deterministic)
next_token = probs.argmax(dim=-1)         # [PT] always same output

# Option B — sampling: randomly pick weighted by probability (creative)
next_token = torch.multinomial(probs, 1)  # [PT] different output each run

# Append predicted token and feed back in for next prediction
idx = torch.cat([idx, next_token], dim=1) # [PT] autoregressive loop
```

**The key difference between training and inference:**
- Training: use all T positions simultaneously, compare against `targets`, compute loss
- Inference: only use the *last* position's logit (`logits[:, -1, :]`), there are no targets, softmax to pick next token

---

### Attention scores — a preview for Phase 3

You asked about attention scores in the context of masking. Here is the one-paragraph version — Phase 3 will build this up fully with concrete numbers.

When the model processes "The cat sat on the mat", every token needs to ask every other token: **"how relevant are you to understanding me?"** It computes a pairwise relevance score between every token pair — token 0 vs token 1, token 0 vs token 2, and so on. These are attention scores.

The **causal mask** sets certain scores to `-∞` before softmax — specifically any score where a token tries to look *forward* in time. After softmax, `-∞` becomes zero probability, so future tokens contribute nothing. Token 2 ("sat") can attend to tokens 0 and 1 but not 3, 4, 5.

```
Attention score matrix for a 5-token sequence (✓ = allowed, ✗ = masked to -∞):

            "The"  "cat"  "sat"  "on"   "mat"
"The"   →  [  ✓      ✗      ✗      ✗      ✗  ]   can only see itself
"cat"   →  [  ✓      ✓      ✗      ✗      ✗  ]   sees The, cat
"sat"   →  [  ✓      ✓      ✓      ✗      ✗  ]   sees The, cat, sat
"on"    →  [  ✓      ✓      ✓      ✓      ✗  ]   sees The..on
"mat"   →  [  ✓      ✓      ✓      ✓      ✓  ]   sees everything

This is torch.tril() — a lower triangular matrix of ones.
Everything above the diagonal = -∞ before softmax = 0 after softmax.
```

In nanochat this mask is created once in `__init__` and reused every forward pass:
```python
self.register_buffer(                          # [PT] saves tensor with model
    'bias',
    torch.tril(torch.ones(block_size, block_size))  # [PT] lower triangular
)
# shape: (1024, 1024) — T × T matrix of 1s below diagonal, 0s above
```

Full treatment — Q, K, V, the score computation, why it works — in Phase 3.

---

---

## ★ Quick Reference — Complete Dimension Trace

*The single most useful table in this journal. Every shape change in the entire model, in order. Pin this mentally — once you know it, all of nanochat's code becomes readable.*

### From get_batch() to attention scores (inside CausalSelfAttention)

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| 0 | `get_batch()` `[NC]` | `(B, T)` | `(12, 1024)` int64 — raw token IDs |
| 1 | `wte(idx)` `[PT]` | `(B, T, C)` | `(12, 1024, 384)` — **C=384 appears. 2D→3D** |
| 2 | `+ wpe(pos)` `[PT]` | `(B, T, C)` | `(12, 1024, 384)` — position added. shape unchanged |
| 3 | `c_attn(x)` `[PT]` | `(B, T, 3C)` | `(12, 1024, 1152)` — Q+K+V fused. **C triples** |
| 4 | `.split(384, dim=2)` `[PT]` | `q,k,v: (B,T,C)` | `(12, 1024, 384)` × 3 — separated |
| 5 | `.view(B,T,6,64)` `[PT]` | `(B, T, n_h, d_h)` | `(12,1024,6,64)` — **4D! heads labelled** |
| 6 | `.transpose(1,2)` `[PT]` | `(B, n_h, T, d_h)` | `(12,6,1024,64)` — heads → dim 1 |
| 7★ | `q @ k.T(-2,-1)` `[PT]` | `(B, n_h, T, T)` | `(12,6,1024,1024)` — **64 cancels → T×T scores** |
| 8 | `× 1/√64` `[NC]` | `(B, n_h, T, T)` | scaled — prevents softmax collapse |
| 9 | `masked_fill(-∞)` `[PT]` | `(B, n_h, T, T)` | future positions zeroed |
| 10 | `softmax(dim=-1)` `[PT]` | `(B, n_h, T, T)` | weights sum to 1.0 per row |
| 11 | `att @ v` `[PT]` | `(B, n_h, T, d_h)` | `(12,6,1024,64)` — weighted value sum |
| 12 | `.transpose(1,2)` `[PT]` | `(B, T, n_h, d_h)` | `(12,1024,6,64)` — heads back to dim 2 |
| 13 | `.view(B,T,384)` `[PT]` | `(B, T, C)` | `(12,1024,384)` — 6×64 concatenated |
| 14 | `c_proj` `[PT]` | `(B, T, C)` | `(12,1024,384)` — **heads mixed. synthesis ✓** |

### From CausalSelfAttention output to loss (the full model)

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| 14 | `c_proj output` | `(B, T, C)` | back in Block.forward() |
| 15 | `x = x + attn_out` `[NC]` | `(B, T, C)` | residual addition |
| 16 | `mlp(ln_2(x))` `[NC]` | `(B, T, C)` | FFN: 384→1536→384 per token |
| 17 | `x = x + mlp_out` `[NC]` | `(B, T, C)` | residual addition |
| 18 | `× 6 blocks total` | `(B, T, C)` | **shape never changes through any block** |
| 19 | `ln_f(x)` `[PT]` | `(B, T, C)` | normalise residual stream — only time |
| 20★ | `lm_head(x)` `[PT]` | `(B, T, V)` | `(12,1024,50257)` — **V=50257 appears** |
| 21 | `F.cross_entropy` `[PT]` | `scalar` | one number — minimise this |

`B=12  ·  T=1024  ·  C=n_embd=384  ·  n_h=n_head=6  ·  d_h=head_dim=64  ·  V=vocab_size=50257`

**Two shape changes in the entire model:**
1. Step 1 — `wte`: `(B,T)` → `(B,T,384)` — C appears
2. Step 20 — `lm_head`: `(B,T,384)` → `(B,T,50257)` — V appears

**Everything between operates at exactly `(B, T, 384)` — always.**

---

## PyTorch Built-ins Quick Reference
*Populated as we go — one entry per new built-in encountered*

| API | What it does | First seen |
|-----|-------------|-----------|
| `torch.tensor(data, dtype=)` | Convert Python list/array to tensor | Phase 1 |
| `torch.randint(high, size)` | Random integers — used for batch sampling | Phase 1 |
| `torch.stack([t1, t2, ...])` | Stack list of 1D tensors into 2D matrix | Phase 1 |
| `torch.from_numpy(arr)` | Convert numpy array to tensor (shares memory) | Phase 1 |
| `.to(device)` | Move tensor to CPU or GPU | Phase 1 |
| `nn.Embedding(num, dim)` | Lookup table: integer → row of weight matrix | Phase 1 (referenced) |
| `nn.Linear(in, out, bias=)` | Fully connected layer: y = xWᵀ + b | Phase 1 (referenced) |
| `F.cross_entropy(logits, targets)` | Loss: softmax + log + negate, fused and stable | Arch overview |
| `F.softmax(x, dim=)` | Convert logits to probabilities summing to 1 | Arch overview |
| `probs.argmax(dim=)` | Index of highest value — greedy decoding | Arch overview |
| `torch.multinomial(probs, n)` | Sample index weighted by probabilities | Arch overview |
| `torch.cat([t1, t2], dim=)` | Concatenate tensors along a dimension | Arch overview |
| `torch.tril(torch.ones(T, T))` | Lower triangular matrix — the causal mask | Arch overview |
| `self.register_buffer(name, tensor)` | Save tensor with model, not a trainable parameter | Arch overview |
| `nn.Dropout(p)` | Zero random fraction of values during training | Phase 2 |
| `nn.ModuleDict(dict)` | Named dict of layers — PyTorch tracks all as parameters | Phase 2 |
| `nn.ModuleList([...])` | Ordered list of layers — PyTorch tracks all as parameters | Phase 2 |
| `nn.LayerNorm(dim)` | Normalise vector to mean=0, std=1 — training stability | Phase 2 |
| `nn.GELU()` | Smooth activation function — like ReLU but fades near 0 | Phase 2 |
| `torch.arange(n, device=)` | Create tensor `[0, 1, 2, ..., n-1]` | Phase 2 |
| `torch.nn.utils.clip_grad_norm_(params, max)` | Cap gradient vector magnitude — prevents explosions | Phase 2 |
| `torch.optim.AdamW(params, lr, weight_decay)` | Optimiser: Adam + correct weight decay | Phase 2 |
| `tensor.view(*shape)` | Reshape without copying — relabels memory layout | Phase 3.1 |
| `tensor.transpose(dim_a, dim_b)` | Swap two dimensions — used to move heads to dim 1 | Phase 3.1 |
| `tensor.split(size, dim)` | Split tensor into equal chunks along a dimension | Phase 3.1 |
| `@` / `torch.matmul(a, b)` | Batched matrix multiply — runs all heads in parallel | Phase 3.1 |

---

## Phase 1 — Tokenisation
*How raw text becomes integer tensors the model can process*

### The problem tokenisation solves

A neural network can only work with numbers. Raw text must become a sequence of integers before any matrix multiplication can happen. The tokeniser is the bridge — it sits entirely outside the PyTorch model, has no trainable parameters, and runs once before training.

```
"Hello world"  →  [15496, 995]  →  nn.Embedding  →  (2, 384) float tensor
    text           token IDs          [PT]            model input
```

Three vocabulary strategies exist, each a trade-off between vocabulary size and tokens-per-word:

| Strategy | Vocab size | Tokens for "unhappiness" | Used in |
|----------|-----------|--------------------------|---------|
| Character-level | 65 (Shakespeare) | 11 (one per char) | nanogpt demo |
| BPE subword | 50,257 | 3 (`un`, `happiness`, end) | nanochat / GPT-2 |
| Word-level | millions | 1 | almost never |

**The context window trade-off:** character-level uses ~4× more tokens than BPE for the same text. A 1024-token context window holds ~930 characters at char-level vs ~4000 characters at BPE. This is the primary reason real LLMs use BPE.

---

### 1.1 — BPE (Byte Pair Encoding)

#### The problem — three bad vocabulary options

Before BPE, there were three naive approaches, each with a fatal flaw:

```
Option A — Word-level vocabulary:
  "unhappiness" and "happiness" → completely separate rows in embedding table
  Share zero information despite sharing meaning.
  1 million unique words = 1 million rows = massive table.
  New words at inference ("GPT-4ish") get no embedding → unknown token problem.

Option B — Character-level vocabulary:
  Only 256 tokens (one per ASCII character).
  "hello" = 5 tokens. Context window fills fast.
  "unhappiness" = 11 tokens vs 3 in BPE → 4× less text per context window.

Option C — BPE subword (the solution):
  Learns which character combinations are worth merging from your actual data.
  "unhappiness" → ["un", "happiness"] — shares "happiness" with "happy".
  Small vocab (50,257), efficient tokens (~4 chars each), handles new words.
```

#### The core idea

BPE is greedy statistics, nothing more. Start with individual characters, repeatedly find the most frequent adjacent pair in your corpus, merge it into a new token, repeat until you hit your target vocabulary size. The ordered list of merge rules that results IS the tokeniser.

#### The algorithm step by step

**Starting corpus** (with end-of-word marker `Ġ`):

```
"low"    → l o w Ġ
"lower"  → l o w e r Ġ
"newest" → n e w e s t Ġ
```

**Initial vocabulary:** just the 9 unique characters: `l o w e r n s t Ġ`

**Counting all adjacent pairs before Merge 1:**

| Pair | Appears in | Count |
|------|-----------|-------|
| `l + o` | "low", "lower" | **2 ← winner** |
| `o + w` | "low", "lower" | 2 (tie — l+o wins first) |
| `w + e` | "lower", "newest" | 2 |
| `w + Ġ` | "low" | 1 |
| `e + r` | "lower" | 1 |
| `e + s` | "newest" | 1 |

**Merge 1:** `l+o` wins — appears twice.
```
l o w Ġ       →  lo w Ġ
l o w e r Ġ   →  lo w e r Ġ
n e w e s t Ġ →  n e w e s t Ġ   (unchanged)
```
Vocabulary grows to 10: adds `lo`

**Merge 2:** `lo+w` now appears twice — wins.
```
lo w Ġ       →  low Ġ
lo w e r Ġ   →  low e r Ġ
```
Vocabulary grows to 11: adds `low`

**After many merges**, `"lowering"` (a word never seen in training) still tokenises cleanly:
```
"lowering" → low + er + ing + Ġ
```
`"low"` is shared across "low", "lower", "lowering", "lowest" — their embeddings start from the same token. Meaning is shared by construction. This is why BPE beats word-level vocabulary.

#### From-scratch Python implementation

```python
# [NC] All code below is nanochat custom — plain Python, no PyTorch
from collections import Counter

def get_pairs(vocab):
    """Count all adjacent pairs across the whole corpus."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Replace all occurrences of the winning pair with the merged token."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# Initial vocab: each word as space-separated chars, with frequency
vocab = {
    'l o w Ġ':      5,   # "low" appears 5 times in corpus
    'l o w e r Ġ':  2,
    'n e w e s t Ġ': 6,
    'w i d e r Ġ':   3,
}

num_merges  = 10
merge_rules = []          # this ordered list IS the tokeniser

for i in range(num_merges):
    pairs = get_pairs(vocab)
    best  = max(pairs, key=pairs.get)   # most frequent pair
    vocab = merge_vocab(best, vocab)
    merge_rules.append(best)
    print(f"Merge {i+1}: {best[0]!r} + {best[1]!r} → {''.join(best)!r}")

# Encoding new text: apply merge rules in order
def tokenise(text, merge_rules):
    words  = text.split()
    tokens = [' '.join(list(w)) + ' Ġ' for w in words]
    for (a, b) in merge_rules:
        tokens = [t.replace(f'{a} {b}', f'{a}{b}') for t in tokens]
    result = []
    for t in tokens:
        result.extend(t.split())
    return result

tokenise("low lower", merge_rules)
# → ['lowĠ', 'low', 'erĠ']
#    ↑ complete word   ↑ two tokens sharing 'low'
```

**Key insight:** `merge_rules` is a plain Python list. Serialise it to disk and you have a complete, reloadable tokeniser — no PyTorch needed.

---

### 1.2 — Character-level tokeniser (nanogpt Shakespeare demo)

The simplest possible tokeniser. Vocabulary = every unique character in your training text. No merge rules, no external dependencies — just two Python dicts.

#### The complete implementation

```python
import torch                          # [PT] only needed for tensor conversion

# ── Step 1: build vocab ──────────────────────────────────────────
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()                   # [NC]

chars     = sorted(list(set(text)))   # [NC] sorted = deterministic mapping
vocab_size = len(chars)               # [NC] 65 for Shakespeare
# sorted() is critical: without it, set() gives random order each run,
# making saved model weights map to wrong characters on reload

# ── Step 2: lookup tables ────────────────────────────────────────
stoi = { ch: i for i, ch in enumerate(chars) }   # [NC] string → int
itos = { i: ch for i, ch in enumerate(chars) }   # [NC] int → string
# stoi['h'] → 34    (integer ID for 'h')
# itos[34]  → 'h'   (back to the character)

# ── Step 3: encode / decode ──────────────────────────────────────
encode = lambda s: [stoi[c] for c in s]            # [NC] str  → list[int]
decode = lambda l: ''.join([itos[i] for i in l])   # [NC] list[int] → str

# round-trip always works:
decode(encode("Hello")) == "Hello"   # True

# ── Step 4: encode entire corpus → tensor ───────────────────────
data = torch.tensor(encode(text), dtype=torch.long)
#      ^^^^^^^^^^^^                  ^^^^^^^^^^^
#      [PT] creates tensor           [PT] long = int64 (required by nn.Embedding)
#                    ^^^^^^^^^^^^^^
#                    [NC] our encode() from step 3
# shape: torch.Size([1115394])  ← 1.1M characters

# ── Step 5: train / val split ────────────────────────────────────
n          = int(0.9 * len(data))   # [NC]
train_data = data[:n]               # [NC]
val_data   = data[n:]               # [NC]

# ── Step 6: batch sampler ────────────────────────────────────────
block_size = 256   # [NC] context length — chars the model sees at once
batch_size = 64    # [NC] independent sequences per batch

def get_batch(split):                                          # [NC]
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))   # [PT]
    x  = torch.stack([d[i  :i+block_size  ] for i in ix])    # [PT]
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])    # [PT]
    return x.to(device), y.to(device)                        # [PT]

xb, yb = get_batch('train')
# xb shape: (64, 256)  — integer tensor, each cell is a char ID
# yb shape: (64, 256)  — same, shifted right by 1 (the targets)
```

#### What x and y look like (block_size=8 for clarity)

```
x:  [18, 47, 56, 57, 58,  1, 15, 47]   →  "First Ci"
y:  [47, 56, 57, 58,  1, 15, 47, 58]   →  "irst Cit"

Position 0 says: "given F,       predict i"
Position 1 says: "given Fi,      predict r"
Position 2 says: "given Fir,     predict s"
...
Position 7 says: "given First Ci, predict t"

One sequence of length 256 gives 256 training examples — free!
```

#### PyTorch built-ins used (all in get_batch)

| Call | What it does |
|------|-------------|
| `torch.randint(high, size)` | Randomly picks `batch_size` start positions |
| `torch.stack([...])` | Stacks list of 1D slices into `(B, T)` matrix |
| `.to(device)` | Moves tensor to GPU if `device='cuda'` |

Everything else — `set()`, `sorted()`, `enumerate()`, `lambda`, `dict` — is plain Python with zero PyTorch.

---

### 1.3 — tiktoken (what nanochat actually uses)

tiktoken is not a new algorithm. It is GPT-2's pre-trained BPE tokeniser — 50,000 merge rules trained by OpenAI on web text, shipped as a fast Rust library. The interface mirrors your char tokeniser exactly: `encode()` and `decode()`.

#### Why tiktoken over char-level

| | Char tokeniser | tiktoken GPT-2 |
|---|---|---|
| Vocab size | 65 | 50,257 |
| Chars per token | 1 | ~4 |
| Context window efficiency | poor | 4× better |
| Vocab trained on | your corpus | OpenAI web text |
| External dependency | none | `pip install tiktoken` |

#### Basic usage

```python
import tiktoken                                    # [NC]

enc = tiktoken.get_encoding("gpt2")               # [NC]
# "gpt2"         → 50,257 tokens (nanochat uses this)
# "cl100k_base"  → 100,277 tokens (GPT-4)
# Downloads ~500KB merge rules once, caches in ~/.tiktoken

print(enc.n_vocab)     # 50257
print(enc.eot_token)   # 50256 ← special end-of-text token ID

# encode: string → list of ints
ids = enc.encode("Hello, I'm a language model")   # [NC]
# → [15496, 11, 314, 1101, 257, 3303, 2746]

# decode: list of ints → string
enc.decode(ids)                                   # [NC]
# → "Hello, I'm a language model"

# Inspect individual tokens — very useful for debugging:
for tok_id in ids:
    print(f"{tok_id:6d}  →  {repr(enc.decode([tok_id]))}")
# → 15496  →  'Hello'
# →    11  →  ','
# →   314  →  ' I'     ← note the leading space!
# →  1101  →  "'m"
# →   257  →  ' a'
# →  3303  →  ' language'
# →  2746  →  ' model'
```

> **Spaces are part of tokens.** `' I'` (space + I) is one token. GPT-2's BPE was trained on text where spaces attach to the following word. Always use the same tokeniser your model was trained with — the space convention is baked into the vocabulary.

#### encode_ordinary vs encode — important distinction

```python
# encode() raises ValueError if special tokens appear, unless allowed:
enc.encode("hello <|endoftext|> world",
           allowed_special={"<|endoftext|>"})
# → [31373, 220, 50256, 995]
#              ↑ 50256 = EOT special token ID

# encode_ordinary() — ignores special tokens, treats them as text.
# THIS is what nanochat's prepare.py uses for raw data:
enc.encode_ordinary("hello <|endoftext|> world")  # [NC]
# → [31373, 1279, 91, 437, 1659, 5239, 91, 29, 995]
#    treats <> as regular characters

# nanochat adds EOT between documents manually:
eot    = enc.eot_token                            # [NC] = 50256
doc1   = enc.encode_ordinary("First document...")
doc2   = enc.encode_ordinary("Second document...")
tokens = doc1 + [eot] + doc2                      # [NC]
```

#### Data preparation — prepare.py (run once before training)

```python
# prepare.py — run ONCE before training, never again            [NC]
import os, tiktoken
import numpy as np

with open('data/input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

enc = tiktoken.get_encoding("gpt2")               # [NC]
ids = enc.encode_ordinary(data)                   # [NC]
# 1M chars ÷ ~4 chars/token ≈ 250K tokens

n        = int(0.9 * len(ids))
train_ids = ids[:n]
val_ids   = ids[n:]

# Save as uint16 — token IDs 0..50256 fit in 2 bytes
# uint16 = 2 bytes/token vs int64 = 8 bytes/token → 4× smaller files
np.array(train_ids, dtype=np.uint16).tofile('data/train.bin')
np.array(val_ids,   dtype=np.uint16).tofile('data/val.bin')
```

#### Loading in the training loop — get_batch with memmap

```python
import torch, numpy as np

# Memory-map — OS loads file pages on demand, doesn't load all into RAM
train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')  # [NC]
val_data   = np.memmap('data/val.bin',   dtype=np.uint16, mode='r')  # [NC]

block_size = 1024   # [NC] GPT-2 context length
batch_size = 12     # [NC] tune to your GPU VRAM

def get_batch(split):                                              # [NC]
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))  # [PT]
    x = torch.stack(                                              # [PT]
        [torch.from_numpy(                                        # [PT]
            data[i  :i+block_size  ].astype(np.int64)
         ) for i in ix]
    )
    y = torch.stack(                                              # [PT]
        [torch.from_numpy(                                        # [PT]
            data[i+1:i+block_size+1].astype(np.int64)
         ) for i in ix]
    )
    return x.to(device), y.to(device)                            # [PT]

xb, yb = get_batch('train')
# xb: torch.Size([12, 1024])  dtype=torch.int64
# yb: torch.Size([12, 1024])  dtype=torch.int64
```

> **Why `.astype(np.int64)` in get_batch?** Files are stored as uint16 (compact). `nn.Embedding` requires int64. The conversion happens here — compact on disk, correct dtype in the model.

#### Spaces are part of tokens — an important detail

```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello, I'm a language model")
# → [15496, 11, 314, 1101, 257, 3303, 2746]

# Inspect each token:
# 15496  →  'Hello'
#    11  →  ','
#   314  →  ' I'     ← note the LEADING SPACE — space attaches to following word
#  1101  →  "'m"
#   257  →  ' a'
#  3303  →  ' language'
#  2746  →  ' model'
```

GPT-2's BPE was trained on text where spaces attach to the following word. This is why you must always use the same tokeniser your model was trained with — the space convention is baked into the 50,257-entry vocabulary. Different tokenisers produce different token IDs for identical text.

#### The vocab_size handoff to the model

`vocab_size` is the single number that connects the tokeniser to the model. It determines the size of two things — and only two things:

```python
import tiktoken
import torch.nn as nn                             # [PT]
from dataclasses import dataclass

enc = tiktoken.get_encoding("gpt2")

@dataclass
class GPTConfig:                                  # [NC]
    vocab_size : int = enc.n_vocab   # 50257 ← from tiktoken
    block_size : int = 1024          # context length — max sequence length
    n_layer    : int = 6             # number of transformer blocks stacked
    n_head     : int = 6             # attention heads per block
    n_embd     : int = 384           # embedding dimension (C)
    dropout    : float = 0.1         # dropout probability
# All hyperparameters in one place — clean, readable, easy to change
# To scale up: increase n_layer, n_head, n_embd. vocab_size stays fixed (from tokeniser).

class GPT(nn.Module):                             # [NC] class
    def __init__(self, config):
        super().__init__()                        # [PT]

        # Place 1 — embedding table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # [PT]
        # 50257 rows × 384 cols = 19.3M parameters
        # Each row = learnable vector for one token

        # Place 2 — output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # [PT]
        # 384 → 50257 logits (one score per vocab token)
```

#### Complete shape flow — Phase 1 output into the full model

| Operation | Shape | dtype | Notes |
|-----------|-------|-------|-------|
| `get_batch()` | `(B, T)` | int64 | token IDs from train.bin |
| `wte` | `(B, T, 384)` | float32 | one vector per token ← Phase 2 |
| `6× Block` | `(B, T, 384)` | float32 | richer meaning per token |
| `lm_head` | `(B, T, 50257)` | float32 | one logit per vocab token |
| `cross_entropy` | `scalar` | float32 | compared against y targets |

`B = batch_size = 12  ·  T = block_size = 1024  ·  384 = n_embd  ·  50257 = vocab_size`

---

### Phase 1 — Key takeaways

1. **The tokeniser is pure Python** — no PyTorch, no GPU, no training. It runs once on your raw text and produces integer arrays.

2. **Two designs, same interface** — char tokeniser uses `stoi`/`itos` dicts; tiktoken uses compiled Rust. Both expose `encode(str) → list[int]` and `decode(list[int]) → str`.

3. **BPE merge rules are the tokeniser** — not the final vocabulary list. To encode new text, apply each merge rule in order to the character sequence.

4. **vocab_size flows into exactly two model layers** — `nn.Embedding` (input) and `nn.Linear` lm_head (output). Change tokeniser → change both.

5. **dtype discipline** — save as `uint16` (compact), load and convert to `int64` in `get_batch()` (required by `nn.Embedding`).

6. **PyTorch built-ins in Phase 1** — only in `get_batch()`:
   `torch.randint`, `torch.stack`, `torch.from_numpy`, `.to(device)`, `torch.tensor`

---

## Phase 2 — Embeddings
*How integer token IDs become float vectors, and how the model knows word order*

### The problem embeddings solve

After Phase 1, `get_batch()` returns `idx` of shape `(B, T)` — a grid of integers like `18`, `47`, `2746`. Integers are just arbitrary IDs. The number `2746` has no mathematical relationship to `2747`. You can't do meaningful matrix multiplication on raw integers.

Embeddings convert each integer into a 384-dimensional float vector where *proximity encodes meaning* — tokens used in similar contexts end up with similar vectors. This is the moment the model's computation actually begins.

---

### 2.1 — `idx`: what it is and what it contains

`idx` is short for "indices." It is the direct output of `get_batch()` — a batch of token ID sequences, nothing more.

```python
idx = tensor([
    [18, 47, 56, 57, 58,  1, 15, 47],   # [NC] sentence 1: "First Ci..."
    [39, 44, 42, 53, 56,  1, 58, 46],   # [NC] sentence 2: "before t..."
    [ 1, 61, 47, 50, 50,  1, 58, 46],   # [NC] sentence 3: " will th..."
])
# shape: (B=3, T=8)   dtype: int64
# Every cell is just an integer — a row number pointing into the embedding table
```

`idx` lives for exactly one line in `forward()`. It enters as integers, `wte` converts it to vectors, and from that point on the variable is called `x` and everything operates in float vector space.

```python
def forward(self, idx, targets=None):   # [NC] idx arrives: (B, T) int64
    B, T = idx.shape                    # [NC]
    tok_emb = self.wte(idx)             # [PT] idx consumed here → (B, T, 384)
    # idx is never used again after this line
```

---

### 2.2 — `nn.Embedding`: the lookup table (`wte`)

`nn.Embedding` is a matrix with a special forward pass: instead of a matrix multiply, it does a **row lookup**. You give it an integer, it returns that row. That is the entire operation.

```
Embedding matrix shape: (50257, 384)

Row 0      → [ 0.12, -0.45,  0.33, ...]   token "!"
Row 257    → [ 0.02,  0.11, -0.03, ...]   token " "
Row 2746   → [ 0.21, -0.54,  0.88, ...]   token "model"   ← idx=2746 → return this row
Row 15496  → [ 0.55,  0.23, -0.18, ...]   token "Hello"
...
Row 50256  → [-0.31,  0.07,  0.55, ...]   token <EOT>
```

```python
self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # [PT]
# Creates matrix of shape (50257, 384)
# Initialised randomly — every value learned during training
# 50257 × 384 = 19.3M learnable parameters

tok_emb = self.wte(idx)   # [PT]
# idx:     (B, T)        int64  — token IDs
# tok_emb: (B, T, 384)   float  — one row per token ID
# Internally: for each integer in idx, return embedding_matrix[integer]
# No matrix multiply. Pure lookup.
```

**What the 384 numbers mean:** They start random and are adjusted by backpropagation over the entire training run. The training signal pushes tokens used in similar contexts to develop similar vectors. After training:
- `"cat"` and `"dog"` are nearby in 384-dim space
- `"bank"` sits as a compromise between river-context and finance-context
- Mathematical relationships emerge: `king − man + woman ≈ queen`

**Static vs contextual embeddings — a critical distinction:**

At this stage, `"bank"` always maps to the same row — `[0.45, 0.38, 0.51, ...]` — regardless of whether it appears in "river bank" or "bank deposit." This is a *static* embedding — one fixed vector per token type, regardless of context.

The transformer blocks above will transform this static vector into a *contextual* vector — one that reflects what "bank" means *in this specific sentence*, having attended to the surrounding tokens. The embedding table is the starting point, not the final answer. By the time the vector exits Block 6, it is completely different from what entered Block 1 — enriched by 6 layers of attention and computation.

---

### 2.3 — Why positional encoding is needed

Self-attention processes all tokens simultaneously — it has no built-in sense of order. Without positional encoding, `"cat bit dog"` and `"dog bit cat"` produce identical attention patterns. The model literally cannot tell which word came first.

**Why not just add position numbers 1, 2, 3?**

```
Option A — raw numbers:
  "bank" at position 2:   [0.25 + 2,   ...] = [2.25,   ...]  ← values explode at pos 100+
  "bank" at position 100: [0.25 + 100, ...] = [100.25, ...]

Option B — normalise to [0, 1]:
  pos 2 in 10-word sentence  → 2/10  = 0.20
  pos 2 in 100-word sentence → 2/100 = 0.02
  Same absolute position, different value — confusing!

Option C — learned vectors (nanochat uses this):
  One 384-dim vector per position, learned during training ✓
  Stable values, consistent meaning, simple to implement
```

---

### 2.4 — `nn.Embedding` again: positional embeddings (`wpe`)

nanochat uses a second `nn.Embedding` table, this time indexed by position (0, 1, 2...) rather than token ID. Same PyTorch built-in, different purpose.

```python
self.wpe = nn.Embedding(config.block_size, config.n_embd)   # [PT]
# shape: (1024, 384)
# 1024 rows — one per position in the context window
# 384 cols  — must match wte so they can be added together
# ~393K parameters — much smaller than wte's 19.3M

pos = torch.arange(0, T, device=idx.device)   # [PT] → tensor([0, 1, 2, ..., T-1])
pos_emb = self.wpe(pos)                        # [PT] → (T, 384)
# Row 0   = "I am the first token"
# Row 1   = "I am the second token"
# Row 1023= "I am the 1024th token"
```

**Sinusoidal vs learned PE:** The original 2017 Transformer used a fixed sine/cosine formula. nanochat/GPT-2 uses learned PE — vectors initialised randomly and trained like any other parameter. Both approaches work. Learned PE is simpler to implement and equally effective for fixed context lengths.

---

### 2.5 — Combining token + position embeddings

The final input to the transformer blocks is element-wise addition of `tok_emb` and `pos_emb`. Each token's 384-dim vector now carries both "what this token means" and "where it sits in the sequence."

```
wte("The")  = [0.11, -0.33,  0.55, ...]   ← what "The" means
wpe(pos=0)  = [0.14,  0.99,  0.28, ...]   ← "I am first position"
              ──────────────────────────
sum         = [0.25,  0.66,  0.83, ...]   ← "The" at position 0

wte("cat")  = [0.65,  0.48,  0.72, ...]   ← what "cat" means
wpe(pos=1)  = [0.31, -0.12,  0.77, ...]   ← "I am second position"
              ──────────────────────────
sum         = [0.96,  0.36,  1.49, ...]   ← "cat" at position 1
```

**PE is added once, not at every layer.** The positional signal injected here persists through all 6 blocks via the residual connections. It is never re-added inside the blocks.

---

### 2.6 — Broadcasting: how (T, 384) adds to (B, T, 384)

`tok_emb` is `(B, T, 384)` but `pos_emb` is only `(T, 384)`. PyTorch's broadcasting rule automatically expands `pos_emb` to match — the same positional vector is added to every item in the batch. This is correct: position 0 means the same thing in every sentence.

```python
tok_emb = self.wte(idx)     # [PT] (B, T, 384) — different per batch item
pos_emb = self.wpe(pos)     # [PT] (T, 384)    — same for every batch item

x = tok_emb + pos_emb       # [PT] broadcasting: pos_emb auto-expands to (B, T, 384)
# Equivalent to:
# x[0] = tok_emb[0] + pos_emb   (sentence 1)
# x[1] = tok_emb[1] + pos_emb   (sentence 2)
# All in one operation — no Python loop
```

---

### 2.7 — Dropout on the embeddings

After combining, nanochat applies dropout before sending `x` into the transformer blocks.

```python
x = self.transformer.drop(tok_emb + pos_emb)   # [PT]
# config.dropout = 0.1 → zeros 10% of values randomly during training
# model.eval() disables this automatically at inference
# Prevents over-reliance on specific embedding dimensions
```

---

### 2.8 — The complete Phase 2 forward pass

```python
class GPT(nn.Module):                                          # [NC]
    def __init__(self, config):
        super().__init__()                                     # [PT]
        self.transformer = nn.ModuleDict(dict(                 # [PT]
            wte  = nn.Embedding(config.vocab_size,             # [PT]
                                config.n_embd),    # (50257, 384)
            wpe  = nn.Embedding(config.block_size,             # [PT]
                                config.n_embd),    # (1024, 384)
            drop = nn.Dropout(config.dropout),                 # [PT]
            h    = nn.ModuleList(                              # [PT]
                [Block(config) for _ in range(config.n_layer)] # [NC]
            ),
            ln_f = nn.LayerNorm(config.n_embd),                # [PT]
        ))
        self.lm_head = nn.Linear(                              # [PT]
            config.n_embd, config.vocab_size, bias=False
        )

    def forward(self, idx, targets=None):                      # [NC]
        B, T = idx.shape                                       # [NC]

        # ── Phase 2: token + position embeddings ────────────
        tok_emb = self.transformer.wte(idx)                    # [PT] (B,T,384)
        pos     = torch.arange(T, device=idx.device)           # [PT] (T,)
        pos_emb = self.transformer.wpe(pos)                    # [PT] (T,384)
        x = self.transformer.drop(tok_emb + pos_emb)           # [PT] (B,T,384)

        # ── Phase 3: transformer blocks ──────────────────────
        for block in self.transformer.h:                       # [NC]
            x = block(x)                                       # [NC] (B,T,384)→(B,T,384)
        x = self.transformer.ln_f(x)                           # [PT] final LayerNorm

        # ── Phase 5: output ───────────────────────────────────
        logits = self.lm_head(x)                               # [PT] (B,T,50257)

        # ── Loss (training only) ──────────────────────────────
        loss = None
        if targets is not None:                                # [NC]
            loss = F.cross_entropy(                            # [PT]
                logits.view(-1, logits.size(-1)),              # [NC] (B*T, 50257)
                targets.view(-1)                               # [NC] (B*T,)
            )
        return logits, loss
```

---

### 2.9 — Layers, blocks, and what's inside each block

`n_layer = 6` means 6 identical blocks stacked vertically. "Layer" and "block" are the same thing — "layer" is the general ML term, "block" is Karpathy's name for one complete processing unit.

**The shape `(B, T, 384)` goes in at the top and comes out `(B, T, 384)` at the bottom of every single block.** Nothing changes dimensionally. Each block makes the meaning richer without touching the shape.

Inside every block there are two halves:

**Half 1 — Attention sublayer** (the new thing transformers introduced):
```
LayerNorm → CausalSelfAttention → residual add (x = x + attn_output)
```

**Half 2 — FFN sublayer** (plain feedforward NN, identical concept to basic NN):
```
LayerNorm → nn.Linear(384→1536) → GELU → nn.Linear(1536→384) → residual add
```

The FFN is literally a 2-layer neural network with a hidden layer and an activation function — the same pattern as the basic NN you built earlier. The only differences: GELU instead of ReLU, and it runs on each token independently.

**GELU vs ReLU:** Both are activation functions that add non-linearity. ReLU hard-cuts all negatives to zero. GELU fades out smoothly near zero, giving gradients a cleaner path during backpropagation — slightly better for language models in practice.

**The 4× expansion pattern:** The FFN expands from 384 → 1536 (4×) then compresses back. This gives the network "working space" to compute intermediate features before returning to the residual stream dimension. The 4× ratio is empirically established — not theoretically derived.

```python
class MLP(nn.Module):                    # [NC]
    def __init__(self, config):
        super().__init__()               # [PT]
        self.c_fc   = nn.Linear(         # [PT] hidden layer — expand
            config.n_embd,
            4 * config.n_embd            # 384 → 1536
        )
        self.gelu   = nn.GELU()          # [PT] activation function
        self.c_proj = nn.Linear(         # [PT] hidden layer — compress
            4 * config.n_embd,
            config.n_embd                # 1536 → 384
        )

    def forward(self, x):               # [NC]
        x = self.c_fc(x)                # [PT] linear  (384→1536)
        x = self.gelu(x)                # [PT] activate
        x = self.c_proj(x)              # [PT] linear  (1536→384)
        return x
```

---

### 2.10 — LayerNorm, val.bin checkpointing, and gradient clipping

#### LayerNorm — mean=0, sd=1 does NOT cap at ±1

`nn.LayerNorm` normalises the values in each vector to have mean=0 and standard deviation=1. This does not cap values at ±1. Values can be any number — what changes is the *distribution*.

Using exam scores as an analogy: scores `[20, 40, 60, 80, 100]`:
```
Step 1 — subtract mean (60):   [-40, -20,   0, +20, +40]  ← centred at 0
Step 2 — divide by std (28.3): [-1.41, -0.71, 0, +0.71, +1.41]  ← spread = 1
```
A score of 200 would become `(200−60)/28.3 = +4.9` — well above 1. There is no hard clipping.

The benefit is *consistency*: every vector entering the next sublayer has the same scale. Attention weights don't have to cope with values that are sometimes 0.001 and sometimes 1000. Training is dramatically more stable.

```python
self.ln_1 = nn.LayerNorm(config.n_embd)   # [PT]
# Applied before attention: x = self.ln_1(x)
# Applied before FFN:       x = self.ln_2(x)
# Applied after final block: x = self.transformer.ln_f(x)
```

#### val.bin checkpointing — why two datasets

During training, nanochat pauses every `eval_interval` steps and measures loss on `val.bin` — data the model has never trained on. Training loss always goes down (the model memorises), but val loss tells you whether the model has learned *general patterns* or just memorised the training examples.

```python
if step % eval_interval == 0:               # [NC] every 500 steps
    model.eval()                            # [PT] dropout off
    with torch.no_grad():                   # [PT] no gradient tracking
        val_loss = estimate_loss('val')     # [NC] run on val.bin
    model.train()                           # [PT] dropout back on

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, 'best_model.pt')  # [PT] save best version
```

```
Good training:          Overfitting:
  train loss ↓            train loss ↓
  val loss   ↓            val loss   ↓ then ↑  ← save checkpoint here, stop
```

The checkpoint saved at lowest val loss is the best model — not the most-trained model.

#### Gradient clipping — capping catastrophic updates

During backpropagation, every weight gets a gradient. Put all gradients together and you have one giant vector. Its magnitude is the "total size" of the update. Normally small and stable. Occasionally, after a bad batch, it explodes:

```
Normal step:    gradient magnitude = 0.4   → small, stable update
Exploding step: gradient magnitude = 847   → destroys everything learned
```

Gradient clipping scales the entire gradient vector down if it exceeds `max_norm=1.0`, preserving direction but capping size:

```python
# Between loss.backward() and optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # [PT]
# If magnitude > 1.0: scale every gradient by (1.0 / magnitude)
# Direction unchanged. Size tamed. No catastrophic single-step updates.
```

Analogy: a speed limiter on a car. Doesn't change your destination — just prevents crashing by going too fast in one step.

---

#### SGD → Adam → AdamW — the full story

**Plain SGD (the baseline):**

Every training step, SGD does one thing:
```
weight = weight − learning_rate × gradient
```

One fixed learning rate for every single weight in the model. Two problems: it zigzags wastefully in loss landscape ravines (oscillating across the narrow dimension instead of moving along it), and it has no way to handle the fact that some gradients are tiny (`0.001`) and others are huge (`9.4`) — one learning rate is either too slow for the first or too explosive for the second.

Gradients are messy raw signals. Sometimes huge, sometimes tiny, sometimes pointing in a bad direction because of one unusual sentence in the batch. Using raw gradients directly (plain SGD) means the model zigzags violently and struggles to converge. Adam is the "smart navigator" that cleans up those signals.

**The foggy mountain analogy:**
- Plain SGD is a person taking one step at a time, exactly where their feet point right now. They might trip, zigzag, or get stuck in a small dip.
- Adam is a person on a snowboard. Their momentum keeps them gliding smoothly over small bumps, and their adaptive brakes automatically slow them down on steep dangerous cliffs and speed up on flat boring plains.

---

**Adam (2015) — adaptive moment estimation:**

Adam tracks two extra numbers per weight, updated every step:

```
m = 0.9 × m_prev + 0.1 × gradient       ← momentum: smoothed direction
v = 0.95 × v_prev + 0.05 × gradient²    ← variance: smoothed size

weight = weight − lr × m / (√v + 1e-8)  ← adaptive update
```

**`m` — Momentum (the "heavy ball" effect)**

`m` is a running average of recent gradient *directions*. Instead of following just the current gradient, Adam says: "If the last 10 gradients pointed north and this one points south, I'm still going to head north for a bit." It smooths out noise — like a heavy ball rolling downhill that doesn't stop instantly just because it hit a small pebble.

`β₁ = 0.9` means each step: momentum is 90% old direction + 10% new gradient.

**`v` — Variance (the "speed governor")**

`v` is a running average of recent gradient *magnitudes squared*. It tracks volatility. If a weight's gradient constantly jumps between +100 and −100, `v` becomes very large — signalling "this area is chaotic."

`β₂ = 0.95` means each step: variance is 95% old size + 5% new gradient².

**The automatic gearbox — what dividing by √v achieves:**

The division `m / √v` creates a different effective learning rate for every single weight automatically:

- **Neglected weight (tiny `v`):** This weight rarely gets a gradient, so `v` stays small. Dividing by a tiny number makes the step *larger*. Adam says: "We haven't moved this weight much — give it a big push."
- **Exploding weight (huge `v`):** This weight gets bombarded with massive, chaotic gradients, so `v` grows large. Dividing by a large number makes the step *smaller*. Adam says: "This area is too volatile — move very cautiously."

Every weight gets its own learning rate. You don't have to manually tune per-layer rates. Adam "feels" the data and adjusts speed for you.

The `β₁ = 0.9` and `β₂ = 0.95` values control how quickly these averages update:
- `β₁ = 0.9`: each step, momentum is 90% old + 10% new gradient
- `β₂ = 0.95`: each step, variance is 95% old + 5% new gradient²

**The Adam bug:** Adam adds weight decay into the gradient *before* the adaptive scaling:
```
gradient_modified = gradient + λ × weight   ← decay baked into gradient
weight = weight − lr × m / (√v + ε)         ← decay gets rescaled by √v!
```
Weights with large gradients have large `v` — so their decay term gets divided by a large `√v` and nearly disappears. The regularisation effect is inconsistent and weaker than intended.

---

**AdamW (2017) — decoupled weight decay:**

AdamW makes one change: apply weight decay *after* the adaptive update, as a separate operation:

```
weight = weight − lr × m / (√v + ε)    ← adaptive gradient update (unchanged)
weight = weight − lr × λ × weight       ← weight decay applied cleanly, after
```

Because the decay is no longer inside the gradient, it never gets divided by `√v`. Every weight decays by exactly `lr × λ × weight` — a predictable, consistent "forgetting pressure" regardless of how large or small its gradients are. This is the correct behaviour. The "W" in AdamW literally stands for decoupled **W**eight decay.

---

#### AdamW parameters in nanochat

| Parameter | nanochat value | What it controls |
|-----------|---------------|-----------------|
| `lr` | `3e-4` | Base step size — most sensitive hyperparameter. Too high = unstable. Too low = very slow. |
| `betas[0]` (β₁) | `0.9` | Momentum decay — 90% old direction, 10% new gradient each step |
| `betas[1]` (β₂) | `0.95` | Variance decay — 95% old size, 5% new gradient² each step |
| `weight_decay` (λ) | `0.1` | Forgetting pressure — nudges weights 10% of lr toward zero each step |
| `eps` (ε) | `1e-8` | Prevents division by zero in √v + ε. Rarely changed. |

#### nanochat's two parameter groups

Weight decay must only apply to weight matrices. Biases and LayerNorm's learned scale (`γ`) and shift (`β`) parameters are 1-dimensional — decaying them would actively fight the normalisation they're supposed to be learning.

```python
# Split parameters into two groups by tensor dimension     [NC]
decay_params    = [p for n,p in model.named_parameters()
                   if p.dim() >= 2]    # weight matrices — apply decay
no_decay_params = [p for n,p in model.named_parameters()
                   if p.dim() < 2]    # biases, LayerNorm γ/β — no decay

optimizer = torch.optim.AdamW([                            # [PT]
    {'params': decay_params,    'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95))

# Count params in each group (useful to log):
n_decay    = sum(p.numel() for p in decay_params)         # [NC]
n_no_decay = sum(p.numel() for p in no_decay_params)      # [NC]
print(f"decay: {n_decay:,} params | no-decay: {n_no_decay:,} params")
```

#### The complete training step — AdamW in context

```python
for step in range(max_iters):                                 # [NC]

    # 1. Zero gradients accumulated from previous step
    optimizer.zero_grad(set_to_none=True)                     # [PT]

    # 2. Forward pass — build computation graph, get loss
    logits, loss = model(x, y)                                # [NC]

    # 3. Backward pass — compute gradient for every weight
    loss.backward()                                           # [PT]

    # 4. Clip gradients — safety net against explosions
    torch.nn.utils.clip_grad_norm_(                           # [PT]
        model.parameters(), max_norm=1.0
    )

    # 5. AdamW step — update every weight
    optimizer.step()                                          # [PT]
    # Internally, for every weight:
    #   m = 0.9 × m_prev + 0.1 × gradient          (momentum)
    #   v = 0.95 × v_prev + 0.05 × gradient²        (variance)
    #   weight -= lr × m / (√v + 1e-8)              (adaptive update)
    #   weight -= lr × 0.1 × weight                 (weight decay, decoupled)
```

---

#### Mapping math to PyTorch — what's inside the optimizer object

When you call `optimizer = torch.optim.AdamW(...)`, PyTorch stores all the Adam state internally. Here is exactly how the math variables map to the code arguments and the internal buffers:

```python
optimizer = torch.optim.AdamW(        # [PT]
    model.parameters(),
    lr     = 3e-4,                    # base step size — the lr multiplier
    betas  = (0.9, 0.95),             # β₁=0.9 controls m, β₂=0.95 controls v
    eps    = 1e-8,                    # the "tiny" in m / (√v + tiny) — prevents ÷0
    weight_decay = 0.1                # λ — the weight penalty coefficient
)
```

**The math-to-code mapping table:**

| Math variable | PyTorch name | Where it lives | Role |
|--------------|-------------|----------------|------|
| `m` (momentum) | `exp_avg` | `optimizer.state[param]` | The "heavy ball" — smoothed direction |
| `v` (variance) | `exp_avg_sq` | `optimizer.state[param]` | The "brakes" — smoothed scale |
| `β₁` | `betas[0]` = `0.9` | optimizer argument | How much to remember direction |
| `β₂` | `betas[1]` = `0.95` | optimizer argument | How much to remember volatility |
| `tiny` / `ε` | `eps` = `1e-8` | optimizer argument | Safety against division by zero |
| `λ` | `weight_decay` = `0.1` | optimizer argument | Penalty pulling weights toward zero |

When you call `optimizer.step()`, PyTorch is not just looking at the current gradients. It is looking at its internal memory — the `exp_avg` and `exp_avg_sq` tensors it has been accumulating across every previous step. These are real stored tensors in GPU memory, one pair per weight tensor in the model.

**The three-step ritual every training loop follows:**

```python
loss.backward()           # [PT] 1. compute gradients (populate .grad on every weight)
optimizer.step()          # [PT] 2. Adam updates m, v, then adjusts the weight
optimizer.zero_grad()     # [PT] 3. clear gradients so they don't accumulate next batch
# (nanochat zeroes before the forward pass instead — both approaches work)
```

---

#### Lambda (λ) — the weight decay coefficient in depth

`λ` (lambda) is the weight decay coefficient — the `weight_decay=0.1` in the optimizer. It is a "penalty tax" on the size of weights that prevents a model from overfitting by becoming a memorisation machine.

**What it physically does:**

In Adam (the broken version), decay is added to the gradient before the adaptive step:
```
modified_gradient = gradient + (λ × weight)
```
The `λ × weight` term grows larger when a weight is large — forcing the optimizer to push big weights back toward zero. But because this gets rescaled by `√v`, the effect is inconsistent.

In AdamW (the correct version), lambda is applied directly after:
```
weight = weight − lr × m / (√v + ε)    ← gradient update
weight = weight − lr × λ × weight       ← weight decay, separate and clean
```

**The bungee cord analogy:**

Imagine each weight is a hiker trying to find the bottom of a valley (minimum loss). The gradient is the slope of the ground telling them which way to walk. Lambda (`λ`) is a bungee cord attached to each hiker, pulling them back toward zero.

- If the gradient slope is steep (clear signal from the data), the hiker can pull hard against the cord and move far.
- If the gradient is flat or noisy, the bungee cord pulls the hiker back toward zero so they don't wander off and get lost.

**Why small weights generalise better:**

Neural networks overfit when specific weights grow to extreme values because they are memorising one unusual sentence in the training data. Small weights produce smoother, simpler functions that generalise to new text. Lambda enforces this simplicity preference — it tells the model: "minimise the error, but do it using the smallest weights you can get away with."

**Why biases and LayerNorm get λ = 0:**

Biases and LayerNorm's learned scale (`γ`) and shift (`β`) parameters are 1-dimensional. Decaying them actively fights what they are supposed to learn. LayerNorm's `γ` and `β` are learned normalisation corrections — pulling them toward zero with weight decay would gradually destroy the normalisation. Only the actual weight matrices (2D, `p.dim() >= 2`) get penalised.

**The summary mapping:**

| Term | Symbol | nanochat value | Job |
|------|--------|---------------|-----|
| Learning rate | `lr` | `3e-4` | How fast to move toward minimum loss |
| Weight decay | `λ` / `weight_decay` | `0.1` | How hard to pull weights back toward zero |
| Momentum decay | `β₁` / `betas[0]` | `0.9` | How much to remember past gradient direction |
| Variance decay | `β₂` / `betas[1]` | `0.95` | How much to remember past gradient size |
| Epsilon | `ε` / `eps` | `1e-8` | Safety floor to prevent division by zero |

---

#### The hidden memory cost of Adam — why training LLMs is expensive

Adam's power comes at a price: **it stores 4 numbers in GPU memory for every single weight in the model.** This is the primary reason training large LLMs costs millions of dollars.

For a model with N parameters:

| What's stored | Count | Why |
|--------------|-------|-----|
| Weights | N | The model itself |
| Gradients | N | Computed by `loss.backward()` |
| `m` (momentum, `exp_avg`) | N | Adam's direction memory |
| `v` (variance, `exp_avg_sq`) | N | Adam's size memory |
| **Total** | **4 × N** | **4× the model size just for training** |

**The VRAM maths for a 7B model (float32, 4 bytes per number):**

```
Weights:    7,000,000,000 × 4 bytes =  28 GB
Gradients:  7,000,000,000 × 4 bytes =  28 GB
m state:    7,000,000,000 × 4 bytes =  28 GB
v state:    7,000,000,000 × 4 bytes =  28 GB
─────────────────────────────────────────────
Total:                               112 GB
```

A consumer GPU has 8–24 GB VRAM. Training a 7B model on one is impossible without tricks. This is why "optimising the optimizer" is an active research field.

**The three main survival strategies:**

**A. Lower precision (the 16-bit trick)**
Use `bfloat16` instead of `float32` — 2 bytes per number instead of 4. Halves memory immediately. nanochat uses this via `torch.amp.autocast`.
```python
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)  # [PT]
with ctx:
    logits, loss = model(x, y)
```

**B. 8-bit optimizers**
Libraries like `bitsandbytes` compress `m` and `v` down to 1 byte each using quantisation — like converting a photo to JPEG. Minor quality loss, 4× smaller optimizer state.

**C. Gradient checkpointing / sharding (ZeRO)**
In large-scale training (Meta, OpenAI), Adam states are split across multiple GPUs. GPU 1 holds `m` and `v` for layers 1–10, GPU 2 for layers 11–20, etc. Each GPU only needs a fraction of the total state.

**For nanochat:** the model has ~10–85M parameters depending on config. At float32, the full training state fits in under 2 GB — well within any modern GPU. But this knowledge matters the moment you try to scale up.

---

#### Mixed precision training — how fp32 master weights actually work

When you use `torch.amp.autocast` with `dtype=torch.bfloat16`, a natural question is: where does the fp32 weight live and how does PyTorch keep it safe while computing in bf16?

**The fp32 weight lives inside the `Linear` module as `self.weight`:**

```python
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
        #                   ↑ fp32, persistent    ↑ bf16 temporary — discarded after matmul
```

When `.to(dtype=x.dtype)` is called, it **creates a new temporary tensor** — it does NOT modify `self.weight`. Once the matrix multiply is done, the temporary bf16 tensor is garbage-collected. The fp32 `self.weight` sits untouched, ready for the optimizer.

**The full picture in GPU memory:**

```
GPU memory
┌─────────────────────────────────────────────┐
│  Linear module                              │
│  ┌────────────────────────────────────┐     │
│  │ self.weight  (fp32, persistent)    │ ◄───┼── optimizer holds reference
│  │ self.bias    (fp32, persistent)    │     │   and updates in-place
│  └────────────────────────────────────┘     │
│                                             │
│  During forward() with autocast:            │
│  ┌────────────────────────────────────┐     │
│  │ temp_bf16 = self.weight.to(bf16)   │     │  ← created, used, discarded
│  │ result    = x @ temp_bf16.T        │     │    each forward pass
│  └────────────────────────────────────┘     │
└─────────────────────────────────────────────┘
```

The optimizer calls `optimizer.step()` and directly mutates `self.weight` (the fp32 tensor) in place:

```python
self.weight.data -= lr * self.weight.grad   # [PT] in-place update on fp32
# The fp32 master copy is always updated — bf16 is only a compute format
```

**Memory cost of mixed precision:**

```
Pure fp32 training:
  Weights: N × 4 bytes
  Grads:   N × 4 bytes
  m, v:    N × 8 bytes
  Total:   N × 16 bytes

Mixed precision (fp32 master weights + bf16 compute):
  self.weight (fp32, persistent):        N × 4 bytes   ← always in VRAM
  temp_bf16 during forward (brief):      N × 2 bytes   ← created and freed
  Grads (usually bf16):                  N × 2 bytes
  m, v (fp32 in AdamW):                  N × 8 bytes
  Total:                            ≈    N × 14 bytes  ← modest saving

Why bother? The bf16 compute is dramatically faster on modern GPUs (A100, H100)
even with the same memory — tensor cores run bf16 at 2× the throughput of fp32.
```

**Why fp32 master weights are necessary:**

If you trained entirely in bf16, the optimizer would accumulate gradient updates directly into bf16 weights. bf16 has only 7 bits of mantissa (vs 23 for fp32) — very small gradient updates (common in later training) get rounded to zero and the model stops learning. The fp32 master copy is precise enough to accumulate tiny updates correctly over millions of steps.

> **TL;DR:** The fp32 weight is a regular `nn.Parameter` living inside the `Linear` module as `self.weight`. The optimizer holds a reference to it and updates it in-place. The bf16 version is a throwaway temporary created fresh each forward pass and discarded immediately after the matmul. Mixed precision = fp32 for storage and optimisation, bf16 for fast matrix compute.

---

#### Regularisation in nanochat — priority order

| Technique | How | Always on? |
|-----------|-----|-----------|
| Weight decay | `AdamW(weight_decay=0.1)` `[PT]` | Yes |
| LayerNorm | `nn.LayerNorm` `[PT]` | Yes |
| Gradient clipping | `clip_grad_norm_` `[PT]` | Yes |
| Dropout | `nn.Dropout(p=0.1)` `[PT]` | Small data only |
| Early stopping | Monitor val loss `[NC]` | Yes |

---

### Phase 2 — Key takeaways

1. **`idx` is just a grid of integers** `(B, T)` — the output of `get_batch()`. It lives for one line in `forward()`, gets consumed by `wte`, and becomes `x` from that point on.

2. **`nn.Embedding` is a lookup table, not a computation** — give it an integer, get back that row of the weight matrix. No matrix multiply. `[PT]`

3. **Two embedding tables, same `nn.Embedding` class** — `wte` indexed by token ID `(50257, 384)`, `wpe` indexed by position `(1024, 384)`. Added together element-wise to give each token both meaning and position.

4. **Broadcasting handles the batch dimension silently** — `pos_emb` is `(T, 384)`, `tok_emb` is `(B, T, 384)`. PyTorch expands automatically. Same positional vector for every sentence in the batch.

5. **PE is added once** — before Block 1. The positional signal propagates through all 6 blocks via residual connections. Never re-added.

6. **A transformer block is hidden layers + attention + residuals** — the FFN half is literally a 2-layer NN with GELU. The attention half is the new thing. LayerNorm and residuals provide stability.

7. **LayerNorm normalises to mean=0, sd=1 — not ±1 clipping.** Values can exceed ±1. The benefit is consistent scale entering each sublayer.

8. **PyTorch built-ins introduced in Phase 2:**

| API | What it does |
|-----|-------------|
| `nn.Embedding(num, dim)` | Lookup table — integer → row of matrix |
| `nn.Dropout(p)` | Zero random fraction of values during training |
| `nn.ModuleDict(dict)` | Named dict of layers PyTorch tracks as parameters |
| `nn.ModuleList([...])` | List of layers PyTorch tracks as parameters |
| `nn.LayerNorm(dim)` | Normalise to mean=0, sd=1 per vector |
| `nn.GELU()` | Smooth activation function — like ReLU but curves near 0 |
| `torch.arange(n)` | Create tensor `[0, 1, 2, ..., n-1]` |
| `torch.nn.utils.clip_grad_norm_(params, max)` | Cap gradient vector magnitude |

---

---

## Phase 3 — Causal Self-Attention
*The heart of the transformer — how tokens talk to each other*

This is the most important phase in the build. Everything before it was setup. The transformer's power comes entirely from what happens here: every token looks at every other token, scores how relevant each one is, and uses those scores to pull in information from across the sequence.

**Keep this shape trace in view as you read — every section maps to one or more rows:**

| Step | Operation | Shape | Section |
|------|-----------|-------|---------|
| 0 | `get_batch()` | `(B, T)` | Phase 1 |
| 1 | `wte(idx)` | `(B, T, C)` | Phase 2 |
| 2 | `+ wpe(pos)` | `(B, T, C)` | Phase 2 |
| 3 | `c_attn(x)` | `(B, T, 3C)` | 3.1 — fused Q+K+V |
| 4 | `.split(384, dim=2)` | `q,k,v: (B, T, C)` | 3.1 — separated |
| 5 | `.view(B, T, 6, 64)` | `(B, T, n_h, d_h)` | 3.1 — **heads labelled, 3D→4D** |
| 6 | `.transpose(1, 2)` | `(B, n_h, T, d_h)` | 3.1 — heads → dim 1 |
| 7 | `q @ k.T(-2,-1)` | `(B, n_h, T, T)` | 3.2 — **raw scores, 64 dims cancel** |
| 8 | `× 1/√64` | `(B, n_h, T, T)` | 3.2 — scaled |
| 9 | `masked_fill(-∞)` | `(B, n_h, T, T)` | 3.2 — future masked |
| 10 | `softmax(dim=-1)` | `(B, n_h, T, T)` | 3.2 — weights sum to 1.0 |
| 11 | `att @ v` | `(B, n_h, T, d_h)` | 3.2 — weighted value sum |
| 12 | `.transpose(1,2).view(B,T,384)` | `(B, T, C)` | 3.3 — concatenated |
| 13 | `c_proj` | `(B, T, C)` | 3.3 — **heads mixed. synthesis ✓** |

---

### 3.1 — Q, K, V Projections

#### Why three projections? Why not just compare embeddings directly?

The token embedding already carries 384 numbers describing the token. Why not just compare those directly? Because the embedding carries *everything* about a token at once — you need three specialised, separate views of it:

| Matrix | Question it answers | Role |
|--------|-------------------|------|
| **Q** (Query) | "What am I looking for?" | Scored against every other token's Key |
| **K** (Key) | "What do I advertise I contain?" | Matched against Queries to produce scores |
| **V** (Value) | "What do I actually give away?" | Content retrieved when attention is high |

**The library analogy:** When you search for "books about cats" you are the Query. Each book's catalogue entry is a Key. The score = how well your query matches each key. The actual pages you read = the Value. A beautifully written book (high Value) with a terrible catalogue entry (low Key match) will never be found. Q, K, V are separate for exactly this reason.

> **The critical distinction:** K determines whether a token *gets* attended to. V determines *what flows* when it does. They are completely separate learned matrices — a token can be highly findable (strong K) while sharing something completely different (V encodes something else entirely).

---

#### The three projection matrices — W_Q, W_K, W_V

Each of Q, K, V is computed by multiplying the token's embedding by a learned weight matrix. These are the actual trainable parameters of the attention layer.

```
For a single token embedding x of shape (384,):

x (384)  ×  W_Q (384 × 64)  =  Q (64)
x (384)  ×  W_K (384 × 64)  =  K (64)
x (384)  ×  W_V (384 × 64)  =  V (64)
```

The output is 64 dims, not 384. That is `head_dim = n_embd // n_head = 384 // 6 = 64`. Each head works in a 64-dim subspace. Six heads × 64 dims = 384 — the same total. The projection compresses to 64 first; the 6 heads recombine at the end.

#### The fused projection — nanochat's efficiency trick

Instead of three separate matrix multiplies, nanochat does one large multiply that produces Q, K, V for all heads simultaneously, then splits the result:

```python
# Three separate projections (conceptual — not what nanochat does):
W_Q = nn.Linear(384, 64, bias=False)    # [PT]
W_K = nn.Linear(384, 64, bias=False)    # [PT]
W_V = nn.Linear(384, 64, bias=False)    # [PT]

# nanochat fuses all three into ONE matrix for efficiency:
self.c_attn = nn.Linear(384, 3 * 384, bias=False)   # [PT]
# Shape: (384, 1152) — computes Q+K+V for ALL heads in one matmul
# Then split into three chunks of 384
```

One GPU operation instead of three. Faster in practice — GPUs prefer large parallel operations over many small ones. Conceptually identical.

---

#### .view() and .transpose() — the head split explained

After the fused projection and split, q, k, v are each `(B, T, 384)`. The 384 contains all 6 heads mixed together. Two operations separate them:

**`.view()` — relabelling, no data movement**

Reinterprets the 384 as 6 groups of 64 — like drawing dividing lines on a strip of paper. The data never moves.

```python
q = q.view(B, T, n_head, head_dim)   # [PT]
# (B, T, 384) → (B, T, 6, 64)
# Head 0 = dims 0–63
# Head 1 = dims 64–127
# Head 2 = dims 128–191  ... and so on
# No copying — same memory, new shape labels
```

**`.transpose(1, 2)` — move heads to dim 1**

Swaps the T and n_head dimensions so heads sit where PyTorch's batched matmul expects them:

```python
q = q.transpose(1, 2)   # [PT]
# (B, T, 6, 64) → (B, 6, T, 64)
#      ↑  ↑           ↑  ↑
#      T  heads  →  heads  T   (swapped)
```

**Why both steps are needed:** `.view()` must come first to correctly label which 64 dims belong to which head (it reads memory strictly left-to-right). Only after correct labelling can you safely reorder dimensions with `.transpose()`. Jumping straight to `.view(B, 6, T, 64)` would scramble the dims — wrong data per head.

**The view intuition — refolding paper, not moving data:**

```
Original memory layout (one token, 384 dims):
[d0, d1, ..., d63, d64, ..., d127, d128, ..., d191, ..., d383]
 ←── head 0 ───→  ←──── head 1 ────→  ←── head 2 ──→  ...

.view(B, T, 6, 64) just draws dividing lines:
[d0...d63 | d64...d127 | d128...d191 | d192...d255 | d256...d319 | d320...d383]
  head 0       head 1       head 2       head 3        head 4        head 5

No data moved. Just new shape labels on the same memory.
```

After both steps, each head independently sees its own `(T, 64)` slice:
```
Head 0: q[b, 0, :, :]  →  (T, 64)  dims 0–63
Head 1: q[b, 1, :, :]  →  (T, 64)  dims 64–127
Head 2: q[b, 2, :, :]  →  (T, 64)  dims 128–191
Head 3: q[b, 3, :, :]  →  (T, 64)  dims 192–255
Head 4: q[b, 4, :, :]  →  (T, 64)  dims 256–319
Head 5: q[b, 5, :, :]  →  (T, 64)  dims 320–383
```

---

#### `q @ k.transpose(-2, -1)` — computing attention scores

With q, k, v all at shape `(B, 6, T, 64)`, this single line computes every token's score against every other token — for all heads and all batches simultaneously.

**Why transpose k's last two dims?**

Matrix multiply requires the inner dimensions to match: `(T, 64) @ (64, T)`. Transposing k from `(B, 6, T, 64)` to `(B, 6, 64, T)` brings the 64s adjacent so they cancel:

```
q:           (B, 6, T, 64)
k.T(-2,-1):  (B, 6, 64, T)   ← last two dims swapped
result:      (B, 6, T, T)    ← 64 cancels, T×T score matrix appears
```

**Concrete dot product example** (head_dim=3 for readability):
```
Q_sat = [0.1,  0.8,  0.4]    ← "sat" asking: what's my subject?
K_cat = [0.8,  0.3,  0.5]    ← "cat" saying: I'm a noun/subject

score = (0.1×0.8) + (0.8×0.3) + (0.4×0.5)
      =   0.08   +   0.24   +   0.20
      =   0.52   ← "sat" attends moderately to "cat" ✓
```

High dot product = vectors point in similar directions = strong attention. Low dot product = weak attention. The matrix form computes all T² pairs at once.

```python
att = q @ k.transpose(-2, -1)   # [PT]
# q:          (B, 6, T, 64)
# k.T(-2,-1): (B, 6, 64, T)
# att:        (B, 6, T, T)   ← one score per token pair per head

# Scale immediately:
att = att * (1.0 / (head_dim ** 0.5))   # [NC]
# head_dim=64 → scale = 1/8 = 0.125
# Prevents softmax from collapsing — covered in 3.2
```

> **Why negative indices?** `-1` = last dim, `-2` = second-to-last. Negative indexing makes this line work regardless of how many batch dimensions sit in front — shape-agnostic.

---

#### The complete dimension trace — get_batch() to attention scores

This is the full journey every tensor takes, one step at a time. The highlighted dimension is the one that changed at each step.

```
Step  Operation               Shape                   Notes
────  ─────────────────────   ─────────────────────   ──────────────────────────────
 0    get_batch()             (B, T)                  (12, 1024)  int64 token IDs
 1    wte(idx)                (B, T, C)               (12, 1024, 384)  C appears!
 2    + wpe(pos)              (B, T, C)               (12, 1024, 384)  unchanged
 3    c_attn(x)               (B, T, 3C)              (12, 1024, 1152) C triples
 4    .split(384, dim=2)      q,k,v: (B, T, C)        (12, 1024, 384) × 3
 5    .view(B,T,n_head,d_h)   (B, T, n_head, d_h)    (12, 1024, 6, 64)  4D!
 6    .transpose(1, 2)        (B, n_head, T, d_h)    (12, 6, 1024, 64)  heads move
 7    q @ k.T(-2,-1)          (B, n_head, T, T)      (12, 6, 1024, 1024) scores!

B=12  T=1024  C=n_embd=384  n_head=6  d_h=head_dim=64
```

**The three moments a new dimension appears:**
1. `wte` — C=384 appears, tensor goes 2D → 3D
2. `.view()` — n_head and head_dim appear, tensor goes 3D → 4D
3. `@ k.T` — second T appears, the 64 dims cancel and become T×T scores

**The memory cost of the score matrix:**
`(12, 6, 1024, 1024)` = 75,497,472 numbers — just the raw scores before mask and softmax. This is why attention is quadratic in T. Double the sequence length → 4× the attention memory. This is the primary reason models have context window limits.

---

#### The complete `CausalSelfAttention.__init__` — every line annotated

```python
class CausalSelfAttention(nn.Module):           # [NC] custom class
    def __init__(self, config):
        super().__init__()                       # [PT]

        assert config.n_embd % config.n_head == 0   # [NC] head_dim must be integer

        # Fused Q+K+V projection — one matrix for all three, all heads
        self.c_attn = nn.Linear(                 # [PT]
            config.n_embd,
            3 * config.n_embd,   # 384 → 1152 (Q+K+V concatenated)
            bias=False
        )

        # Output projection — recombines all heads back to n_embd
        self.c_proj = nn.Linear(                 # [PT]
            config.n_embd,
            config.n_embd,       # 384 → 384
            bias=False
        )

        # Dropout for attention weights and residual
        self.attn_dropout  = nn.Dropout(config.dropout)   # [PT]
        self.resid_dropout = nn.Dropout(config.dropout)   # [PT]

        self.n_head = config.n_head   # 6     [NC]
        self.n_embd = config.n_embd   # 384   [NC]

        # Causal mask — registered as buffer, not a trainable parameter
        self.register_buffer('bias',                  # [PT]
            torch.tril(                               # [PT] lower triangular
                torch.ones(config.block_size,
                           config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )
        # bias shape: (1, 1, 1024, 1024)
        # (1,1,...) broadcasts across B and n_head dimensions
        # 1s below diagonal = allowed to attend
        # 0s above diagonal = will be masked to -inf before softmax
```

---

### 3.1 — Key takeaways

1. **Q, K, V are three separate learned views of the same embedding.** Q asks, K advertises, V delivers. K determines whether a token gets attended to. V determines what flows when it does.

2. **nanochat fuses Q+K+V into `c_attn`** — shape `(384, 1152)`. One matmul instead of three. `.split(384, dim=2)` cuts the output into three equal chunks.

3. **`.view()` relabels, `.transpose()` reorders.** `.view(B, T, 6, 64)` correctly labels which 64 dims belong to each head. `.transpose(1, 2)` moves heads to dim 1 so batched matmul runs all heads in parallel.

4. **`q @ k.transpose(-2, -1)`** produces `(B, 6, T, T)` — one score per token pair per head. The 64 dims cancel. T² pairs computed simultaneously in one GPU operation.

5. **The complete shape journey:**
```
(B, T) → (B,T,C) → (B,T,C) → (B,T,3C) → q,k,v:(B,T,C)
      → (B,T,6,64) → (B,6,T,64) → att:(B,6,T,T)
```

6. **Attention is quadratic in T** — the `(T, T)` score matrix grows with the square of sequence length. At T=1024 with B=12, n_head=6: over 75 million scores. This is why context window length is the primary memory bottleneck.

7. **New PyTorch built-ins in Phase 3.1:**

| API | What it does |
|-----|-------------|
| `tensor.view(*shape)` | Reshape without copying — relabels memory layout |
| `tensor.transpose(dim_a, dim_b)` | Swap two dimensions |
| `tensor.split(size, dim)` | Split tensor into equal chunks along a dimension |
| `@` operator / `torch.matmul` | Batched matrix multiply — runs all heads in parallel |

---

---

## Phase 3.2 — Scaled Dot-Product Causal Attention
*Turning raw scores (B, 6, T, T) into attention weights — three operations*

From Phase 3.1 we have `att` of shape `(B, 6, T, T)`. Each cell `att[b, h, i, j]` is a raw dot-product score — how much token i wants to attend to token j. Three operations remain before these are usable:

```
(B, 6, T, T)  raw scores
    ↓  ÷ √d_h          ① scale
    ↓  mask -∞          ② causal mask
    ↓  softmax          ③ normalise
(B, 6, T, T)  attention weights  (sum to 1.0 per row)
```

---

### ① Scale by 1/√d_h — why and what happens

**The problem without scaling:**

With `d_h=64`, dot products tend to produce values around ±8 (= √64). Feeding these large values into softmax causes it to collapse to near-binary weights — one token gets ~100% attention, everything else gets ~0%. The gradient from near-zero softmax outputs is essentially zero. Training stalls.

```
Without scaling — softmax on large values:
softmax([8.0, 7.9, 0.1, 0.1]) ≈ [0.50, 0.50, 0.00, 0.00]
                                              ↑ gradient ≈ 0

After scaling by 1/√64 = 0.125:
[8.0, 7.9, 0.1, 0.1] × 0.125 = [1.0, 0.99, 0.01, 0.01]
softmax([1.0, 0.99, 0.01, 0.01]) ≈ [0.38, 0.37, 0.12, 0.12]
                                     ↑ smooth, trainable gradient
```

**Why √d_h specifically?** If Q and K have unit variance, their dot product has variance = d_h. Standard deviation = √d_h. Dividing by √d_h normalises variance back to 1 — keeping scale stable regardless of head size.

```python
att = q @ k.transpose(-2, -1)                      # [PT] (B,6,T,T) raw scores
att = att * (1.0 / math.sqrt(k.size(-1)))           # [NC] scale by 1/√64 = 0.125
# k.size(-1) = head_dim = 64
```

---

### ② The causal mask — enforcing the past-only rule

A language model must only use past tokens to predict the next one — never the future. Without a mask, token 0 could attend to token 1023 and simply copy the answer. The causal mask makes this impossible by setting all future attention scores to -∞ before softmax.

```
Mask matrix (T=4 shown) — 1s below diagonal, 0s above:

           pos 0  pos 1  pos 2  pos 3
pos 0  →  [  1      0      0      0  ]  can only see itself
pos 1  →  [  1      1      0      0  ]  sees pos 0 and 1
pos 2  →  [  1      1      1      0  ]  sees pos 0, 1, 2
pos 3  →  [  1      1      1      1  ]  sees everything past

0s → replaced with -∞ before softmax
```

**Why -∞ and not 0?** `exp(0) = 1` — a zero score still gets attention weight. `exp(-∞) = 0` — no contribution whatsoever. Only -∞ guarantees the masked positions are completely ignored.

```python
# Created once in __init__, reused every forward pass:
self.register_buffer('bias',                        # [PT] saved with model, not a parameter
    torch.tril(torch.ones(block_size, block_size))  # [PT] lower triangular matrix
    .view(1, 1, block_size, block_size)
)
# shape: (1, 1, 1024, 1024) — broadcasts across B and n_head

# Applied in forward():
att = att.masked_fill(                              # [PT]
    self.bias[:, :, :T, :T] == 0,  # [NC] True where mask is 0 (future positions)
    float('-inf')                   # [NC] replace those positions with -infinity
)
```

---

### ③ Softmax — scores become attention weights

Softmax applied row by row along `dim=-1`. For each query token, converts its row of scores into a probability distribution. The -∞ values become exactly 0.0. Remaining values sum to 1.0.

**Why -∞ and not just 0 for masked positions?**
```
If future scores were set to 0:
  exp(0) = 1   ← future token STILL gets some attention weight (1 out of the sum)
  It gets less weight but not zero — the model can still "cheat" by looking ahead

If future scores are set to -∞:
  exp(-∞) = 0   ← exactly zero, guaranteed
  Future tokens contribute absolutely nothing to the weighted value sum
```

**Concrete example — "sat" query row:**
```
After scaling + masking:  [0.024,  0.065,  0.089,   -∞ ]
                           "The"   "cat"   "sat"   "on" (future, masked)

exp() of each:            [1.024,  1.067,  1.093,   0.0]
sum:                       3.184

Attention weights:
  "The":  1.024 / 3.184 = 0.322  (32.2%)
  "cat":  1.067 / 3.184 = 0.335  (33.5%)
  "sat":  1.093 / 3.184 = 0.343  (34.3%)  ← highest: attends to itself most
  "on":   0.000 / 3.184 = 0.000  (0.0%)   ← future: zero weight guaranteed
```

These weights tell us: "sat" attends roughly equally to all three past tokens. This makes grammatical sense — "sat" needs its subject ("cat") and determiner context ("The"). **The model learned this pattern from training — not hard-coded.**

```python
att = F.softmax(att, dim=-1)    # [PT] dim=-1: apply across keys (per query row)
att = self.attn_dropout(att)    # [PT] randomly zero some weights during training
```

---

### ④ Weighted sum of V — the actual information retrieval

Attention weights say how much to attend to each token. Multiplying by V retrieves the actual content. This is the fundamental operation of attention — **the weights determine relevance, the values determine what flows.**

```python
y = att @ v    # [PT]
# att: (B, 6, T, T)   attention weights summing to 1.0 per row
# v:   (B, 6, T, 64)  value vectors for each token and head
# y:   (B, 6, T, 64)  weighted blend of value vectors

# Concretely for "sat" (using weights from above):
# y_sat = 0.322 × V_The  +  0.335 × V_cat  +  0.343 × V_sat  +  0.000 × V_on
#
# "sat"'s output vector is a BLEND:
#   34.3% of what "sat" itself knows
#   33.5% of what "cat" knows (its subject)
#   32.2% of what "The" knows (the determiner context)
#     0% of "on" (future — completely excluded)
#
# "sat" has reached back into the sequence and gathered contextual information
# proportional to how relevant it found each past token via Q·K scores
```

> **Q · K determines relevance. V determines what flows.** These are kept completely separate so the model can learn nuanced retrieval — a token can be highly "findable" (strong K) while sharing something completely different (V encodes something else). This separation is what gives attention its expressive power.

---

### ⑤ Reassemble heads → output projection

Reverse the head split and project back to n_embd:

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)    # [PT]
# (B, 6, T, 64) → (B, T, 6, 64) → (B, T, 384)
# .contiguous() required: .transpose() creates non-contiguous memory
# .view() requires contiguous memory — this is the one place data IS copied

y = self.resid_dropout(self.c_proj(y))               # [PT]
# c_proj: nn.Linear(384, 384) — mixes the 6 head outputs together
# Output: (B, T, 384) — same shape as input to attention
```

---

### The complete CausalSelfAttention.forward()

```python
def forward(self, x):                                            # [NC]
    B, T, C = x.shape

    # ── Phase 3.1: Q, K, V projections ──────────────────────────
    qkv = self.c_attn(x)                                        # [PT] (B,T,1152)
    q, k, v = qkv.split(self.n_embd, dim=2)                    # [PT] each (B,T,384)
    k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # [PT] (B,6,T,64)
    q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # [PT]
    v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # [PT]

    # ── Phase 3.2: scaled dot-product causal attention ───────────
    att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))  # [PT] ① scale
    att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))   # [PT] ② mask
    att = F.softmax(att, dim=-1)                                    # [PT] ③ softmax
    att = self.attn_dropout(att)                                    # [PT]
    y   = att @ v                                                   # [PT] ④ weighted sum

    # ── Reassemble heads ─────────────────────────────────────────
    y = y.transpose(1,2).contiguous().view(B, T, C)                # [PT]
    y = self.resid_dropout(self.c_proj(y))                         # [PT]
    return y   # (B, T, 384) — same shape as input
```

---

### Attention granularity — per token, per layer, or per head?

A common source of confusion: at what granularity does attention "happen"? The answer is all three — but they mean different things.

**Per forward pass:** attention runs `n_layer` times (12 in nanochat) — once per transformer block. Total attention calls = 12, regardless of sequence length.

**Per layer:** one attention call processes all T tokens simultaneously in a single fused GPU operation. Not a loop over tokens.

**Per head:** within one layer, the single attention call is internally split into 6 parallel head computations. Each head sees all T tokens but only 64 of the 384 embedding dimensions.

**Per token:** each of the T tokens produces its own Q, K, V vectors and receives its own output. All tokens processed in parallel.

```
ONE LAYER — attention called once, touches every token:

          Head 0   Head 1   Head 2   Head 3   Head 4   Head 5
Token 0:  [attn]   [attn]   [attn]   [attn]   [attn]   [attn]  →  output 0
Token 1:  [attn]   [attn]   [attn]   [attn]   [attn]   [attn]  →  output 1
...
Token T-1:[attn]   [attn]   [attn]   [attn]   [attn]   [attn]  →  output T-1

Each head sees ALL tokens independently.
All tokens processed IN PARALLEL within one matmul.
```

| Level | Count | Description |
|-------|-------|-------------|
| Per forward pass | 12 | One attention call per block |
| Per layer | 1 | Processes all T tokens at once |
| Per layer, per head | 6 parallel | Each head independent |
| Per layer, per token | T queries | Each token attends to all past tokens |

**The key insight:** at every layer, every token "looks at" every past token. In a 12-layer model, each token is updated 12 times. After layer 0, it has incorporated information from all prior tokens. After layer 1, it incorporates information about *those updates*. After 12 layers, each token's representation is a deeply contextualised fusion of the entire sequence. This is why depth matters — each layer adds another round of cross-token information sharing.

---

### Token types vs token positions — two different things called "token"

The word "token" is overloaded and causes real confusion. There are two completely separate concepts:

**Token types (the vocabulary) — vocab_size = 32,768:**
All the *possible distinct tokens* the model knows about — its dictionary. Each has a unique integer ID (0 to 32,767) and a corresponding row in `wte`.

**Token positions (in a sequence) — up to sequence_len = 2,048:**
The actual tokens *in your specific input*. "The cat sat" has 3 token positions, each holding one of the 32,768 possible token types.

```python
# "The cat sat" tokenised:
token_ids = [142, 8417, 1923]   # 3 positions, each an ID from 0..32767

# After embedding (wte lookup):
x = wte(token_ids)              # shape: (3, 768)
# The 32,768 vocabulary has "disappeared" — now just three 768-dim vectors
```

**Where vocab_size actually appears in the model:**

| Component | Shape | Vocab involved? |
|-----------|-------|----------------|
| `wte` (input embed) | `(32768, 768)` | ✓ Yes — one row per token type |
| `c_q, c_k, c_v, c_proj` | `(768, 768)` | ✗ No — works on 768-dim vectors |
| MLP `c_fc, c_proj` | `(768, 3072)` etc. | ✗ No — works on 768-dim vectors |
| `lm_head` (output) | `(768, 32768)` | ✓ Yes — one column per token type |
| `value_embeds` | `(32768, 768)` | ✓ Yes — one row per token type |

**Only embeddings and lm_head depend on vocab_size.** Everything in between — all 36 attention heads, all MLP layers — operates on 768-dim vectors and is completely vocabulary-agnostic. You could change `vocab_size` from 32,768 to 100,000 without touching a single attention or MLP weight.

**Tracing the dimensions through the model:**

```
Input:  "The cat sat"
         ↓ tokenise
         [142, 8417, 1923]                     shape: (T=3,)
         ↓ wte lookup (table: 32768 × 768)
         [[v_142], [v_8417], [v_1923]]          shape: (T=3, 768)
                                                ← vocab_size disappears here
         ↓ 12 × (attention + MLP) — all 768×768
         [[a_0], [a_1], [a_2]]                  shape: (T=3, 768)
         ↓ lm_head (768 → 32768)
         [[l_0], [l_1], [l_2]]                  shape: (T=3, 32768)
                                                ← vocab_size reappears here
         ↓ softmax + sample
         next token ID (one of 32,768 possibilities)
```

**Why does `c_q` have shape (768, 768) and not (32768, 768)?**

Because `c_q` never sees token IDs — it sees 768-dim embedding vectors. The embedding has already done the work of "what token is this." Once "cat" becomes `[0.21, -0.54, 0.88, ...]`, the attention mechanism treats it as a generic 768-dim vector. It does not ask "is this 'cat' or 'dog'?" — it just operates on whatever 768-dim vector it receives.

This is one of the most elegant properties of the transformer: **internal computations are vocabulary-agnostic.** The vocabulary only enters through the embedding lookup at the boundary.

---

### Parameters vs activations — what scales with sequence length

Another frequent confusion: model parameters do **not** scale with sequence length. Only activations do.

**Parameters** are the weight matrices — the same numbers regardless of whether you process 10 tokens or 2,048 tokens. They get applied to every token but the count does not grow.

**Activations** are the intermediate tensors computed during the forward pass — these scale with T.

```python
# Parameters (fixed, stored on GPU — do not grow with T):
c_q.weight:   (768, 768)    ← same for T=3 or T=2048
c_k.weight:   (768, 768)
c_v.weight:   (768, 768)
c_proj.weight:(768, 768)

# Activations (computed per forward pass — scale with T):
q:   (B, T, 768)            ← grows with sequence length
k:   (B, T, 768)
v:   (B, T, 768)
att: (B, n_head, T, T)      ← grows quadratically with T
```

**Rough parameter count for nanochat:**

```
Per transformer block:
  Attention (c_q, c_k, c_v, c_proj):  4 × 768² = 2.36M
  MLP (c_fc + c_proj):                 2 × 768 × 3072 = 4.72M
  LayerNorms:                          negligible
  Total per block:                     ≈ 7.1M

12 blocks:                             ≈ 85M params

Embeddings:
  wte:                       32768 × 768 = 25M
  lm_head (weight-tied):     0M extra
  value_embeds (6 layers):   6 × 32768 × 768 = 150M
  wpe:                       2048 × 768 = 1.6M

Total:                       ≈ 260M params
```

Note: value embeddings are actually the **biggest** parameter chunk — a quirk of nanochat's modern architecture.

**KV cache at inference (not parameters — computed per step):**

```
KV cache shape: (n_layer, B, T, n_kv_head, head_dim)

Calculation (B=1, T=2048, bf16):
  K cache: 12 × 1 × 2048 × 6 × 128 × 2 bytes = 36 MB
  V cache: 36 MB
  Total:   72 MB

Simpler formula:
  2 (K+V) × n_layer × B × T × n_embd × bytes
= 2 × 12 × 1 × 2048 × 768 × 2 = 75 MB ≈ 72 MB
```

| Setting | KV cache size |
|---------|--------------|
| B=1, T=2048 | ~72 MB |
| B=1, T=8192 | ~288 MB |
| B=8, T=2048 | ~576 MB |
| B=8, T=8192 | ~2.3 GB |

**This is why KV cache memory dominates long-context inference** — not parameter memory. A 260M-parameter model uses ~520 MB for weights (bf16) but the KV cache at long context can easily exceed that. GQA reduces KV cache proportionally: `n_kv_head=2` instead of 6 shrinks the cache by 3×.
*Why x = x + f(x) is the most important design decision in the transformer*

---

### What are residual connections?

A residual connection is simply adding the input back to the output of a sublayer:

```python
x = x + self.attn(self.ln_1(x))   # [NC] residual after attention
x = x + self.mlp(self.ln_2(x))    # [NC] residual after MLP
```

That `x +` is the entire residual connection. No parameters. No matrix. Just addition. It has two effects: it solves the vanishing gradient problem, and it changes what each layer learns.

---

### The vanishing gradient problem

**Why gradients shrink without residuals:**

During backpropagation, the gradient travels backward from the loss through every layer. At each layer it gets multiplied by that layer's derivative. For a simple linear layer `output = W × x`, the derivative is just `W`. If weights are small (which they are at the start of training — initialised near zero), the gradient shrinks at every layer it passes through.

```
Weights initialised randomly near zero:
W1=0.08, W2=0.05, W3=0.12, W4=0.09, W5=0.07, W6=0.11

Gradient reaching Layer 1:
= loss_gradient × W6 × W5 × W4 × W3 × W2 × W1
= 1.0 × 0.11 × 0.07 × 0.09 × 0.12 × 0.05 × 0.08
= 0.000000033   ← essentially zero

Layer 1 update: W1 = W1 - lr × 0.000000033 ≈ no change at all
Layer 1 learns nothing.
```

**Why this is not a fixed 0.1:** The shrinkage depends on the actual weight values, which vary. But the principle holds for any small weights — multiplying many numbers less than 1 together always gives a tiny result. `0.5^6 = 0.016`. `0.3^6 = 0.0007`. Deep networks were essentially untrainable before residual connections.

---

### The chain rule — the mathematics behind it

The chain rule answers: "if A affects B, and B affects C, how much does A affect C?" The answer is: multiply the individual rates.

```
Real-world analogy:
  foot → accelerator → engine → car speed
  d(car speed)/d(foot) = d(car)/d(engine) × d(engine)/d(foot) = 3.0 × 2.0 = 6.0

Neural network — 3 layers:
  d(loss)/d(W1) = d(loss)/d(Layer3) × d(Layer3)/d(Layer2) × d(Layer2)/d(W1)
                =       1.0         ×          W3          ×          W2
                =       1.0         ×         0.08         ×         0.05
                =       0.004
```

Every extra layer adds one more multiplication to the chain. Six layers = six multiplications. The deeper you go, the smaller the product.

---

### The actual weight update formula

```python
# 1. Forward pass — compute loss
prediction = model(x)                          # [NC]
loss = F.cross_entropy(prediction, target)      # [PT]

# 2. Backward pass — compute gradient for every weight
loss.backward()    # [PT] PyTorch applies chain rule automatically
# Now every weight W has W.grad = d(loss)/d(W)
# W6.grad ≈ 0.42    (close to loss, gradient healthy)
# W1.grad ≈ 0.000000033  (deep layer, gradient dead)

# 3. Weight update — move in direction that reduces loss
optimizer.step()   # [PT] for plain SGD:
# W = W - learning_rate × W.grad
# W6 = 0.42 - 0.0003 × 0.42    = 0.41987   ← meaningful update
# W1 = 0.35 - 0.0003 × 0.000000033 ≈ 0.35  ← essentially no update
```

The gradient IS the answer to: "which direction should this weight move?" A dead gradient means the weight gets no direction — it stops learning entirely.

---

### How the +1 fixes it — the mathematics

The residual `output = x + f(x)` changes the derivative:

```
Without residual:
  output = f(x) = W × x
  d(output)/dx  = W          ← gradient multiplied by W (could be tiny)

With residual:
  output = x + f(x) = x + W × x
  d(output)/dx  = 1 + W      ← gradient multiplied by (1 + W)
                               ← the 1 is permanent — can never vanish
```

**Why d(output)/dx = 1 + W:**

```
output = x + W × x
       = A + B        where A = x  and  B = W × x

d(output)/dx = d(A)/dx + d(B)/dx
             = d(x)/dx + d(W×x)/dx
             =    1    +    W
```

`d(x)/dx = 1` always — the rate of change of x with respect to itself is always 1. That is where the permanent +1 comes from.

**With 6 layers — the difference:**

```
Without residuals:  1.0 × 0.08 × 0.05 × 0.12 × 0.09 × 0.07 × 0.11 = 0.000000033
With residuals:     1.0 × 1.08 × 1.05 × 1.12 × 1.09 × 1.07 × 1.11 = 1.54
```

---

### The Text Improvement Station — a concrete analogy

Think of a residual block as a **Text Improvement Station on an assembly line**. We track the input `x` — the phrase "The cat" — as it passes through.

**Step 1 — x is the raw context:**
```python
x = [[0.2, 0.1, ...],    # "The" — 384 numbers
     [0.8, -0.3, ...]]   # "cat" — 384 numbers
```

**Step 2 — The fork in the road:**

When x hits the block it splits into two copies:
- One copy goes on the **Highway** (identity path) — unchanged
- One copy goes into the **Factory** (the attention layer) — gets processed

**Step 3 — y is the suggested edits:**

The attention mechanism notices that "The" and "cat" are strongly connected (subject-noun relation). It outputs a correction vector y — not a full rewrite, just small adjustments:

```python
y = [[ 0.05, -0.01, ...],   # small tweak for "The"
     [-0.10,  0.08, ...]]   # small tweak for "cat"
# y values are small — attention is not rewriting x from scratch
# it's saying: "add a little 'subject' flavour to 'cat'"
```

**Step 4 — x = x + y:**

```
  "The cat"     +    "subject flavour"    =    "The cat [subject aware]"
   (original x)      (attention edit y)         (improved output)
```

This improved x flows into the next block for further refinement.

> **x is the story so far. y is the specific edits. x = x + y is the result.**

---

### Backpropagation through a residual block — the inspector analogy

If forward propagation is the assembly line building a prediction, backpropagation is the **quality control inspector walking backward** through the factory to find out who caused the mistake.

**The goal — the blame game:**

At the end of the network, we compute a loss. Backpropagation computes the gradient — a number for every weight that says: "if you increase this weight by a tiny bit, does the error go up or down, and by how much?"

To find this for early layers, we multiply derivatives through every layer — the chain rule.

**Without residuals — the Wall of Death:**

The gradient is like a messenger running backward through a series of thick doors (layers). Each door has a transmission fee (the weight matrix W). If weights are small, the messenger loses energy at every door. By the time it reaches Layer 1, it is whispering so quietly the layer cannot hear it. This is the vanishing gradient problem.

**With residuals — the Superhighway:**

When the inspector walks backward and hits a residual block, they see **two paths**:

```
Path A — The Scenic Route (through the attention/MLP layer):
  The inspector calculates how the weights contributed to the error.
  Gradient gets multiplied by W — could be small.

Path B — The Express Lane (the identity path):
  The inspector walks straight past the factory.
  d(x)/dx = 1 — gradient passes at full volume, always.
```

Because `d(x + y)/dx = 1 + dy/dx`, the gradient **splits** at every residual block:

```python
# Backward pass through one residual block:
g_in = g_out * (1 + dy_dx)
     = (g_out * 1) + (g_out * dy_dx)
#       ↑ Express Lane    ↑ Scenic Route
#       passes at         passes through
#       full volume       the weights (may shrink)
```

**The 1 acts as an elevator** — it lets the error signal from the very last layer fly all the way back to the very first embedding layer without losing volume.

---

### Numerical trace — gradient through 3 residual blocks

Assume the total loss at the end of the model is 10.

```
Loss = 10.0

Block 3: gradient arrives = 10.0
         Express Lane (×1):           10.0  → passes straight through
         Scenic Route (×W=0.1):        1.0  → passes through attention
         Combined to previous layer: 10.0 + 1.0 = 11.0

Block 2: gradient arrives = 11.0
         Express Lane (×1):           11.0  → passes straight through
         Scenic Route (×W=0.1):        1.1  → passes through attention
         Combined to previous layer: 11.0 + 1.1 = 12.1

Block 1: gradient arrives = 12.1
         Express Lane (×1):           12.1  → passes straight through
         Scenic Route (×W=0.1):        1.21 → passes through attention
         Layer 1 receives:           12.1 + 1.21 = 13.31
```

**Without residuals**, those same blocks would give:

```
Block 3: 10.0 × 0.1 = 1.0
Block 2:  1.0 × 0.1 = 0.1
Block 1:  0.1 × 0.1 = 0.01   ← Layer 1 barely updates
```

| | Standard layer | Residual layer |
|--|----------------|----------------|
| Backward math | `g_in = g_out × W` | `g_in = g_out + (g_out × W)` |
| Small weights (W=0.1) | 10 × 0.1 = **1.0** | 10 + (10 × 0.1) = **11.0** |
| Result | Signal vanishes — loses 90% | Signal survives — stays at full volume |

---

### What the layer actually learns — the base case shift

In a standard network, the **default state** of a layer is nothingness (0). If weights are zero, signal dies.

In a residual network, the **default state is identity** (1). If weights are zero:

```python
y = attention(x) ≈ 0         # layer has learned nothing useful yet
output = x + y = x + 0 = x   # signal passes through unchanged
```

A 100-layer model can start training as if it were a 1-layer model. Signal slides through "useless" early layers until it finds a layer that knows how to help. Each layer learns to add a correction — not to rewrite x from scratch.

**This is not cheating the math.** We are not adding 1 to a result — we are changing the function:

```
Old function:      f(x)   = Attention(x)         # default output = 0
Residual function: H(x)   = x + Attention(x)     # default output = x (identity)
```

The +1 in the derivative is the mathematical consequence of x being physically present at the output. It is a direct reflection of the architecture, not a trick.

---

### The 100-story building — final intuition

Imagine your nanochat model is a 100-story building:

**Without residuals:** You have to climb 100 flights of stairs to carry a message to the basement. You are exhausted by floor 10. The basement (Layer 1) never gets the message.

**With residuals:** There is an elevator (the +1 path) that goes to every floor simultaneously. You can hop off the elevator at any floor to deliver a specific message (the attention gradient), then hop back on to reach the basement instantly. The basement always receives the full message at full volume.

This elevator is why we can train models with 100+ layers (GPT-3 has 96) instead of being stuck at 5 or 10.

---

### Why it doesn't explode — the safety valves

If +1 is added at every layer, you might worry the gradient grows to infinity. Two mechanisms prevent this:

```python
# LayerNorm — keeps the "volume" of x in check
x = self.ln_1(x)                   # [PT] normalises x before each sublayer

# Weight Decay — keeps the "edits" y small
optimizer = torch.optim.AdamW(..., weight_decay=0.1)  # [PT]
# nudges weights toward zero → attention corrections y stay small
# the gradient through the scenic route (×W) stays bounded
```

LayerNorm keeps x from growing too large. Weight decay keeps W (and therefore y) small. Together they ensure the gradient is preserved (doesn't vanish) without exploding.

---

### Does the residual change the meaning of training?

No — it makes training *more meaningful*, not less. It changes what each layer is asked to learn.

**Without residuals:** each layer must learn the complete transformation of x.

```
x → [complete rewrite] → x_new
```

**With residuals:** each layer only learns the *correction* to x.

```
x → [what should I add?] → small correction Δx
x_new = x + Δx
```

This is a much easier learning problem. A layer can safely output zero (Δx = 0) when it has nothing to contribute — the signal passes through unchanged. Without residuals, outputting zero would kill the signal entirely.

```python
# What well-trained layers learn with residuals:
attn_output ≈ 0     # "I have nothing useful to add here"
x = x + 0 = x      # x passes through unchanged — safe!

attn_output ≈ Δx   # "add this small correction"
x = x + Δx         # x improved — also safe!
```

This is why residuals are named "residuals" — each layer learns the *residual* (the leftover correction) rather than the full representation.

---

### Does x keep growing? How does LayerNorm relate?

Yes — x accumulates additions from every block and can grow in magnitude. LayerNorm handles this. The pattern is always:

```python
# normalise BEFORE each sublayer — not after the residual addition
x = x + attn(ln_1(x))   # ln_1 normalises the INPUT to attention
x = x + mlp(ln_2(x))    # ln_2 normalises the INPUT to MLP

# x itself is never normalised — it accumulates freely
# But every sublayer always receives a well-scaled input (mean=0, std=1)
# The unnormalised residual stream is intentional — it preserves the gradient highway
```

The residual stream x is the "main road." LayerNorm only applies at the entrance to each sublayer — it cleans up the input before computation but never touches the main road itself.

---

### MLP / FFN — what it does

The MLP (Multi-Layer Perceptron) is Karpathy's code name for what papers call the FFN (Feed-Forward Network). They are the same thing.

**The one-line summary:**
- Attention = tokens *talk* to each other (cross-sequence communication)
- MLP = each token *thinks* by itself (per-token computation, no cross-talk)

After attention, each token has gathered information from the rest of the sequence. The MLP then processes that information independently — the same MLP applied to each of the T token positions, with no information crossing between them.

```python
class MLP(nn.Module):                   # [NC] same as FFN in papers
    def __init__(self, config):
        super().__init__()              # [PT]
        self.c_fc   = nn.Linear(        # [PT] W_fc — expand
            config.n_embd,
            4 * config.n_embd           # 384 → 1536 (4× expansion)
        )
        self.gelu   = nn.GELU()         # [PT] activation function
        self.c_proj = nn.Linear(        # [PT] W_proj — compress
            4 * config.n_embd,
            config.n_embd               # 1536 → 384 (back to original)
        )

    def forward(self, x):               # [NC] x: (B, T, 384)
        x = self.c_fc(x)               # [PT] 384 → 1536
        x = self.gelu(x)               # [PT] non-linearity
        x = self.c_proj(x)             # [PT] 1536 → 384
        return x                        # (B, T, 384) — per-token, no cross-token info
```

**The 4× expansion (384 → 1536 → 384):**

```
384 dims → (W_fc) → 1536 dims → (GELU) → 1536 dims → (W_proj) → 384 dims
  ↑ problem on one line   ↑ working space / scratch pad   ↑ answer on one line
```

The expansion gives the network room to compute complex intermediate features that couldn't fit in 384 dimensions. Research shows the MLP layers store factual knowledge — things like "Paris is in France", "the sky is blue." The 4× ratio is empirically established — large enough to store rich patterns, efficient enough to train.

**c_proj appears in two places — don't confuse them:**
- Inside `CausalSelfAttention`: `self.c_proj = nn.Linear(384, 384)` — mixes the 6 head outputs
- Inside `MLP`: `self.c_proj = nn.Linear(1536, 384)` — compresses back from 4× expansion
- Same name, different classes, different shapes

---

### The complete transformer Block class

```python
class Block(nn.Module):                         # [NC]
    def __init__(self, config):
        super().__init__()                      # [PT]
        self.ln_1 = nn.LayerNorm(config.n_embd) # [PT] normalise before attention
        self.attn = CausalSelfAttention(config)  # [NC] the attention sublayer
        self.ln_2 = nn.LayerNorm(config.n_embd)  # [PT] normalise before MLP
        self.mlp  = MLP(config)                  # [NC] the FFN sublayer

    def forward(self, x):                       # [NC] x: (B, T, 384)
        x = x + self.attn(self.ln_1(x))         # [NC] residual 1: communicate
        #   ↑           ↑
        #   original x  what attention learned to add
        x = x + self.mlp(self.ln_2(x))          # [NC] residual 2: compute
        #   ↑          ↑
        #   updated x  what MLP learned to add
        return x                                 # (B, T, 384) — same shape, richer meaning
```

**What each line does:**
1. `self.ln_1(x)` — normalise x (mean=0, std=1) before feeding to attention
2. `self.attn(...)` — tokens attend to each other, output is (B, T, 384) correction
3. `x = x + attn_output` — add the correction. Original x preserved.
4. `self.ln_2(x)` — normalise the updated x before MLP
5. `self.mlp(...)` — each token thinks independently, output is (B, T, 384) correction
6. `x = x + mlp_output` — add the correction. Everything preserved.

The entire block is stacked 6 times in `nn.ModuleList([Block(config) for _ in range(config.n_layer)])` `[PT]`. Input shape = output shape = `(B, T, 384)` at every block. The model deepens without the gradient dying because every `+` sign keeps the highway open.

---

### New PyTorch built-ins in Phases 3.2 and 4

| API | What it does |
|-----|-------------|
| `tensor.masked_fill(mask, value)` | Replace positions where mask is True with a value (-inf) |
| `F.softmax(x, dim=)` | Convert scores to probabilities summing to 1.0 per row |
| `tensor.contiguous()` | Make tensor memory layout contiguous (required before .view() after .transpose()) |

---

---

## Phase 3.3 — Multi-Head Attention
*How 6 heads run in parallel, combine, and why c_proj is the synthesis step*

---

### Why multiple heads?

A single attention head computes one set of Q, K, V projections and produces one attention pattern. But natural language has many types of relationships simultaneously — grammatical dependencies, coreference ("it" = what?), topic similarity, positional proximity. One head can only focus on one type at a time.

Multiple heads run the full attention mechanism in parallel, each with its own Q, K, V weight matrices, each developing its own specialisation.

**The constraint that creates specialisation:** Each head only gets 64 of the 384 dimensions. It physically cannot track everything — it is forced to focus. The model learns what each head specialises in during training. You never assign roles manually.

```
Single head:       one (T, T) attention pattern — tries to capture everything at once
Multi-head (n=6):  six independent (T, T) attention patterns — each specialises
```

---

### The full shape journey through CausalSelfAttention

Your understanding is exactly right. Here is the complete picture confirmed:

```
Input x:        (B, T, 384)    ← token vectors entering attention

c_attn(x)  →   (B, T, 1152)   ← fused Q+K+V projection
.split     →   q,k,v: (B,T,384) ← separated
.view      →   (B, T, 6, 64)  ← 6 heads labelled, 64 dims each
.transpose →   (B, 6, T, 64)  ← heads moved to dim 1

Each head independently runs:
  q @ kᵀ   →  (B, 6, T, T)   ← raw scores
  ÷ √64    →  (B, 6, T, T)   ← scaled
  mask -∞  →  (B, 6, T, T)   ← future masked
  softmax  →  (B, 6, T, T)   ← weights sum to 1
  × V      →  (B, 6, T, 64)  ← weighted value output

.transpose →   (B, T, 6, 64)  ← heads back to dim 2
.view      →   (B, T, 384)    ← 6 × 64 concatenated

c_proj     →   (B, T, 384)    ← heads mixed together

Output y:       (B, T, 384)    ← same shape as input x ✓
```

All 6 heads run simultaneously — PyTorch treats `(B × 6)` as the batch dimension and runs one large GPU operation, not a loop.

---

### Why c_proj is necessary — the isolation problem

After concatenation back to `(B, T, 384)`, the tensor looks like this inside:

```
dims   0–63:   exactly what Head 0 computed  (grammar?)
dims  64–127:  exactly what Head 1 computed  (subject-verb?)
dims 128–191:  exactly what Head 2 computed  (coreference?)
dims 192–255:  exactly what Head 3 computed  (topic?)
dims 256–319:  exactly what Head 4 computed  (nearby context?)
dims 320–383:  exactly what Head 5 computed  (long-range?)
```

**The heads have never spoken to each other.** Head 0 ran attention using only dims 0–63 of Q, K, V. Head 1 used only dims 64–127. Their outputs are sitting next to each other in memory but are completely separate — each in its own silo.

If you added this directly to the residual stream without c_proj:

```python
# WITHOUT c_proj — heads never combine:
x = x + y

# dim 7 of x  → updated only by Head 0's output (dim 7 is in range 0–63)
# dim 70 of x → updated only by Head 1's output (dim 70 is in range 64–127)
# dim 200 of x → updated only by Head 3's output
#
# Head 0's grammar insight never touches Head 3's region of x
# Head 3's topic insight never touches Head 0's region of x
# The 6 heads contribute to completely separate parts of x — never combined
```

---

### What c_proj actually does

`c_proj` is `nn.Linear(384, 384, bias=False)` `[PT]` — a `(384, 384)` weight matrix that mixes all 384 input dimensions into all 384 output dimensions simultaneously.

For each output dimension, c_proj has learned a **mixing recipe** — which combination of all 384 input dims (spanning all 6 heads) to draw from:

```
Without c_proj:
  output dim 7  = dim 7 of Head 0's output only
  output dim 70 = dim 6 of Head 1's output only
  Heads isolated. No cross-head influence.

With c_proj:
  output dim 7  = 0.3 × Head0_dim3  +  0.8 × Head2_dim41  +  (-0.2) × Head5_dim12  + ...
  output dim 70 = 0.5 × Head1_dim6  +  0.7 × Head3_dim55  +  0.1 × Head0_dim22   + ...
  Every output dim draws from every head simultaneously.
```

The `(384, 384)` weight matrix has 147,456 numbers — all started random, all adjusted by backpropagation. Over training the model discovers: "to produce a good representation, output dim 7 should combine Head 0's grammar signal with Head 2's coreference signal in these proportions." The weight matrix crystallises that discovery.

```python
y = self.resid_dropout(self.c_proj(y))   # [PT]
# c_proj.weight shape: (384, 384)
# For each token: run 384 dot products
# Each dot product multiplies across all 384 input dims
# Result: (B, T, 384) — every dim is now a blend of all 6 heads
```

---

### The researcher analogy

```
Six researchers each write a separate report on a document:
  Head 0 report: grammar analysis
  Head 1 report: subject-verb relationships
  Head 2 report: coreference ("it" refers to "the cat")
  Head 3 report: topic classification
  Head 4 report: nearby word relationships
  Head 5 report: long-range dependencies

Concatenation = stapling the 6 reports together (still separate)

c_proj = one editor reads ALL 6 reports and writes a
         single unified synthesis that draws insights
         from all of them: "Head 0 found X, Head 3 found Y,
         together they imply Z — here is the combined conclusion."
```

---

### What happens if you remove c_proj?

The model would still train — but each head's insight would go into its own silo of the residual stream and never influence the others. The 6 heads would effectively be 6 independent single-head attentions that never combine. c_proj is what makes multi-head attention genuinely more powerful than running 6 separate single-head attentions.

---

### The complete CausalSelfAttention class — all of Phase 3

```python
class CausalSelfAttention(nn.Module):              # [NC]
    def __init__(self, config):
        super().__init__()                         # [PT]
        assert config.n_embd % config.n_head == 0  # [NC] 384 ÷ 6 = 64 exactly

        self.c_attn = nn.Linear(                   # [PT] fused Q+K+V, all heads
            config.n_embd, 3 * config.n_embd, bias=False)  # 384 → 1152

        self.c_proj = nn.Linear(                   # [PT] output — mixes heads
            config.n_embd, config.n_embd, bias=False)      # 384 → 384

        self.attn_dropout  = nn.Dropout(config.dropout)    # [PT]
        self.resid_dropout = nn.Dropout(config.dropout)    # [PT]
        self.n_head = config.n_head   # 6                  # [NC]
        self.n_embd = config.n_embd   # 384                # [NC]

        self.register_buffer('bias',               # [PT] causal mask — not a parameter
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))
        # shape: (1, 1, 1024, 1024) — broadcasts across B and n_head
        # 1s below diagonal = allowed, 0s above = masked to -inf

    def forward(self, x):                          # [NC]
        B, T, C = x.shape

        # ── Phase 3.1: project → split → reshape into heads ─────
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)        # [PT]
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # [PT]
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # [PT]
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)  # [PT]
        # q, k, v: (B, 6, T, 64) — 6 heads, ready to run in parallel

        # ── Phase 3.2: scaled dot-product causal attention ───────
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))  # [PT] scale
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))   # [PT] mask
        att = F.softmax(att, dim=-1)                                    # [PT] softmax
        att = self.attn_dropout(att)                                    # [PT]
        y   = att @ v                                                   # [PT] weighted sum
        # y: (B, 6, T, 64) — 6 independent head outputs

        # ── Phase 3.3: reassemble → mix heads ────────────────────
        y = y.transpose(1, 2).contiguous().view(B, T, C)              # [PT]
        # (B,6,T,64) → (B,T,6,64) → (B,T,384) — concatenated, still isolated
        y = self.resid_dropout(self.c_proj(y))                        # [PT]
        # (B,T,384) → (B,T,384) — now mixed across all 6 heads

        return y   # (B, T, 384) — same shape as input x
```

---

### Phase 3 — complete key takeaways

1. **Q asks, K advertises, V delivers.** Three separate learned projections from the same embedding, each answering a different question. K determines whether a token gets attended to. V determines what flows.

2. **The fused c_attn trick** — one `nn.Linear(384, 1152)` computes all Q, K, V for all heads in one GPU operation. `.split(384, dim=2)` separates them.

3. **view → transpose splits into heads.** `.view(B,T,6,64)` labels which 64 dims belong to each head. `.transpose(1,2)` moves heads to dim 1 for batched matmul. Must be done in this order — view reads memory left-to-right.

4. **Scale → mask → softmax.** Divide by √64 to prevent softmax collapse. Mask -∞ to enforce past-only attention. Softmax row by row — each query gets a probability distribution over past tokens.

5. **att @ v is the retrieval.** Attention weights say how much to attend to each token. V says what flows when attended to. Weighted sum blends past value vectors in proportion to relevance.

6. **Concatenation is just the reverse of the split.** `.transpose(1,2).contiguous().view(B,T,384)`. The heads are placed side by side — still isolated, not yet mixed.

7. **c_proj is the synthesis step.** `nn.Linear(384, 384)` — a `(384×384)` matrix of learned mixing recipes. Each output dim draws from every input dim (every head). Turns 6 isolated perspectives into one unified representation. Without it, the heads never combine.

8. **Shape is preserved end to end.** `(B, T, 384)` enters CausalSelfAttention. `(B, T, 384)` leaves. The residual `x = x + y` requires this — you cannot add tensors of different shapes.

9. **New PyTorch built-ins in Phase 3:**

| API | What it does |
|-----|-------------|
| `tensor.masked_fill(mask, value)` | Replace positions where mask is True with a fixed value (-inf) |
| `F.softmax(x, dim=-1)` | Convert scores to probabilities summing to 1.0, applied row by row |
| `tensor.contiguous()` | Force contiguous memory layout — required before .view() after .transpose() |
| `tensor.masked_fill` | Sets -inf where causal mask is 0 — enforces past-only attention |

---

---

## Phase 4 — The Complete Transformer Block
*Assembling every component into the Block class, stacking 6 into GPT, and understanding every line of GPT.forward()*

---

### 4.1 — The residual stream — what it is and what ln_1 actually does

**The residual stream is the `x` variable flowing through the model.** It is called a stream because it flows continuously — accumulating additions from each sublayer without ever being replaced or directly normalised inside the blocks. The corrections each sublayer adds are the "residuals" — each sublayer learns to add a small improvement, not to rewrite x from scratch.

**What ln_1 does — it normalises a COPY, not x itself:**

The compact nanochat version is:
```python
x = x + self.attn(self.ln_1(x))   # [NC]
x = x + self.mlp(self.ln_2(x))    # [NC]
```

Expanded to make every step explicit:

```python
def forward(self, x):                         # [NC]  x: (B, T, 384)

    # ── Attention half ────────────────────────────────────────────
    normalised = self.ln_1(x)                 # [PT]  copy of x, mean=0 std=1
                                              #       x itself is NOT changed
    correction = self.attn(normalised)        # [NC]  full attention → (B,T,384)
    x          = x + correction              # [NC]  residual: original + what attn learned

    # ── MLP half ──────────────────────────────────────────────────
    normalised = self.ln_2(x)                 # [PT]  copy of updated x, mean=0 std=1
    correction = self.mlp(normalised)         # [NC]  FFN 384→1536→384 → (B,T,384)
    x          = x + correction              # [NC]  residual: post-attn + what MLP concluded

    return x                                  # [NC]  (B, T, 384) — same shape as input
```

**The flow through one Block — what gets normalised vs what doesn't:**

```
ENTERING BLOCK
x: (B, T, 384)   ← the residual stream. Raw, unnormalised.
│
│  ┌─────────────────────────────────────────────────────────┐
│  │  ln_1(x)  →  normalised copy  →  attn()  →  correction │
│  └─────────────────────────────────── correction ──────────┤
│                                                             ▼
x ═══════════════════════════════════════════════════════════ + → x (updated)
│                                                             ▲
│  ┌─────────────────────────────────────────────────────────┤
│  │  ln_2(x)  →  normalised copy  →  mlp()   →  correction │
│  └─────────────────────────────── correction ──────────────┘
│
LEAVING BLOCK
x: (B, T, 384)   ← same shape, richer meaning

Key:
  ════  x (residual stream) flows straight through — never normalised inside blocks
  ─────  copies of x go into sublayers, get normalised, produce corrections
  +     corrections added back to x (the residual additions)
```

**Why does x grow and why does ln_f matter?**

After 6 blocks of residual additions, x accumulates values from every block:

```python
# x after each block (values grow with each addition):
x = wte(idx) + wpe(pos)     # (B,T,384)  starting point — small values
x = x + attn_0 + mlp_0      # Block 0 adds its corrections
x = x + attn_1 + mlp_1      # Block 1 adds its corrections
x = x + attn_2 + mlp_2      # Block 2 ...
x = x + attn_3 + mlp_3      # Block 3 ...
x = x + attn_4 + mlp_4      # Block 4 ...
x = x + attn_5 + mlp_5      # Block 5 — x can be quite large by now

x = self.transformer.ln_f(x) # [PT]  ← first and ONLY time x itself is normalised
logits = self.lm_head(x)     # [PT]  ← receives consistent, well-scaled input
```

Inside blocks: only the copies going into sublayers are normalised (ln_1, ln_2). The residual stream x is never touched directly. `ln_f` is the one exception — it normalises x itself right before `lm_head`, ensuring the output projection always receives a stable input.

---

### 4.2 — The complete Block class

```python
class Block(nn.Module):                          # [NC]
    def __init__(self, config):
        super().__init__()                       # [PT]
        self.ln_1 = nn.LayerNorm(config.n_embd) # [PT] normalise before attention
        self.attn = CausalSelfAttention(config)  # [NC] full Phase 3 attention
        self.ln_2 = nn.LayerNorm(config.n_embd)  # [PT] normalise before MLP
        self.mlp  = MLP(config)                  # [NC] FFN: 384 → 1536 → 384

    def forward(self, x):                        # [NC] x: (B, T, 384)
        x = x + self.attn(self.ln_1(x))          # [NC] communicate + residual
        x = x + self.mlp(self.ln_2(x))           # [NC] compute + residual
        return x                                  # (B, T, 384) — same shape
```

That is the entire Block. 8 lines. Every component you learned in Phases 2–3 assembled.

**What each line of forward() does — read inside out:**

| Step | Code | What it does |
|------|------|-------------|
| 1 | `self.ln_1(x)` | Normalise a copy of x — mean=0, std=1. x itself unchanged. |
| 2 | `self.attn(...)` | Full CausalSelfAttention: Q/K/V, 6 heads, scores, mask, softmax, weighted V, c_proj. Returns (B,T,384) correction. |
| 3 | `x = x + ...` | Residual: add correction to original x. Gradient highway: d/dx = 1 + d(attn)/dx. |
| 4 | `self.ln_2(x)` | Normalise the post-attention x before MLP. Separate LayerNorm, separate γ/β. |
| 5 | `self.mlp(...)` | FFN: 384→1536→GELU→384. Per-token, no cross-token info. Each token thinks alone. |
| 6 | `x = x + ...` | Residual again. x is now post-attention + MLP contribution. |

**Two roles, one block:**
- Attention half: tokens **communicate** — each token gathers from others via Q·K·V
- MLP half: tokens **compute** — each token processes what it gathered, independently

Shape in: `(B, T, 384)`. Shape out: `(B, T, 384)`. Always. Every block.

---

### 4.3 — What lm_head does — the matrix multiply explained

After 6 blocks, `x` is `(B, T, 384)` — each token position has a 384-dim vector encoding rich contextual meaning. `lm_head` is `nn.Linear(384, 50257, bias=False)` — its weight matrix is `(50257, 384)`.

**What the multiply does:**

For each of the B×T token positions, compute one dot product per vocabulary token:

```python
# Context vector for "sat" after 6 blocks:
context = [0.82, -0.41, 0.33, ...]   # 384 numbers

# lm_head.weight shape: (50257, 384) — one row per vocabulary token
# row 0    → "!"      [0.12, -0.45,  0.33, ...]
# row 257  → " "      [0.02,  0.11, -0.03, ...]
# row 2746 → "model"  [0.21, -0.54,  0.88, ...]

# For each of the 50257 rows, compute dot product:
score_0    = 0.82*0.12 + (-0.41)*(-0.45) + 0.33*0.33 + ...  # = 2.1  "!"
score_257  = 0.82*0.02 + (-0.41)*0.11    + 0.33*(-0.03) + ...# = 0.3  " "
score_2746 = 0.82*0.21 + (-0.41)*(-0.54) + 0.33*0.88 + ...  # = 8.7  "model" ← highest

# Result: 50257 logit scores for this one token position
# Stacked across all T=1024 positions → shape (B, T, 50257)
```

**This is a form of similarity.** Each dot product measures how much the context vector "points toward" that vocabulary token's direction in 384-dim space. A high dot product means alignment — "the context is similar to what this token's embedding looks like." It's not cosine similarity (which divides by magnitudes) but the same geometric idea: alignment = high score.

**Why not cosine similarity?** The dot product is simpler, faster, and differentiable. Magnitudes are handled implicitly by training. Also, lm_head shares weights with wte (weight tying) — the same matrix that encoded tokens at the entrance is reused to score them at the exit. The entrance and exit use identical geometry.

---

#### lm_head — what it actually does (and the naming confusion)

`lm_head` has nothing to do with the attention heads. The name is short for **language model head** — it is the output layer of the entire model.

By the time tokens reach `lm_head`, all the attention head business is long finished. The 6 attention heads ran, concatenated, went through `c_proj`, added back to the residual stream, went through MLP — all of that is done inside the 6 transformer blocks. What comes out is `x` of shape `(B, T, 384)` — one clean 384-dim vector per token, representing its full contextual meaning.

`lm_head` then does one simple thing:

```python
logits = self.lm_head(x)    # [PT]
# nn.Linear(384, 50257, bias=False)
# (B, T, 384) → (B, T, 50257)
# For each token position: "which of the 50,257 vocabulary tokens
#                           is most likely to come next?"
```

It projects from 384 dimensions to 50,257 dimensions — one score per vocabulary token. That is it.

**The naming confusion — two completely different things called "head":**

| | Attention heads | lm_head |
|--|----------------|---------|
| **Count** | 6 per block × 6 blocks = 36 total | 1, at the very end |
| **What it is** | Split of 384 dims into 6 × 64-dim subspaces | Single `nn.Linear(384, 50257)` |
| **Purpose** | Let tokens attend to each other | Predict the next token |
| **Output shape** | Recombined back to `(B, T, 384)` via c_proj | `(B, T, 50257)` logits |
| **When it runs** | Inside each Block, during Phase 3 | After all 6 blocks, Phase 5 |

**And softmax is not inside lm_head — it is a separate step:**

```python
# Training — F.cross_entropy applies softmax internally:
logits = self.lm_head(x)                        # [PT] raw scores only
loss   = F.cross_entropy(                        # [PT] softmax happens inside here
    logits.view(-1, 50257),
    targets.view(-1)
)

# Inference — softmax applied explicitly after lm_head:
logits     = self.lm_head(x)                    # [PT] raw scores
probs      = F.softmax(logits[:, -1, :], dim=-1) # [PT] softmax here
next_token = torch.multinomial(probs, 1)         # [PT] sample
```

`lm_head` only produces raw scores (logits). Softmax is always a separate step — either fused inside `F.cross_entropy` during training, or applied explicitly during generation.

**The complete flow — where lm_head sits:**

```
Transformer blocks (Phases 3 + 4)
  6 attention heads × 6 blocks = 36 attention computations
  All combined back to (B, T, 384) via c_proj and residuals
        ↓
  ln_f — normalise the residual stream
        ↓
  lm_head — nn.Linear(384, 50257)
  "given this 384-dim summary of context, score each vocabulary token"
        ↓
  logits (B, T, 50257) — raw scores, no softmax yet
        ↓
  softmax (training: inside F.cross_entropy / inference: explicit)
        ↓
  probabilities over 50,257 tokens
```

`lm_head` is the exit door. The attention heads are the thinking that happened inside the blocks. They are unrelated beyond sharing the word "head."

#### lm_head — the full pipeline (training vs inference)

`lm_head` produces raw logits only. What happens after it diverges depending on whether you are training or generating:

```
Hidden states from final block:   (B, T, 768)
              ↓
    lm_head: Linear(768, 32768)
              ↓
Raw logits:                        (B, T, 32768)
              ↓
    softcap: 15 × tanh(logits / 15)    ← smoothly caps to ±15 (nanochat modern addition)
              ↓
Capped logits:                     (B, T, 32768)
              │
              ├──► [Training]
              │      F.cross_entropy(logits, targets)
              │      → applies log-softmax internally
              │      → compares against target token
              │      → loss scalar
              │
              └──► [Inference]
                     logits / temperature
                     → F.softmax(logits, dim=-1)     ← softmax here, not inside lm_head
                     → torch.multinomial(probs, 1)   ← sample one token
                     → next token ID
```

**Why softmax, not sigmoid?**

Sigmoid maps each logit independently to (0, 1) — used for binary or multi-label problems where multiple options can be true simultaneously. Softmax normalises all logits together so they sum to 1 — used for multi-class problems where exactly one option is chosen.

Next-token prediction is multi-class: pick exactly one token from 32,768 possibilities. Softmax is the correct tool. The 32,768 scores must compete against each other and redistribute probability mass — sigmoid cannot do this.

**The softcap (`15 × tanh(logits / 15)`):**

A modern addition not in the original GPT-2. After lm_head, extreme logit values (say +847 or -312) cause numerical instability in softmax. The tanh smoothly clamps any value into the range (−15, +15) without hard-clipping — gradients still flow cleanly through tanh at these boundaries. This is another technique from the modded-nanoGPT speedrun community.

---

### Value embeddings — why vocab × n_embd, not n_embd × n_embd

A natural question after seeing value embeddings: why does each layer's value embedding table have shape `(vocab_size, n_embd)` = `(32768, 768)` rather than the smaller `(768, 768)` that attention weight matrices use?

**The answer is the difference between a lookup table and a transformation:**

```python
# c_v — a transformation (768 × 768):
v_dynamic = c_v(x)           # takes a 768-dim hidden state → produces 768-dim V vector
                               # works on whatever vector flows through
                               # does NOT know which specific token this is

# value_embed — a lookup table (32768 × 768):
v_static = value_embed[token_ids]   # takes a token ID integer → looks up its row
                                     # produces a vector specific to THIS token type
                                     # DOES know which specific token this is
```

A `(768, 768)` matrix is applied to the hidden state — it transforms whatever 768-dim vector arrives. It cannot say "token ID 8417 specifically gets this vector" because it never sees the token ID. It only sees the already-embedded 768-dim representation.

A `(32768, 768)` table has one dedicated row per token type. Token ID 8417 always gets row 8417, regardless of context. This is fundamentally different: it is **identity-based** retrieval, not context-based transformation.

**The two V paths serve completely different purposes:**

```
Standard V path (c_v, every layer):
  "What value does this contextualised representation contribute?"
  Input: 768-dim hidden state (context-dependent)
  Shape: (768, 768) — a transformation rule

Value embedding path (selected layers):
  "What value does THIS SPECIFIC TOKEN TYPE contribute?"
  Input: integer token ID
  Shape: (32768, 768) — a lookup table with one row per token type

Combined: v = v_dynamic + gate × v_static
```

If value embeddings used a `(768, 768)` matrix, they would just be another transformation of the hidden state — functionally redundant with `c_v`. The entire point is to provide a signal that is **not** derived from the contextualised hidden state.

**Could the tables be shared across layers?**

Yes — one shared `(32768, 768)` table used by all VE layers would cost less. But per-layer tables are more expressive: "cat" can contribute a different static signal at layer 1 (where syntactic features dominate) vs layer 11 (where semantic features dominate). The model learns what raw token identity means at each depth. nanochat uses per-layer tables, kept to 6 layers (alternating) to cap the cost.

**Parameter cost comparison:**

| Shape | Type | Per-layer cost | What it represents |
|-------|------|---------------|-------------------|
| `(768, 768)` | Linear | ~590K params | Transformation of hidden state — redundant with c_v |
| `(32768, 768)` | Embedding | ~25M params | One static V vector per token type — what VE actually is |

The 42× cost difference is the price of true token-identity lookup over hidden-state transformation. nanochat judges this worth paying because: (1) lookup is cheap at inference (no matmul), (2) the signal is qualitatively different from anything c_v can produce, (3) it empirically speeds up training convergence.

This is the key insight that ties everything together.

By the time a token reaches lm_head, its 384-dim vector has absorbed everything:

```python
# What goes into one token's 384-dim vector:
token_vector = (
    wte_embedding          # what this token type means (static)
    + wpe_embedding        # where it sits in the sequence
    + attn_corrections[0]  # what it gathered from Block 0's attention
    + mlp_corrections[0]   # what Block 0's MLP concluded
    + attn_corrections[1]  # Block 1 ...
    + mlp_corrections[1]
    + ...                  # Blocks 2–5
    + attn_corrections[5]
    + mlp_corrections[5]
)
# = one 384-dim float vector
# = compressed summary of everything relevant for predicting the next token
```

**The 384 is the bottleneck that forces compression.** The model cannot store the whole sequence in 384 numbers — it must learn what matters. Attention decides what to pull in. MLP processes it. The 384-dim vector at lm_head is the model's best compressed summary of "relevant context for what comes next."

**One sequence gives T independent training examples for free:**

```
Sequence: "The  cat  sat  on  the  mat"
           T0   T1   T2   T3   T4   T5

T0 "The": given [The]                          → predict "cat"
T1 "cat": given [The, cat]                     → predict "sat"
T2 "sat": given [The, cat, sat]                → predict "on"
T3 "on":  given [The, cat, sat, on]            → predict "the"
T4 "the": given [The, cat, sat, on, the]       → predict "mat"
T5 "mat": given [The, cat, sat, on, the, mat]  → predict "."
```

Six training examples from one sequence of length 6. At T=1024 with B=12: **12,288 independent training examples per forward pass**, all computed simultaneously.

**The causal mask ensures correctness.** T2's 384-dim vector only contains information from T0, T1, T2 — never from T3, T4, T5. So when lm_head uses T2's vector to predict "on", it genuinely hasn't seen "on" yet. The mask enforces this during the attention computation — future tokens score -∞ before softmax, becoming exactly 0 weight.

---

### 4.5 — Stacking 6 Blocks — the GPT model

```python
self.transformer = nn.ModuleDict(dict(             # [PT]
    wte  = nn.Embedding(vocab_size, n_embd),       # [PT] token embeddings
    wpe  = nn.Embedding(block_size, n_embd),       # [PT] position embeddings
    drop = nn.Dropout(dropout),                    # [PT]
    h    = nn.ModuleList([                         # [PT] ← must be ModuleList
        Block(config)
        for _ in range(config.n_layer)             # [NC] 6 identical Blocks
    ]),
    ln_f = nn.LayerNorm(n_embd)                    # [PT] normalise residual stream
))
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # [PT]

# Weight tying — entrance and exit share the same matrix:
self.lm_head.weight = self.transformer.wte.weight  # [NC]
```

**Why `nn.ModuleList` not a plain Python list?**

```python
# WRONG — PyTorch cannot see parameters inside:
self.blocks = [Block(config) for _ in range(6)]
# model.parameters() won't include these — AdamW won't update them

# CORRECT — PyTorch tracks all parameters in all 6 blocks:
self.blocks = nn.ModuleList([Block(config) for _ in range(6)])  # [PT]
# All parameters in all 6 blocks appear in model.parameters()
# All get updated by AdamW every training step
```

**Parameter count for this config:**
```
wte:         50257 × 384  = 19.3M
wpe:          1024 × 384  =  0.4M
Per Block:
  ln_1:          2 × 384  = 0.001M  (γ and β — no weight decay)
  c_attn:  384 × 1152     =  0.4M   (fused Q+K+V)
  c_proj:  384 × 384      =  0.1M   (attention output)
  ln_2:          2 × 384  = 0.001M  (γ and β — no weight decay)
  c_fc:    384 × 1536     =  0.6M   (MLP expand)
  mlp_proj:1536 × 384     =  0.6M   (MLP compress)
  Block total:            ≈  1.7M
6 blocks:                 = 10.2M
ln_f + lm_head (weight tied to wte):  = 0.0M extra
─────────────────────────────────────────────
Total:                    ≈ 29.9M parameters
```

---

### 4.6 — The complete GPT.forward()

```python
def forward(self, idx, targets=None):              # [NC]
    B, T = idx.shape                               # [NC]

    # ── Phase 2: Embeddings ─────────────────────────────────────
    tok_emb = self.transformer.wte(idx)            # [PT] (B,T) → (B,T,384)
    pos     = torch.arange(T, device=idx.device)   # [PT] [0,1,...,T-1]
    pos_emb = self.transformer.wpe(pos)            # [PT] (T,384) → broadcasts
    x = self.transformer.drop(tok_emb + pos_emb)   # [PT] (B,T,384)

    # ── Phases 3+4: Transformer blocks ──────────────────────────
    for block in self.transformer.h:               # [NC]
        x = block(x)                               # [NC] (B,T,384) → (B,T,384) × 6
    x = self.transformer.ln_f(x)                   # [PT] normalise residual stream

    # ── Phase 5: Output projection ───────────────────────────────
    logits = self.lm_head(x)                       # [PT] (B,T,384) → (B,T,50257)

    # ── Loss — only computed during training ─────────────────────
    loss = None                                     # [NC]
    if targets is not None:                         # [NC]
        loss = F.cross_entropy(                     # [PT]
            logits.view(-1, logits.size(-1)),       # [NC] (B*T, 50257)
            targets.view(-1)                        # [NC] (B*T,)
        )
    return logits, loss
```

**`if targets is not None` — one function, two modes:**
- **Training:** pass `idx` and `targets` → get `logits` and `loss` → `loss.backward()` → `optimizer.step()`
- **Inference:** pass `idx` only, `targets=None` → get `logits` only → softmax → sample next token

The loss branch simply doesn't execute during generation.

---

### 4.7 — F.cross_entropy and the .view() calls explained

**The shape problem:**

```
logits shape:  (B, T, 50257)   ← predictions for every position
targets shape: (B, T)          ← correct answer for every position

F.cross_entropy expects:
  predictions: (N, num_classes)   ← must be 2D
  targets:     (N,)               ← must be 1D
```

**What .view() does:**

```python
logits.view(-1, logits.size(-1))
# logits: (B=12, T=1024, 50257)
# -1 means "figure this dimension out automatically"
# 12 × 1024 = 12288
# result: (12288, 50257) ← 12288 rows, each with 50257 scores

targets.view(-1)
# targets: (B=12, T=1024)
# result: (12288,) ← 12288 correct token IDs, flattened
```

**What cross_entropy computes for each of the 12,288 positions:**

```python
# For each position i (out of 12288):
#   logits[i] = [2.1, 0.3, 8.7, 6.2, ...]   (50257 scores)
#   targets[i] = 2746                         (correct token ID = "model")
#
#   Step 1: softmax(logits[i]) → probabilities summing to 1.0
#   Step 2: look up prob of correct token: probs[2746]
#   Step 3: loss_i = -log(probs[2746])
#
#   If model was confident and correct:   probs[2746] = 0.95 → loss = 0.05  ✓
#   If model was uncertain:               probs[2746] = 0.50 → loss = 0.69
#   If model was confident and wrong:     probs[2746] = 0.01 → loss = 4.61  ✗
#
# Final loss = mean of all 12,288 individual losses = one scalar

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # [NC] (B*T, 50257)
    targets.view(-1)                     # [NC] (B*T,)
)
# F.cross_entropy fuses softmax + log + negate internally
# Numerically stable — never apply softmax manually before this
```

**Why B×T independent positions?**

Yes — multiple unrelated sentences run simultaneously. B=12 sentences, each T=1024 tokens, all batched together for GPU efficiency. The causal mask already ensured sentences didn't contaminate each other during attention — each sentence ran in its own row of the `(B, T, 384)` tensor. The `.view(-1)` just collapses the batch and time dimensions for arithmetic convenience. Each of the 12,288 positions is genuinely independent — its 384-dim vector carries only the causal history within its own sentence.

**The loss is one number that represents the model's average wrongness** across all 12,288 predictions in this batch. That number flows backward through every layer, and every weight nudges itself slightly to reduce it.

---

### 4.8 — End-to-end shape trace — the complete pipeline

| Operation | Shape | Notes |
|-----------|-------|-------|
| `get_batch()` | `(B, T)` | int64 — raw token IDs from train.bin |
| `wte(idx)` | `(B, T, C)` | **C=384 appears** — 2D becomes 3D |
| `+ wpe(pos)` | `(B, T, C)` | position added via broadcasting |
| `Block 0: ln_1 → attn → +x` | `(B, T, C)` | communicate: tokens talk to each other |
| `Block 0: ln_2 → mlp → +x` | `(B, T, C)` | compute: each token thinks alone |
| `Blocks 1–5 (same pattern)` | `(B, T, C)` | shape never changes through any block |
| `ln_f` | `(B, T, C)` | normalise residual stream itself |
| `lm_head(x)` | `(B, T, V)` | **V=50257 appears** — only 2nd shape change |
| `.view(-1, 50257)` | `(B*T, 50257)` | flatten for cross_entropy |
| `F.cross_entropy` | `scalar` | one number — minimise this |

`B=12  T=1024  C=n_embd=384  V=vocab_size=50257`

**Two moments the shape changes in the entire model:**

1. `wte`: `(B, T)` → `(B, T, 384)` — C appears
2. `lm_head`: `(B, T, 384)` → `(B, T, 50257)` — V appears

Everything between these two points operates at exactly `(B, T, 384)`. All 6 blocks, all 36 attention heads, all MLP layers — same shape throughout.

---

### Phase 4 — Key takeaways

1. **The residual stream is the `x` variable.** It flows continuously through the model, accumulating additions from each sublayer. It is never directly replaced. ln_1 and ln_2 normalise copies that go into sublayers — x itself is only normalised once, by ln_f before lm_head.

2. **Each Block is two lines.** `x = x + attn(ln_1(x))` communicates. `x = x + mlp(ln_2(x))` computes. Shape in = shape out = `(B, T, 384)`. Always.

3. **lm_head is a matrix multiply producing 50257 scores.** Each score is a dot product between the 384-dim context vector and one row of the weight matrix — how well the context "aligns with" that vocabulary token. This is geometric similarity, not lookup.

4. **Each token position is a complete, independent training example.** Its 384-dim vector is a compressed summary of everything it attended to and computed across 6 blocks. lm_head reads that summary to predict the next token.

5. **`.view(-1, 50257)` flattens for cross_entropy.** B×T = 12,288 independent positions, each with its own loss. Average loss flows backward to update every weight.

6. **`nn.ModuleList` is mandatory.** A plain Python list hides parameters from PyTorch. Only `nn.ModuleList` ensures all 6 blocks' parameters appear in `model.parameters()` and get updated by AdamW.

7. **`if targets is not None`** — one `forward()` function serves both training (with loss) and inference (without).

---

---

## Phase 5 — Training and Generation
*Where nanochat comes alive — the training loop and autoregressive text generation*

---

### 5.1 — Autoregressive Generation

#### The core idea

During training the model saw all T positions simultaneously — one forward pass, T predictions, T losses. During generation it produces **one token at a time**. Each new token is appended to the context and fed back in as the next input. The model's own output becomes its next input. This is called autoregressive generation.

```
Step 1:  "The cat sat"           → model → "on"
Step 2:  "The cat sat on"        → model → "the"
Step 3:  "The cat sat on the"    → model → "mat"
Step 4:  "The cat sat on the mat"→ model → "."
```

Each step is a full forward pass — embeddings, 6 blocks, lm_head. Nothing changes architecturally. The only difference from training: no targets are passed, no loss is computed, and only the **last position's logit** is used to pick the next token. All earlier positions' logits are discarded.

---

#### What happens at each generation step

```python
# Context so far: "The cat sat" → token IDs [464, 3797, 6096]
idx = torch.tensor([[464, 3797, 6096]])       # [NC] shape: (1, T=3)

# 1. Full forward pass — identical to training, no targets
logits, _ = model(idx)                        # [NC] shape: (1, 3, 50257)
# logits[:, 0, :] = predictions after "The"         ← not needed
# logits[:, 1, :] = predictions after "The cat"     ← not needed
# logits[:, 2, :] = predictions after "The cat sat" ← THIS ONE only

# 2. Take only the last position's logits
last_logits = logits[:, -1, :]                # [NC] shape: (1, 50257)

# 3. Convert to probabilities
probs = F.softmax(last_logits, dim=-1)        # [PT] shape: (1, 50257)

# 4. Sample the next token
next_tok = torch.multinomial(probs, 1)        # [PT] e.g. [[319]] = "on"

# 5. Append to context and repeat
idx = torch.cat([idx, next_tok], dim=1)       # [PT] shape: (1, 4)
```

> **Why `logits[:, -1, :]`?** The last position has attended to the entire context and is predicting what comes next. Earlier positions predicted tokens we already have — they're not useful during generation.

---

#### Greedy vs sampling — two ways to pick the next token

```python
# Option A — Greedy: always pick highest probability
next_tok = probs.argmax(dim=-1)               # [PT] deterministic

# Option B — Sampling: pick weighted random  ← nanochat uses this
next_tok = torch.multinomial(probs, 1)        # [PT] stochastic
```

| | Greedy | Sampling |
|--|--------|----------|
| **How** | Always picks highest prob token | Picks randomly weighted by probs |
| **Output** | Deterministic — same every run | Different every run |
| **Problem** | Repetitive, gets stuck in loops | Occasional low-quality picks |
| **Use when** | Exact reproducibility needed | Natural, diverse text generation |

Greedy tends to produce "the the the the" — it keeps picking the most probable token even when that creates repetition. Sampling picks "on" 62% of the time, "by" 18%, "near" 10% — maintaining diversity and producing more natural-sounding text.

---

#### Temperature — controlling confidence

Temperature is a single number that reshapes the probability distribution by dividing the logits before softmax. It does not change the model weights — only the sampling behaviour.

```python
logits = logits[:, -1, :] / temperature       # [NC] divide BEFORE softmax
probs  = F.softmax(logits, dim=-1)            # [PT]

# temperature = 0.5 → divide by 0.5 = ×2 → gaps between logits grow
#               → "on" 88%, "by" 9%, "near" 3%   ← peaked, conservative
#
# temperature = 1.0 → unchanged
#               → "on" 62%, "by" 18%, "near" 10%  ← model's raw distribution
#
# temperature = 2.0 → divide by 2.0 = ×0.5 → gaps between logits shrink
#               → "on" 32%, "by" 24%, "near" 21%  ← flat, creative/random
```

**Practical values:**
- `0.7–0.9` — focused, coherent creative writing
- `1.0` — model's raw learned distribution
- `1.0–1.2` — more varied, sometimes surprising output
- `→ 0` — approaches greedy (fully deterministic)
- `→ ∞` — approaches uniform random (all tokens equally likely)

---

#### Top-k sampling — restricting the candidate pool

Even with temperature sampling, very low probability tokens can occasionally get picked — producing nonsense words or characters. Top-k solves this by zeroing out all logits below the k-th highest before softmax, guaranteeing only the top k tokens are ever sampled.

```python
if top_k is not None:                                      # [NC]
    v, _ = torch.topk(logits, top_k)                      # [PT] find k largest values
    threshold = v[:, [-1]]                                 # [NC] the kth largest value
    logits[logits < threshold] = float('-inf')             # [NC] zero out the rest
    # exp(-inf) = 0 → those tokens get exactly zero probability after softmax
```

**Without top-k:** All 50,257 tokens eligible. "xkzptq" has 0.00001% probability — can still occasionally get sampled → gibberish in the output.

**With top-k=50:** Only the top 50 tokens eligible. Probabilities renormalised over just 50 tokens. Nonsense tokens are guaranteed zero probability. The quality floor is raised.

**Combined usage:** Temperature and top-k are always applied together, in this order:

```python
# Correct order: temperature → top-k → softmax
logits = logits[:, -1, :] / temperature        # [NC] 1. reshape distribution
if top_k is not None:                          # [NC] 2. restrict pool
    v, _ = torch.topk(logits, top_k)          # [PT]
    logits[logits < v[:, [-1]]] = float('-inf') # [NC]
probs = F.softmax(logits, dim=-1)              # [PT] 3. normalise
next_tok = torch.multinomial(probs, 1)         # [PT] 4. sample
```

Starting point: `top_k=50, temperature=0.8`.

---

#### The complete generate() function

```python
@torch.no_grad()                                           # [PT]
def generate(model, idx, max_new_tokens,                   # [NC]
             temperature=1.0, top_k=None):

    for _ in range(max_new_tokens):                        # [NC]

        # Crop context to block_size if it has grown too long
        idx_cond = (idx if idx.size(1) <= block_size       # [NC]
                    else idx[:, -block_size:])
        # Model's attention mask is fixed at (1024, 1024)
        # Can't feed in more tokens than it was trained on

        # Forward pass — no targets, no loss computed
        logits, _ = model(idx_cond)                        # [NC]

        # Take last position, apply temperature
        logits = logits[:, -1, :] / temperature            # [NC]

        # Optional top-k restriction
        if top_k is not None:                              # [NC]
            v, _ = torch.topk(logits, top_k)              # [PT]
            logits[logits < v[:, [-1]]] = float('-inf')    # [NC]

        # Softmax → probabilities → sample one token
        probs    = F.softmax(logits, dim=-1)               # [PT]
        next_tok = torch.multinomial(probs, 1)             # [PT]

        # Append sampled token and loop
        idx = torch.cat([idx, next_tok], dim=1)            # [PT]

    return idx                                             # [NC]


# Usage:
model.eval()                                              # [PT] disable dropout
context = torch.tensor([[enc.encode("The cat sat")]])     # [NC] (1, T)
out = generate(model, context,                            # [NC]
               max_new_tokens=100,
               temperature=0.8,
               top_k=50)
print(enc.decode(out[0].tolist()))                        # [NC] back to text
```

> **`@torch.no_grad()`** `[PT]` — disables gradient tracking for the entire function. During generation there is no loss, no backward pass, no weight update. Disabling gradients saves memory (no computation graph built) and speeds up inference. Always use this when generating — forgetting it wastes GPU memory silently.

> **`model.eval()`** `[PT]` — switches dropout off. During training dropout randomly zeros values to prevent over-reliance. During inference you want the full, deterministic signal from every neuron. Call `model.eval()` before generating, `model.train()` before resuming training.

---

#### The context cropping line — why it exists

```python
idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
```

The causal mask is fixed at `(1024, 1024)`. If you've been generating for 1200 steps, you can't feed all 1200 tokens in — the model was never trained on sequences longer than 1024. The solution: always feed only the last 1024 tokens. The model loses memory of the very beginning but can keep generating indefinitely. The quality may gradually degrade as important early context scrolls out.

---

### 5.1 — Key takeaways

1. **Autoregressive = one token at a time.** Each forward pass is identical to training except no loss is computed. The model's own outputs feed back in as inputs.

2. **Only the last logit matters.** `logits[:, -1, :]` — the final position has attended to the full context and is predicting next. All earlier positions' logits are discarded during generation.

3. **Sampling over greedy.** `torch.multinomial(probs, 1)` picks weighted random rather than always highest. Avoids repetition, produces natural text.

4. **Temperature divides logits before softmax.** Low (0.5) = peaked = conservative. High (2.0) = flat = creative. 0.7–0.9 is the practical sweet spot for most tasks.

5. **Top-k zeros out the long tail.** Only the top k tokens can ever be sampled. Raises the quality floor by eliminating nonsense tokens. Apply after temperature, before softmax.

6. **`@torch.no_grad()`** is mandatory during inference. No gradients needed — disabling them saves memory and speeds up generation.

7. **New PyTorch built-ins in Phase 5:**

| API | What it does |
|-----|-------------|
| `@torch.no_grad()` | Decorator/context manager — disables gradient tracking |
| `model.eval()` | Switches dropout off for inference |
| `model.train()` | Switches dropout back on for training |
| `torch.topk(tensor, k)` | Returns the k largest values and their indices |
| `tensor.size(dim)` | Returns the size of a specific dimension |
| `torch.save(obj, path)` | Save model checkpoint to disk |
| `torch.load(path)` | Load checkpoint from disk |

---

---

### 5.2 — The Complete Training Loop

#### The 5-step ritual — every training iteration

Every training step follows the same five operations in the same order. Changing the order breaks training.

```python
for step in range(max_iters):                                  # [NC]

    # ① Zero gradients from the previous step
    optimizer.zero_grad(set_to_none=True)                      # [PT]
    # set_to_none=True: sets grads to None instead of 0.0
    # None skips the memory zeroing — slightly faster and uses less memory
    # PyTorch treats None identically to zero when accumulating next gradient

    # ② Forward pass — build computation graph, compute loss
    x, y = get_batch('train')                                  # [NC]
    logits, loss = model(x, y)                                 # [NC]
    # builds the full computation graph PyTorch uses for backprop

    # ③ Backward pass — compute gradients for every weight
    loss.backward()                                            # [PT]
    # walks backward through the computation graph
    # populates W.grad for every parameter in the model
    # uses the chain rule at every layer

    # ④ Clip gradients — safety net against catastrophic updates
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # [PT]
    # if gradient vector magnitude > 1.0: scale everything down proportionally
    # direction preserved, magnitude capped — prevents one bad batch ruining training

    # ⑤ Update weights — AdamW uses grads + m/v buffers
    optimizer.step()                                           # [PT]
    # for every weight:
    #   m = 0.9×m + 0.1×grad           (smooth direction)
    #   v = 0.95×v + 0.05×grad²        (smooth magnitude)
    #   weight -= lr × m/(√v + 1e-8)   (adaptive update)
    #   weight -= lr × 0.1 × weight    (weight decay)
```

> **Why zero_grad before the forward pass, not after?** Convention — clearing at the start of each step ensures you never accidentally accumulate gradients across steps. Some codebases zero after the optimizer step; both work as long as you're consistent.

---

#### Learning rate scheduling — why a fixed rate fails

A fixed learning rate is too aggressive for the entire run:
- **Early in training:** the model is far from a good solution — large steps make fast progress
- **Late in training:** the model is close — large steps overshoot the minimum and loss bounces

nanochat uses a cosine schedule with linear warmup — three phases:

```
Phase 1: Linear warmup   (steps 0 → warmup_iters)
  lr grows from 0 → max_lr linearly
  gives optimizer time to build stable m and v estimates

Phase 2: Cosine decay    (steps warmup_iters → lr_decay_iters)
  lr decays from max_lr → min_lr following a cosine curve
  smooth, gradual reduction — avoids abrupt drops

Phase 3: Flat minimum    (steps > lr_decay_iters)
  lr stays at min_lr
  fine-tuning at a very small rate
```

```python
def get_lr(step):                                              # [NC]
    # Phase 1: linear warmup 0 → max_lr
    if step < warmup_iters:
        return max_lr * step / warmup_iters

    # Phase 2: cosine decay max_lr → min_lr
    if step <= lr_decay_iters:
        ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # Phase 3: stay at min_lr
    return min_lr

# Apply each step — update all param groups:
lr = get_lr(step)                                              # [NC]
for param_group in optimizer.param_groups:                     # [PT]
    param_group['lr'] = lr
```

**Why warmup?** At step 0, model weights are random and AdamW's m and v buffers are zero. A large learning rate on a random model causes chaotic, destructive early updates. Warmup gives the optimizer ~100 steps to build sensible m and v estimates before stepping at full rate.

**Typical nanochat values:**
```
max_lr          = 3e-4    # peak learning rate
min_lr          = 3e-5    # 10× smaller at the end
warmup_iters    = 100     # ~100 warmup steps
lr_decay_iters  = max_iters  # decay for the full run
```

---

#### Validation loss and checkpointing

Training loss always goes down — the model can simply memorise the training data. Validation loss, measured on data the model has **never trained on**, tells you whether it has learned general patterns or just memorised.

```
Good training:              Overfitting:
  train loss ↓                train loss ↓
  val loss   ↓                val loss   ↓ then ↑  ← stop here, use best checkpoint
```

```python
@torch.no_grad()                                               # [PT]
def estimate_loss():                                           # [NC]
    model.eval()                                               # [PT] dropout off
    losses = {}
    for split in ['train', 'val']:
        batch_losses = torch.zeros(eval_iters)                 # [PT]
        for k in range(eval_iters):
            x, y    = get_batch(split)                         # [NC]
            _, loss = model(x, y)                              # [NC]
            batch_losses[k] = loss.item()                      # [PT]
        losses[split] = batch_losses.mean()                    # [PT]
    model.train()                                              # [PT] dropout back on
    return losses

# Every eval_interval steps:
if step % eval_interval == 0:                                  # [NC]
    losses = estimate_loss()                                   # [NC]
    print(f"step {step}: train {losses['train']:.4f} "
          f"val {losses['val']:.4f} lr {lr:.2e}")

    # Save checkpoint only when val loss improves
    if losses['val'] < best_val_loss:                          # [NC]
        best_val_loss = losses['val']                          # [NC]
        torch.save({                                           # [PT]
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step':      step,
            'val_loss':  best_val_loss,
        }, 'best_model.pt')
        print("checkpoint saved")
```

**Why save optimizer state too?** `optimizer.state_dict()` `[PT]` saves AdamW's m and v buffers — all the accumulated momentum the optimizer has built up. If you only save model weights and resume, the optimizer forgets its momentum. The first few hundred steps after resuming are choppy and inefficient until m and v rebuild. Always save both.

**Loading a checkpoint to resume:**
```python
checkpoint = torch.load('best_model.pt')                       # [PT]
model.load_state_dict(checkpoint['model'])                     # [PT]
optimizer.load_state_dict(checkpoint['optimizer'])             # [PT]
step = checkpoint['step']                                      # [NC]
best_val_loss = checkpoint['val_loss']                         # [NC]
```

---

#### The complete training loop — everything assembled

```python
import math
import torch
import torch.nn.functional as F

# ── Setup ──────────────────────────────────────────────────────
model     = GPT(GPTConfig()).to(device)                        # [NC]
optimizer = model.configure_optimizers(                        # [NC]
    weight_decay=0.1,
    learning_rate=3e-4,
    betas=(0.9, 0.95)
)
best_val_loss = float('inf')                                   # [NC]

# ── Training loop ──────────────────────────────────────────────
for step in range(max_iters):                                  # [NC]

    # ── Learning rate schedule ─────────────────────────────────
    lr = get_lr(step)                                          # [NC]
    for pg in optimizer.param_groups:                          # [PT]
        pg['lr'] = lr

    # ── Periodic evaluation + checkpointing ────────────────────
    if step % eval_interval == 0:                              # [NC]
        losses = estimate_loss()                               # [NC]
        print(f"step {step}: train {losses['train']:.4f} "
              f"val {losses['val']:.4f} lr {lr:.2e}")
        if losses['val'] < best_val_loss:                      # [NC]
            best_val_loss = losses['val']                      # [NC]
            torch.save({                                       # [PT]
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step':      step,
                'val_loss':  best_val_loss,
            }, 'best_model.pt')
            print("checkpoint saved")

    # ── The 5-step ritual ──────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)                      # [PT] ① clear
    x, y = get_batch('train')                                  # [NC] ② get data
    logits, loss = model(x, y)                                 # [NC]    forward
    loss.backward()                                            # [PT] ③ backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # [PT] ④ clip
    optimizer.step()                                           # [PT] ⑤ update

# ── Generation after training ──────────────────────────────────
model.eval()                                                   # [PT]
context = torch.zeros((1, 1), dtype=torch.long, device=device) # [PT]
# token ID 0 = blank prompt — model generates from nothing
out = generate(model, context,                                 # [NC]
               max_new_tokens=500,
               temperature=0.8,
               top_k=50)
print(enc.decode(out[0].tolist()))                             # [NC]
```

> **`torch.zeros((1,1))`** as starting context — token ID 0 fed in as a blank prompt. The model generates from nothing, producing whatever patterns it learned from the training data. This is the "hello world" moment for nanochat — the first time it speaks.

---

### 5.2 — Key takeaways

1. **Five steps, always in this order:** zero_grad → forward → backward → clip → step. Changing the order breaks training. Forgetting zero_grad accumulates gradients across steps and the model diverges.

2. **Learning rate scheduling is not optional.** Fixed learning rates are too aggressive late in training. Cosine decay with linear warmup is the standard: ramp up from 0 → max_lr over ~100 steps, then decay smoothly to min_lr = max_lr/10 over the full run.

3. **Warmup lets the optimizer stabilise.** At step 0, AdamW's m and v buffers are zero. Large steps on random weights cause chaotic early updates. ~100 warmup steps builds sensible m and v before full-rate training begins.

4. **Val loss is the real metric.** Train loss always decreases — the model can memorise. Val loss on held-out data tells you if it's generalising. Save a checkpoint every time val loss improves. Use the best checkpoint, not the most recent one.

5. **Save both model and optimizer state.** `model.state_dict()` saves the weights. `optimizer.state_dict()` saves AdamW's m/v momentum buffers. Without the optimizer state, training resumes choppy for hundreds of steps while momentum rebuilds.

6. **`torch.zeros((1,1))`** is the blank prompt — the model's "hello world" starting point for generation after training.

7. **New PyTorch built-ins in Phase 5.2:**

| API | What it does |
|-----|-------------|
| `optimizer.zero_grad(set_to_none=True)` | Clear gradients — set_to_none is faster than zeroing |
| `loss.backward()` | Compute gradients for all parameters via chain rule |
| `optimizer.param_groups` | List of param groups — used to update lr each step |
| `loss.item()` | Extract scalar value from a tensor — needed for logging |
| `torch.save(obj, path)` | Serialise a Python object (checkpoint dict) to disk |
| `torch.load(path)` | Load a checkpoint from disk |
| `model.state_dict()` | All model parameters as an ordered dict |
| `optimizer.state_dict()` | AdamW m/v buffers and settings as a dict |
| `model.load_state_dict(sd)` | Restore model parameters from a saved dict |

---

### Phase 5 — Complete picture

The full nanochat pipeline from raw text to generated output:

```
Raw text file
    ↓ tiktoken.encode()                   Phase 1: Tokenisation
    ↓ save as uint16 → train.bin, val.bin
    ↓
get_batch() → (B, T) int64               Phase 1 output
    ↓
wte + wpe → (B, T, 384)                  Phase 2: Embeddings
    ↓
× 6 Block(attn + mlp) → (B, T, 384)     Phase 3 + 4: Transformer
    ↓
lm_head → (B, T, 50257)                  Phase 4: Output
    ↓
F.cross_entropy → scalar loss
    ↓
loss.backward() + optimizer.step()        Phase 5: Training
    ↓ (repeat for max_iters steps)
    ↓
model.eval() + generate()                 Phase 5: Generation
    ↓
enc.decode() → generated text
```

---

*This journal covers the complete nanochat build — Phases 1 through 5. Return here to add clarifications, deeper dives, or notes from subsequent learning sessions.*

---

## Appendix A — Attention Variants and Modern Efficiency Techniques

*Requires understanding standard attention (Phase 3) first. These are the techniques used in production models beyond nanochat.*

---

### A.1 — The problem with standard attention

You know the attention score matrix is `(B, n_head, T, T)`. At T=2048, B=12, n_head=32 (GPT-3 scale):

```
Score matrix memory:
  12 × 32 × 2048 × 2048 × 4 bytes = 6.4 GB
  just for the scores — before softmax, before multiplying V
```

But the memory problem is worse than the size alone — it is about **where** the computation happens:

```
Standard attention — GPU memory round trips:

  GPU compute cores → write T×T scores to HBM (slow off-chip RAM)
  HBM               → read back to compute softmax
  GPU compute cores → write softmax result to HBM
  HBM               → read back to multiply by V
  GPU compute cores → write output to HBM

4–6 round trips between fast compute and slow memory per attention op.
GPU compute cores sit idle waiting for memory transfers.
This is called being memory-bandwidth bound.
```

---

### A.2 — FlashAttention (Dao et al. 2022)

FlashAttention does not change the **math** at all — the result is identical to standard attention. What it changes is the **order of operations**, keeping data in fast on-chip SRAM instead of writing to slow HBM.

**The key insight — tiling:**

```
Standard attention:
  Compute all T×T scores  →  write to HBM      ← slow
  Read scores from HBM    →  softmax            ← slow
  Write softmax to HBM                         ← slow
  Read softmax from HBM   →  multiply V        ← slow
  Write output to HBM                          ← slow

FlashAttention:
  Take tile of Q (64 rows), tile of K/V
  Compute partial scores in SRAM               ← fast (on-chip)
  Partial softmax correction in SRAM           ← fast (on-chip)
  Accumulate partial output in SRAM            ← fast (on-chip)
  Move to next tile, repeat
  Write final output to HBM once               ← one slow write total
```

The softmax can be computed **incrementally** using the log-sum-exp identity — you do not need the full row in memory at once. The `(T, T)` matrix never exists in HBM.

```
Speed:   2–4× faster wall-clock time
Memory:  O(T²) → O(T) — T×T matrix never materialised
Result:  IDENTICAL to standard attention — not an approximation
```

| Version | Year | Key improvement |
|---------|------|----------------|
| FlashAttention-1 | 2022 | Tiling — eliminates T×T in HBM |
| FlashAttention-2 | 2023 | Better parallelism, ~2× faster than v1 |
| FlashAttention-3 | 2024 | H100-specific async memory, tensor core overlap |

---

### A.3 — Reducing the KV cache: MQA and GQA

The **KV cache** stores past K and V tensors during generation. At each step you need K and V for all previous tokens. With standard MHA this is expensive:

```
KV cache size = 2 × n_layer × n_head × head_dim × T × bytes_per_value
```

For a 7B model generating 2048 tokens: several GB just for the cache.

**Multi-Query Attention (MQA) — Shazeer 2019:**

```
Standard MHA:  Q heads = 32   K heads = 32   V heads = 32
MQA:           Q heads = 32   K heads = 1    V heads = 1

All 32 query heads share ONE key and ONE value head.
KV cache: 32× smaller
Quality:  slightly worse — information bottleneck in shared K/V
Used in:  early PaLM, Falcon
```

**Grouped Query Attention (GQA) — Ainslie et al. 2023:**

```
Standard MHA:  Q heads = 32   K heads = 32   V heads = 32
GQA:           Q heads = 32   K heads = 8    V heads = 8

Groups of 4 query heads share one K/V pair.
KV cache: 4× smaller than MHA
Quality:  close to MHA — sweet spot between MQA and MHA

Used in: LLaMA-2/3, Mistral, Gemma — most modern open models
         This is what n_kv_head in your config implements.
```

```python
# In GPTConfig:
n_head    = 32   # number of query heads
n_kv_head = 8    # number of K/V heads
                 # n_kv_head = n_head  → standard MHA
                 # n_kv_head < n_head  → GQA
```

---

### A.4 — Reducing T² cost: Sliding Window Attention

```
Standard:  every token attends to ALL T past tokens  →  O(T²)
SWA:       every token attends to W nearest tokens   →  O(T × W)

Token at position 1000, W=256:
  Standard → attends to positions 0–999    (1000 tokens)
  SWA      → attends to positions 744–999  (256 tokens only)
```

Information beyond the window still propagates indirectly through multiple layers. Layer 2 can attend to what Layer 1 already absorbed from distant tokens.

```
Used in:   Mistral-7B (W=4096), Longformer
Trade-off: cannot directly attend to tokens > W positions ago
           information propagates through layer depth instead
```

**Longformer** combines sliding window + global tokens:

```
Local:   every token attends to W=512 nearby tokens
Global:  special tokens ([CLS], question tokens) attend to ALL tokens
         and ALL tokens attend back to them

Perfect for: document QA where one token needs full document context
```

---

### A.5 — Better positional encoding: RoPE

**RoPE (Rotary Position Embedding) — Su et al. 2021**

Standard nanochat uses learned absolute positional embeddings (`wpe`) — a 384-dim vector added per position. This works but does not generalise to sequences longer than the training length.

RoPE instead **rotates** Q and K vectors by position-dependent angles before the dot product:

```
Standard PE:  score(i,j) = (Q_i + pos_i) · (K_j + pos_j)
RoPE:         score(i,j) = (R_θ,i × Q_i) · (R_θ,j × K_j)

Key property: the dot product depends only on (i - j) — relative position
              not absolute positions i and j separately
```

```
Benefits:
  Generalises better to longer sequences than learned PE
  Can be extended (RoPE scaling) to 4×–8× training length at inference
  Relative position is more natural for language
  "3 tokens ago" matters more than "position 512 in absolute terms"

Used in: LLaMA, Mistral, GPT-NeoX, Gemma — most modern open models
         Replaced learned wpe in most production models
```

---

### A.6 — Sparse attention

**Sparse Transformer — Child et al. 2019:**

```
Standard: every token → every past token  (dense, T² connections)
Sparse:   each token attends to a fixed subset only

Two patterns combined across heads:
  Local heads:   attend to the W nearest tokens     (nearby context)
  Strided heads: attend to every k-th token         (long-range context)

Cost: O(T × √T) instead of O(T²)
Trade-off: misses some connections — information must hop through multiple layers
```

---

### A.7 — Linear attention

**Linear Attention — Katharopoulos et al. 2020:**

```
Standard: softmax(QKᵀ / √d) × V   →  O(T²)  must form T×T matrix
Linear:   φ(Q) × (φ(K)ᵀ × V)     →  O(T)   reorder the multiplications

The trick — change the order:
  Standard:  (Q × Kᵀ) × V     →  T×T intermediate (grows with sequence)
  Linear:    Q × (Kᵀ × V)     →  d×d intermediate (does not grow with T)

φ() is a kernel function approximating softmax

Status: quality noticeably worse on language — softmax's peaked
        "winner takes most" distribution cannot be replicated by linear kernels
        Active research — has not replaced softmax in practice yet
```

---

### A.8 — What is actually used in 2025

```
Training large models (standard recipe):
  FlashAttention-2/3              ← universal — everyone uses this
  GQA (n_kv_head < n_head)        ← standard in all new models
  RoPE positional encoding         ← replaced learned PE in most models

Efficient inference / long context:
  Sliding Window Attention         ← Mistral, some variants
  KV cache quantisation            ← compress K/V to int8/int4
  Speculative decoding             ← small draft model + large verify model

Very long sequences (100k+ tokens):
  Ring Attention                   ← distributes T dimension across GPUs
  FlashAttention with seq parallel ← same idea

Research / not yet mainstream:
  Linear Attention                 ← quality gap not yet closed
  State Space Models (Mamba)       ← O(T) alternative to attention entirely
  Hybrid (Attention + SSM)         ← Jamba, Zamba — active area
```

---

### A.9 — Summary table

| Variant | What changes | Memory cost | Quality vs MHA | Used in |
|---------|-------------|-------------|----------------|---------|
| Standard MHA | baseline | O(T²) | baseline | GPT-2, nanochat |
| FlashAttention | order of ops only | O(T) | **identical** | all modern training |
| MQA | 1 K/V head shared | KV ÷ n_head | slightly worse | PaLM, Falcon |
| GQA | K/V heads grouped | KV ÷ group_size | close to MHA | LLaMA-2/3, Mistral |
| Sliding Window | attend to W tokens only | O(T×W) | worse for long-range | Mistral-7B |
| Sparse Attention | fixed sparse pattern | O(T√T) | slightly worse | Sparse Transformer |
| Linear Attention | kernel replaces softmax | O(T) | noticeably worse | research |
| RoPE | positional encoding only | same as MHA | same or better | LLaMA, most modern |

---

### A.10 — Hyperparameter decisions and scaling laws

*How are model dimensions (n_layer, n_embd, vocab_size, sequence_len) chosen?*

**The honest answer:** a mix of empirical research, compute budgets, and scaling laws — not derived from first principles.

**vocab_size:** A tokeniser decision. Too small (1k) wastes context window on fragments. Too large (500k) makes the embedding table enormous and rare tokens never get enough training signal. The sweet spot is 32k–100k. Powers of 2 (32,768) are convenient for GPU memory alignment.

**sequence_len:** Set by GPU memory and task requirements. The `(T, T)` attention matrix grows quadratically — doubling T quadruples attention memory. 2,048 became a standard because it fits several paragraphs, trained well on consumer hardware, and GPT-3 used it. Modern models push to 8k–128k using FlashAttention + RoPE scaling.

**n_layer vs n_embd — the depth/width tradeoff:**

```
Doubling n_layer (12 → 24):    parameters ≈ ×2
Doubling n_embd (768 → 1536):  parameters ≈ ×4
```

Width scales parameters faster. For a fixed budget, larger models are proportionally wider. Observed ratios across real models:

| Model | Params | n_layer | n_embd | n_embd / n_layer |
|-------|--------|---------|--------|-----------------|
| GPT-2 small | 117M | 12 | 768 | 64 |
| GPT-2 medium | 345M | 24 | 1024 | 43 |
| GPT-3 | 175B | 96 | 12288 | 128 |
| LLaMA-7B | 7B | 32 | 4096 | 128 |
| LLaMA-70B | 70B | 80 | 8192 | 102 |

No single fixed ratio — but as models grow, width scales faster than depth.

**head_dim converges to 128.** Most modern models use head_dim = 64 or 128 regardless of model size. FlashAttention is optimised for these sizes. Larger models get more heads by increasing n_embd, not by making individual heads larger.

**The Chinchilla finding — the most important ratio:**

Chinchilla (Hoffmann et al. 2022) showed that most models before it were **undertrained** — too many parameters relative to training tokens.

```
Kaplan (2020): given compute C, maximise parameters
               GPT-3: 175B params, 300B tokens

Chinchilla (2022): optimal is to scale both equally
               optimal tokens ≈ 20 × num_parameters

Examples:
  7B  model → needs ~140B tokens minimum
  13B model → needs ~260B tokens minimum
  70B model → needs ~1.4T tokens minimum

GPT-3 (175B params, 300B tokens) should have seen ~3.5T tokens.
LLaMA proved this by training a 7B model on 1T tokens and beating GPT-3.
```

**For nanochat (85M params):** Chinchilla says you need ~1.7B training tokens. The Shakespeare corpus is ~1M tokens — the model has far more capacity than the data can fill, so it memorises rather than generalises. To use an 85M model well you need OpenWebText (9B tokens) or similar.

---

### A.11 — Value Embeddings (ResFormer / Value Residual Learning)

*A very recent technique (2024) — not in most tutorials. nanochat includes it as part of its "modern minimalism" approach.*

#### What are value embeddings?

Standard attention computes V (values) from the **previous layer's hidden states**:

```python
V = c_v(x)    # x is the activations from the previous layer
              # V changes every layer, every context, every token
```

**Value embeddings** add a *second* source for V — a **learned lookup table indexed directly by token ID**, just like `wte`:

```python
# Standard attention:
V_standard = c_v(x)                    # [NC] dynamic — from previous layer

# With value embeddings:
V_static   = value_embed[token_ids]    # [NC] static — direct lookup by token ID
V_combined = V_standard + V_static     # [NC] mix both sources
```

For each token ID, there is a learned "what content does this token contribute to attention" vector that is **the same regardless of layer or context** — a static, token-specific value sitting alongside the dynamic one.

#### The key intuition — shortcut path for token identity

In a deep transformer, by layer 6 or 8, the activations have been transformed so much that the original "this is the token 'cat'" signal has diffused into abstract representations. Value embeddings give attention a **direct path back to raw token identity** at any layer.

It is similar in spirit to residual connections — a way to preserve original information that might otherwise get lost in deep stacks.

```
Without value embeddings:
  Token "cat" → wte → Block 0 → Block 1 → ... → Block 11 → lm_head
  By Block 11, "cat" identity has been heavily transformed

With value embeddings:
  Token "cat" → wte → Block 0 → Block 1 → ... → Block 11 → lm_head
                          ↑          ↑                 ↑
                       VE table   VE table          VE table
  Each VE-enabled block can directly access "cat"'s static token vector
```

#### Where this comes from

**Paper:** "Value Residual Learning For Alleviating Attention Concentration In Transformers" (Zhou et al., 2024) — also called **ResFormer**.

Key findings:
- In deep transformers, attention tends to **concentrate on too few tokens** in later layers — diversity loss
- Adding direct value paths from input embeddings **improves attention diversity** and overall quality
- You do not need it on every layer — alternating works almost as well at half the parameter cost

Popularised by the **modded-nanoGPT speedrun community** (Keller Jordan et al.) who found it as one of several tricks to set training speed records on the FineWeb benchmark. nanochat inherits it from there.

#### Why selective layers — not every layer

```python
def has_ve(layer_idx, n_layer):                           # [NC]
    return layer_idx % 2 == (n_layer - 1) % 2
    # True for alternating layers, always True for the last layer
```

**Cost of one value embedding table:** `vocab_size × n_embd` parameters.

```
nanochat config: vocab_size=50257, n_embd=768
One VE table: 50257 × 768 ≈ 38M params

12 layers, every layer:   12 × 38M = 456M extra params  ← too expensive
12 layers, alternating:    6 × 38M = 228M extra params  ← acceptable
Performance difference:              nearly identical
```

**Why the modular formula guarantees the last layer always gets one:**

```python
# If n_layer = 12 (even):
#   target parity = (12-1) % 2 = 1  → odd-indexed layers get VE
#   last layer index = 11 (odd)      → last layer gets VE ✓

# If n_layer = 13 (odd):
#   target parity = (13-1) % 2 = 0  → even-indexed layers get VE
#   last layer index = 12 (even)     → last layer gets VE ✓

# A simpler layer_idx % 2 == 1 would break for odd n_layer.
# The formula generalises correctly for any n_layer.
```

The last layer is guaranteed a value embedding because it feeds directly into `lm_head` — having a clean token-identity signal there is most valuable right before the output projection.

#### The value embedding gate — controlling the mix

Once you have both `V_dynamic` and `V_static`, nanochat doesn't simply add them with equal weight. A small learned gate controls the blend:

```python
gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
# gate shape: (B, T, n_kv_head) = (B, T, 6)

v = v + gate.unsqueeze(-1) * ve
# gate.unsqueeze(-1):   (B, T, 6, 1)
# ve:                   (B, T, 6, 128)    ← value embedding, per head
# result:               (B, T, 6, 128)    ← V_final = V_dynamic + gate × V_static
```

Breaking down the gate expression right to left:

**Step 1 — `x[..., :self.ve_gate_channels]` — slice the input:**

```python
x shape:                  (B, T, 768)
x[..., :12]:              (B, T, 12)    # only the first 12 channels — 1.5% of embedding
```

Why such a tiny slice? With only 12 input channels, the gate has very limited expressive power — it behaves like a **learned per-layer constant** with mild per-token variation. It is not doing fine-grained content-aware reasoning. It is a near-free learnable knob the model can tune during training.

**Step 2 — `self.ve_gate(...)` — small linear projection:**

```python
self.ve_gate = Linear(ve_gate_channels, n_kv_head, bias=False)
# Linear(12, 6) — only 72 parameters per layer!
# Maps 12-channel slice → 6 gate values (one per head)
```

Each attention head gets its own independent gate value. The entire gate module costs **72 parameters per layer** — essentially free.

**Step 3 — `torch.sigmoid(...)` — squash to (0, 1):**

```
sigmoid(-∞) → 0.0   (ignore the static value embedding)
sigmoid(0)  → 0.5   (mix half and half)
sigmoid(+∞) → 1.0   (fully use the static value embedding)
```

**Step 4 — `3 ×` — rescale to (0, 3):**

The unusual part. Instead of leaving the gate at `(0, 1)`, the output is scaled to `(0, 3)`. This lets the gate **amplify**, not just attenuate:

```
(0, 1) gate:  can only shrink V_static's contribution
(0, 3) gate:  can shrink, match, or amplify — up to 3× the raw magnitude
default (sigmoid ≈ 0.5): gate ≈ 1.5 — already mildly boosted

The model can express: "this static signal is more important than a
single residual step" — something impossible with a (0,1) gate.
```

The factor of 3 is empirical from the modded-nanoGPT speedrun community — worked best in their ablations.

**The complete gate annotated:**

```python
gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
#       ↑        ↑                ↑                ↑
#       │        │                │                └─ 12-channel slice (cheap)
#       │        │                └─ Linear(12→6): 72 params, one gate per head
#       │        └─ sigmoid: smooth bound to (0, 1)
#       └─ ×3: rescale to (0, 3) — allow amplification, not just attenuation
```

**Why per-head gating (not per-token or per-channel)?**

Each attention head specialises in a different type of relationship (syntax, semantics, long-range dependencies). Gating at the head level lets specialised heads adopt specialised strategies — a syntax head might strongly use V_static (raw token identity matters for POS tags) while a coreference head might not (context matters more). Within a head, the gate is the same across all 128 channels — expressive enough without exploding parameters.

**Why this design works — five reasons:**

| Property | How |
|----------|-----|
| Per-head decisions | Different heads can independently decide how much to trust raw token identity |
| Almost free | 72 params per layer + matmul on only 12 channels |
| Smooth and differentiable | sigmoid plays well with backpropagation |
| Bounded but expressive | (0, 3) covers off / neutral / amplify |
| Mostly a learned knob | With 12 input channels, behaves like per-layer constant with light token variation |

> **TL;DR:** The gate is a per-head, 72-parameter learnable mixing weight in the range (0, 3) that controls how strongly each head blends static value embeddings into dynamic V. With only 12 input channels it is mostly a learned per-layer constant — not fine-grained per-token reasoning. The 3× scaling trick (from modded-nanoGPT) lets the gate amplify rather than just attenuate. Think of it as: "each head learns a roughly-constant mixing weight, with a small freedom to adapt it per token."

```
Token IDs ──► wte ──► Block 0  (no VE)
                      Block 1  (VE) ◄── VE Table 1  [vocab × n_embd]
                      Block 2  (no VE)
                      Block 3  (VE) ◄── VE Table 3  [vocab × n_embd]
                      Block 4  (no VE)
                      Block 5  (VE) ◄── VE Table 5  [vocab × n_embd]
                      ...
                      Block 11 (VE) ◄── VE Table 11 [vocab × n_embd]
                      ──► ln_f ──► lm_head ──► logits
```

Each VE-enabled layer has its **own** lookup table — not shared. At layer 3, attention can look up what token "cat" contributes at that depth, which is different from what it contributes at layer 11.

#### Summary

| Aspect | Detail |
|--------|--------|
| **What it is** | Learned lookup table: token ID → static V vector, added to dynamic V |
| **Why** | Gives attention a direct path back to raw token identity at deeper layers |
| **Effect** | Improves attention diversity, reduces concentration in later layers |
| **Cost** | ~vocab_size × n_embd params per layer — hence alternating layers only |
| **Origin** | Zhou et al. 2024 (ResFormer), adopted by modded-nanoGPT speedrun community |
| **In nanochat** | Alternating layers via `has_ve(layer_idx, n_layer)`, last layer guaranteed |
| **Note** | Not in GPT-2/LLaMA tutorials — nanochat is intentionally on the modern edge |
