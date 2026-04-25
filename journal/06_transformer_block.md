
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

**The output is logits, not probabilities:**

```python
logits = self.lm_head(x)    # [PT] (B, T, 384) → (B, T, 50257)
# 50257 raw scores per token position — one per vocabulary token
# Can be any value — positive or negative
# NOT probabilities — they do not sum to 1.0
# Become probabilities only after softmax
# Highest logit = most likely next token (before softmax)

# During inference — convert to probabilities and sample:
probs = F.softmax(logits[:, -1, :], dim=-1)   # [PT] last position only
next_token = torch.multinomial(probs, 1)       # [PT] sample
```

---

### 4.4 — Each token position is a complete training example

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
