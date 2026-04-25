
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

> **🔧 Actual nanochat** (`gpt.py:197-199`)
>
> nanochat does **not** have a `wpe` table. It uses **Rotary Position Embeddings (RoPE)** instead — positional information is injected inside each attention layer by rotating Q and K vectors, not by adding a learned vector to the token embeddings.
>
> ```python
> # In __init__ — precompute rotation matrices once
> cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
> self.register_buffer("cos", cos, persistent=False)
> self.register_buffer("sin", sin, persistent=False)
> ```
>
> - **No `wpe` at all** — no learned positional embedding table, no `pos_emb` addition.
> - RoPE encodes *relative* position (how far apart two tokens are), not *absolute* position (which slot a token occupies). This generalises better to unseen sequence lengths.
> - The rotation is applied to Q and K inside every attention layer, so positional information is refreshed at every block — unlike learned PE which is added once before Block 1.
> - We will cover RoPE mechanics in detail in the attention phases.

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

> **🔧 Actual nanochat** (`gpt.py:428-438`)
>
> nanochat has **no dropout anywhere** — not on embeddings, not in attention, not in the MLP. Instead, it applies RMSNorm after embedding and then a "smear gate" that mixes in the previous token's embedding:
>
> ```python
> # Embedding forward — no wpe, no dropout
> x = self.transformer.wte(idx)   # embed current token
> x = x.to(COMPUTE_DTYPE)         # ensure compute dtype (bf16)
> x = norm(x)                     # RMSNorm after embedding (not dropout!)
>
> # Smear gate — cheap bigram mixing (training only)
> gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
> x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
> ```
>
> - **No dropout** — nanochat relies on other regularisation (weight decay, small model size, short training) instead of randomly zeroing values.
> - **RMSNorm instead of dropout** — the embedding is normalised by its root-mean-square immediately after lookup.
> - **Smear gate** — a new feature not in GPT-2. For each position after the first, a small learned gate mixes in the previous token's embedding vector. This gives every token cheap access to its predecessor before attention even runs — like a lightweight bigram model baked into the embedding step.

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

> **🔧 Actual nanochat** (`gpt.py:428-464`)
>
> The real nanochat forward pass differs substantially from the simplified version above. Here is what it actually does, with the key differences annotated:
>
> ```python
> def forward(self, idx, targets=None):
>     B, T = idx.shape
>
>     # ── Embedding: no wpe, no dropout ──────────────────────
>     x = self.transformer.wte(idx)       # token embedding only
>     x = x.to(COMPUTE_DTYPE)             # master weights fp32 → compute in bf16
>     x = norm(x)                         # RMSNorm (not LayerNorm, not dropout)
>
>     # ── Smear gate: cheap bigram mixing ────────────────────
>     gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
>     x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
>
>     x0 = x                              # save initial embedding for later blending
>
>     # ── Transformer blocks with per-layer scalars ──────────
>     for i, block in enumerate(self.transformer.h):
>         # Value embeddings (ResFormer) — alternating layers get a direct token→value lookup
>         ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
>         x = block(x, ve=ve, cos=self.cos, sin=self.sin)
>         # Per-layer residual scaling + initial embedding blend-back
>         x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
>
>     x = norm(x)                         # final RMSNorm (not LayerNorm)
>
>     # ── Backout: subtract mid-layer residual ───────────────
>     if x_backout is not None:
>         x = x - self.backout_lambda.to(x.dtype) * x_backout
>
>     logits = self.lm_head(x)            # project to vocab
> ```
>
> Key differences from the simplified GPT-2 version:
>
> - **No `wpe`** — position is handled by RoPE inside attention (cos/sin buffers passed to each block).
> - **No dropout** — anywhere in the entire model.
> - **RMSNorm** replaces LayerNorm everywhere (including the final norm).
> - **Smear gate** — mixes previous token's embedding into current position before the blocks run.
> - **`x0` blend-back** — the initial embedding is saved and blended back into the residual stream at every layer via learned `x0_lambdas`. This prevents the original token identity from being washed out by deep residual accumulation.
> - **Per-layer `resid_lambdas`** — each layer scales its residual contribution independently (learned scalars, not fixed 1.0).
> - **Value embeddings** — alternating layers get a direct token-ID-to-value lookup (ResFormer), giving the model a shortcut from token identity into the value stream.
> - **Backout** — subtracts a scaled mid-layer residual before the final projection, removing information the model has learned is unhelpful for prediction.
> - **Custom `Linear`** (`gpt.py:45-50`) — master weights stay in fp32, but `forward()` casts to the activation dtype (bf16) for the matmul. No bias term.
>
> ```python
> class Linear(nn.Linear):
>     def forward(self, x):
>         return F.linear(x, self.weight.to(dtype=x.dtype))
> ```

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

> **🔧 Actual nanochat** (`gpt.py:42-43`)
>
> nanochat uses **RMSNorm** (Root Mean Square Normalisation) instead of LayerNorm. It is simpler — it only scales by the RMS of the vector, without centering (subtracting the mean) and without learnable parameters:
>
> ```python
> def norm(x):
>     return F.rms_norm(x, (x.size(-1),))
> ```
>
> - **No centering** — LayerNorm subtracts the mean then divides by std. RMSNorm skips the mean subtraction and divides by RMS = `√(mean(x²))`. Empirically works just as well for transformers, with less computation.
> - **No learnable γ, β** — standard LayerNorm has a learned scale (`γ`) and shift (`β`) per dimension. nanochat's `norm()` has zero parameters — it is a pure function.
> - **Used everywhere** — after embedding, before attention, before MLP, before final projection. Same `norm()` call throughout.

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

> **🔧 Actual nanochat** (`nanochat/optim.py`)
>
> nanochat does **not** use plain AdamW for all parameters. It uses a **Muon + AdamW hybrid** — two different optimizers for different parameter types:
>
> ```python
> # Muon for 2D matrix parameters (attention projections, MLP weights)
> #   - Uses momentum + "polar express" orthogonalisation + variance reduction
> #   - More aggressive updates for the large weight matrices
>
> # AdamW for everything else (embeddings, scalars, lm_head, 1D params)
> #   - Standard adaptive optimizer for parameters that need gentler handling
> ```
>
> - **Why two optimizers?** Large 2D weight matrices (the bulk of the model's parameters) benefit from Muon's orthogonalisation step, which keeps weight matrices well-conditioned. Embeddings and scalar parameters are better served by AdamW's per-element adaptivity.
> - **Muon** (Momentum + Unitarisation) applies Newton-Schulz iterations to orthogonalise the update direction — a more principled update for matrix-valued parameters than Adam's element-wise approach.
> - The conceptual AdamW explanation above still applies to the AdamW half of nanochat's optimizer, and the momentum/variance concepts carry over to understanding Muon.

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
