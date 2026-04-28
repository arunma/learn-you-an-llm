
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

> **🔧 Actual nanochat** (`gpt.py:106`) — Flash Attention handles scaling internally. There is no explicit `* (1.0 / math.sqrt(...))` call. Instead, nanochat applies QK-norm before attention:
> ```python
> q, k = norm(q), norm(k)   # RMSNorm on Q and K
> q = q * 1.2               # learned-scale replacement
> k = k * 1.2
> ```
> - QK-norm + fixed scalar replaces the classic `1/sqrt(d_h)` scaling
> - Flash Attention's internal scaling is effectively a no-op when inputs are already normalised

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

> **🔧 Actual nanochat** (`gpt.py:106-118`) — No explicit mask matrix. Flash Attention applies causal masking internally when `causal=True`:
> ```python
> # Training path:
> y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
>
> # Inference path (with KV cache):
> k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
> y = flash_attn.flash_attn_with_kvcache(
>     q, k_cache, v_cache,
>     k=k, v=v,
>     cache_seqlens=kv_cache.cache_seqlens,
>     causal=True,
>     window_size=window_size,
> )
> ```
> - No `register_buffer('bias', ...)` — no stored mask tensor at all
> - `causal=True` tells Flash Attention to enforce the lower-triangular rule inside the fused kernel
> - `window_size=(N, 0)` adds **sliding window attention**: each token only attends to the N nearest past tokens, not the entire history
> - Memory-efficient: never materialises the full `(B, H, T, T)` attention matrix

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

> **🔧 Actual nanochat** (`gpt.py:65-126`) — The entire manual attention computation is replaced by Flash Attention. Here is the real forward pass:
> ```python
> class CausalSelfAttention(nn.Module):
>     def __init__(self, config, layer_idx):
>         super().__init__()
>         self.layer_idx = layer_idx
>         self.n_head = config.n_head
>         self.n_kv_head = config.n_kv_head
>         self.n_embd = config.n_embd
>         self.head_dim = self.n_embd // self.n_head
>         self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
>         self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
>         self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
>         self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
>         self.ve_gate_channels = 12
>         self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
>             if has_ve(layer_idx, config.n_layer) else None
>
>     def forward(self, x, ve, cos_sin, window_size, kv_cache):
>         B, T, C = x.size()
>         q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
>         k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
>         v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
>         # Value embeddings — gated additive signal on v
>         if ve is not None:
>             ve = ve.view(B, T, self.n_kv_head, self.head_dim)
>             gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
>             v = v + gate.unsqueeze(-1) * ve
>         # Rotary position embeddings (replace absolute position embeddings)
>         cos, sin = cos_sin
>         q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
>         # QK-norm + fixed scalar (replaces 1/sqrt(d_h) scaling)
>         q, k = norm(q), norm(k)
>         q = q * 1.2
>         k = k * 1.2
>         # Flash Attention — scale, mask, softmax, att@v all fused in one kernel
>         if kv_cache is None:
>             y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
>         else:
>             k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
>             y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
>                 cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size)
>             if self.layer_idx == kv_cache.n_layers - 1:
>                 kv_cache.advance(T)
>         # Reassemble — no transpose needed (FA3 outputs B, T, H, D)
>         y = y.contiguous().view(B, T, -1)
>         y = self.c_proj(y)   # no dropout wrapper
>         return y
> ```
> Key differences from the conceptual version above:
> - **No `q @ k.transpose()`, no manual mask, no explicit softmax** — Flash Attention fuses all four steps (scale, mask, softmax, att@v) into a single GPU kernel
> - **Separate Q/K/V projections** (`c_q`, `c_k`, `c_v`) instead of fused `c_attn` — enables grouped-query attention (`n_kv_head < n_head`)
> - **Rotary position embeddings** (`apply_rotary_emb`) instead of learned absolute position embeddings
> - **QK-norm** (`norm(q)`, `norm(k)`) + fixed scalar 1.2 replaces classical `1/sqrt(d_h)` scaling
> - **Value embeddings** (`ve`) — a gated additive signal on v, active on selected layers
> - **Sliding window attention** — `window_size=(N, 0)` limits each token's attention span per layer
> - **KV cache** for efficient autoregressive inference
> - **No dropout** anywhere — no `attn_dropout`, no `resid_dropout`
> - **No `.transpose(1,2)`** on reassembly — Flash Attention outputs `(B, T, H, D)` directly
>
> Sliding window sizes are computed per-layer (`gpt.py:285-312`):
> ```python
> def _compute_window_sizes(self, config):
>     pattern = config.window_pattern.upper()  # e.g. "SSSL"
>     long_window = config.sequence_len         # 2048
>     short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile (768)
>     # Tile pattern across layers, final layer always gets full context
> ```
> - Pattern like `"SSSL"` repeats across layers: 3 short-window layers, then 1 full-context layer
> - Short window = `ceil(seq_len / 4 / 128) * 128` — aligned to Flash Attention tile size
> - Final layer always sees full sequence regardless of pattern

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
| `wte` (input embed) | `(32768, 768)` | Yes — one row per token type |
| `c_q, c_k, c_v, c_proj` | `(768, 768)` | No — works on 768-dim vectors |
| MLP `c_fc, c_proj` | `(768, 3072)` etc. | No — works on 768-dim vectors |
| `lm_head` (output) | `(768, 32768)` | Yes — one column per token type |
| `value_embeds` | `(32768, 768)` | Yes — one row per token type |

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
