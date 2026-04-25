
---

## Phase 3 — Causal Self-Attention
*The heart of the transformer — how tokens talk to each other*

This is the most important phase in the build. Everything before it was setup. The transformer's power comes entirely from what happens here: every token looks at every other token, scores how relevant each one is, and uses those scores to pull in information from across the sequence.

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

> **🔧 Actual nanochat** (`gpt.py:75-77`)
>
> nanochat does NOT use a fused `c_attn`. It uses three separate `Linear` layers:
> ```python
> self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
> self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
> self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
> ```
> - **No fused projection.** Three separate matrices, not one big one split afterward.
> - **GQA support:** K and V project to `n_kv_head * head_dim`, not `n_head * head_dim`. When `n_kv_head < n_head`, K/V have fewer heads than Q — this is Grouped Query Attention (GQA). Multiple query heads share the same K/V head, reducing memory with minimal quality loss.
> - **Config values:** `n_head = 6`, `n_kv_head = 6`, `head_dim = n_embd // n_head = 768 // 6 = 128`. Note `head_dim` is 128, not 64 — nanochat uses a larger embedding (768) than the simplified example (384).

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

> **🔧 Actual nanochat** (`gpt.py:87-89`)
>
> nanochat skips the `.transpose()` entirely. Flash Attention 3 expects `(B, T, H, D)`, so the `.view()` output is the final layout:
> ```python
> q = self.c_q(x).view(B, T, self.n_head, self.head_dim)      # (B, T, 6, 128)
> k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, 6, 128)
> v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)   # (B, T, 6, 128)
> ```
> - **No `.transpose(1, 2)`** — Flash Attention's native layout is `(B, T, H, D)`, not `(B, H, T, D)`. One fewer operation.
> - **No `.split()`** — since there is no fused projection, each linear produces its own output directly.
> - **K and V may have fewer heads than Q** when `n_kv_head < n_head` (GQA). Flash Attention handles the head broadcasting internally.

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

> **🔧 Actual nanochat** (`gpt.py:57-63, 98-102`)
>
> nanochat does not use the `1/√d_h` scaling on the score matrix. Instead, it applies **RoPE** (Rotary Position Embeddings) and **QK Norm** directly to the Q and K vectors before attention:
>
> **RoPE** — encodes position by rotating Q and K vectors (`gpt.py:98-99`):
> ```python
> cos, sin = cos_sin
> q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
> ```
> Where `apply_rotary_emb` (`gpt.py:57-63`):
> ```python
> def apply_rotary_emb(x, cos, sin):
>     d = x.shape[3] // 2
>     x1, x2 = x[..., :d], x[..., d:]
>     y1 = x1 * cos + x2 * sin
>     y2 = x1 * (-sin) + x2 * cos
>     return torch.cat([y1, y2], 3)
> ```
> RoPE splits each head's dimensions in half, then rotates the two halves using precomputed cos/sin values that depend on sequence position. The dot product between two rotated vectors naturally encodes their *relative* distance — closer tokens produce higher scores. This replaces learned positional embeddings (`wpe`) entirely.
>
> **QK Norm** — applied after RoPE (`gpt.py:100-102`):
> ```python
> q, k = norm(q), norm(k)   # RMSNorm on Q and K
> q = q * 1.2               # sharper attention scaling
> k = k * 1.2
> ```
> - Instead of scaling scores by `1/√d_h` after the matmul, nanochat normalizes Q and K with RMSNorm *before* the matmul, then applies a fixed factor of 1.2.
> - This stabilizes training — raw dot products can explode or vanish, and normalizing the inputs prevents both.
>
> **Value Embedding (ResFormer)** — mixed into V before attention (`gpt.py:92-95`):
> ```python
> if ve is not None:
>     ve = ve.view(B, T, self.n_kv_head, self.head_dim)
>     gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
>     v = v + gate.unsqueeze(-1) * ve
> ```
> A gated residual added to V from a separate "value embedding" — not present in standard transformers. The gate is learned per-layer and controls how much extra information flows through V.

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

> **🔧 Actual nanochat** (`gpt.py:65-81`)
>
> The real `__init__` differs substantially:
> ```python
> class CausalSelfAttention(nn.Module):
>     def __init__(self, config, layer_idx):
>         super().__init__()
>         self.layer_idx = layer_idx
>         self.n_head = config.n_head
>         self.n_kv_head = config.n_kv_head
>         self.n_embd = config.n_embd
>         self.head_dim = self.n_embd // self.n_head
>         assert self.n_embd % self.n_head == 0
>         assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
>         self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
>         self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
>         self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
>         self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
>         self.ve_gate_channels = 12
>         self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
> ```
> Key differences from the simplified version:
> - **No `c_attn`** — three separate projections (`c_q`, `c_k`, `c_v`) instead of one fused layer.
> - **No causal mask buffer** — Flash Attention handles causal masking internally, so no `register_buffer('bias', ...)`.
> - **No dropout layers** — no `attn_dropout` or `resid_dropout`. Modern architectures often drop dropout entirely.
> - **`layer_idx` parameter** — each attention layer knows its position in the stack, used to decide whether to apply value embedding (`has_ve`).
> - **`n_kv_head` tracked separately** — enables GQA where K/V have fewer heads than Q.
> - **`ve_gate`** — a small linear layer (12 input channels) that gates the value embedding, present only on certain layers.

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
