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

---

### A.12 — Smear (Bigram Injection / Local Context Mixing)

*A very recent nanochat technique — not in standard tutorials. Runs once before the transformer blocks.*

---

#### What smear does

Smear mixes a small amount of the **previous token's embedding** into the current token's embedding, with a learned gate controlling how much. For every position `t` (except position 0):

```
x[t] = x[t] + gate[t] × x[t-1]
```

Each token gets a fractional "echo" of its predecessor added to it before attention runs.

---

#### Why this helps — the bigram problem

Attention is great at long-range relationships — it can attend to a token 1000 positions away just as easily as one position away. But this strength is also a weakness for **bigram-level information**: simple "what's the previous token?" patterns.

To learn that "the" often precedes "cat", an attention head would have to:
- Spend capacity learning to attend specifically to position `t-1`
- Burn one of its limited 6 heads on this trivial task
- Pay the quadratic attention cost just to look one step back

That's expensive machinery for the simplest possible context.

**Smear handles this by physically mixing each token with its predecessor before attention even runs.** After smear, attention doesn't need to reach back to position `t-1` — that information is already baked into position `t`. Attention is freed up to focus on longer-range relationships.

```
Without smear:
  Attention head must learn: "always look at t-1 for bigram context"
  Cost: one full attention head dedicated to a trivial task

With smear:
  x[t] already contains a fraction of x[t-1] before attention starts
  Attention can specialise on longer-range, harder patterns
```

The name comes from the metaphor of a paintbrush: each token's colour "smears" slightly forward into its neighbour.

---

#### The gate — 25 parameters total

```python
self.smear_gate   = Linear(24, 1, bias=False)   # [NC] 24 params — per-token scalar
self.smear_lambda = nn.Parameter(torch.zeros(1)) # [NC]  1 param  — global strength
```

**`smear_gate`** — a tiny `Linear(24, 1)`. Takes the first 24 channels of each token's embedding and produces one scalar per token. Output shape: `(B, T-1, 1)`.

**`smear_lambda`** — a single global scalar. Controls the overall strength of smearing across the whole model.

```python
gate = smear_lambda * torch.sigmoid(smear_gate(x[:, 1:, :24]))
# gate shape: (B, T-1, 1)
# = global strength × per-token modulation in (0, 1)
```

The gate applies uniformly across all 768 channels of that token's embedding via broadcasting:

```
gate:          (B, T-1, 1)     ← per-token scalar
x[:, :-1]:     (B, T-1, 768)   ← previous tokens' embeddings
gate × x[:,:-1]: (B, T-1, 768) ← broadcasting expands 1 → 768
```

**`smear_lambda` initialised to zero** — at training start, smear contributes exactly nothing. The model must learn during training that smearing is useful and grow `smear_lambda` to a useful value. If smear doesn't help, the model simply leaves `smear_lambda` near zero and the operation becomes a no-op. The model opts in to smearing rather than being forced to use it.

**Why only 24 channels for the gate input?** Same reasoning as the value embedding gate — the gate behaves mostly like a learned per-token constant with mild input-dependent variation. 24 channels gives just enough flexibility at essentially zero parameter cost.

---

#### Why position 0 is skipped

```python
x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
#              ↑ position 0       ↑ positions 1 through T-1
#              untouched          each gets fraction of predecessor
```

Position 0 has no previous token — `x[-1]` doesn't exist. So the gate is computed only for positions 1 through T-1, and position 0 is passed through untouched and re-glued at the front. The shapes align naturally:

```
x[:, 1:]    (B, T-1, 768)   ← positions to smear INTO  (1 through T-1)
x[:, :-1]   (B, T-1, 768)   ← positions to smear FROM  (0 through T-2)
gate        (B, T-1, 1)     ← one gate per smear operation
```

**Concrete example with T=4:**

```
Before smear:
  pos 0: [x_0]    pos 1: [x_1]    pos 2: [x_2]    pos 3: [x_3]

After smear:
  pos 0: [x_0]              ← untouched (no predecessor)
  pos 1: [x_1 + g1 × x_0]  ← gets fraction of x_0
  pos 2: [x_2 + g2 × x_1]  ← gets fraction of x_1
  pos 3: [x_3 + g3 × x_2]  ← gets fraction of x_2
```

---

#### The three code branches

Smear is implemented three ways depending on the inference scenario:

**Branch 1 — Training or full sequence (`kv_cache is None`):**

```python
gate = smear_lambda * torch.sigmoid(smear_gate(x[:, 1:, :24]))   # [NC]
x    = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1) # [NC]
# Full sequence in memory — vectorise smear across all T positions at once
```

**Branch 2 — Prefill with KV cache (`T > 1`):**

```python
x_pre_smear = kv_cache.prev_embedding         # [NC] retrieve saved embedding
kv_cache.prev_embedding = x[:, -1:, :]        # [NC] save last token for next pass
# Same smear logic as Branch 1, plus stash the last embedding
```

**Branch 3 — Single token decode (`T == 1`):**

```python
gate = smear_lambda * torch.sigmoid(smear_gate(x[:, :, :24]))   # [NC]
x    = x + gate * x_pre_smear                                    # [NC]
# No "previous token in this pass" — smear from the cached embedding instead
```

The KV cache carries the previous token's embedding across forward passes so generation feels continuous:

```
Forward pass 1: tokens [t0, t1, t2, t3, t4]
  → smear normally (Branch 2)
  → save x[4] in kv_cache.prev_embedding

Forward pass 2: token [t5]
  → smear: x[5] += gate × cached x[4]   (Branch 3)
  → save x[5] in kv_cache.prev_embedding

Forward pass 3: token [t6]
  → smear: x[6] += gate × cached x[5]
```

---

#### Where smear sits in the model

```python
def forward(self, idx, ...):
    x = self.transformer.wte(idx)    # [PT] embed tokens
    x = norm(x)                       # [PT] normalise

    # ── Smear runs HERE — once, before any blocks ──────────────
    if kv_cache is None:
        gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
        x    = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

    # ── Then through 12 transformer blocks ─────────────────────
    for block in self.transformer.h:
        x = block(x, ...)
```

Smear runs **once per model**, not once per layer. It prepares the embeddings with bigram context before the transformer blocks begin. Running it inside every block would be redundant — attention already mixes token information at each layer.

---

#### Summary

| Aspect | Detail |
|--------|--------|
| **What it does** | Mixes fraction of previous token's embedding into current token |
| **Why** | Injects bigram context cheaply so attention can focus on longer-range patterns |
| **Parameters** | 25 total — `smear_gate` (24) + `smear_lambda` (1) |
| **Gate range** | `smear_lambda × sigmoid(...)` — global scalar × per-token (0,1) |
| **Initialisation** | `smear_lambda = 0` — starts as no-op, model opts in during training |
| **Position 0** | Always skipped — no predecessor to smear from |
| **Runs** | Once per model, before all transformer blocks |
| **KV cache** | Carries previous token's embedding across forward passes for continuous generation |
| **Related to** | Value embedding gate (same 24-channel input slice philosophy) |

---

### A.13 — Resid Lambdas, x0 Lambdas, and Backout

*Three more speedrun-era techniques from the modded-nanoGPT community. Each costs a handful of scalars and provides 1–3% faster convergence.*

---

#### Resid lambdas and x0 lambdas — input re-injection

In a standard transformer, data flows straight through each block:

```python
for block in blocks:
    x = block(x)    # x0 (the original embedding) is never seen again after block 0
```

By layer 8, the original token embedding is buried under 8 rounds of transformation. nanochat fixes this by **rebalancing the residual stream** before each block:

```python
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0   # [NC]
```

For each layer `i`:
- `resid_lambdas[i]` — how much of the **current state** to keep
- `x0_lambdas[i]` — how much of the **original embedding** to re-inject
- Both are learnable scalars, one per layer

**Initialisation — decaying injection over depth:**

```python
for i in range(n_layer):
    resid_lambdas[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))   # [NC]
    x0_lambdas[i]    = 0.20 - (0.15 * i / max(n_layer - 1, 1))   # [NC]
```

For `n_layer=12`:

| Layer | resid_lambda | x0_lambda | Meaning |
|-------|-------------|-----------|---------|
| 0 | 1.15 | 0.20 | Heavy re-injection of original embedding |
| 3 | 1.12 | 0.15 | Moderate re-injection |
| 6 | 1.10 | 0.13 | Half strength |
| 11 | 1.05 | 0.05 | Minimal re-injection |

The pattern: **early layers get strongly re-injected with the original embedding, later layers less so.**

```
Standard transformer:
  x0 → block 0 → block 1 → block 2 → ... → block 11
       (x0 never seen again after entering block 0)

With resid + x0 lambdas:
  x0 ──────────────────────────────────────────────┐
       ↓                ↓                ↓         ↓
  block 0 input:  block 1 input:  ...  block 11 input:
  1.15×x0+0.20×x0  1.14×x1+0.19×x0      1.05×x10+0.05×x0
  ↑ current state  ↑ original           ↑ mostly current state
  (heavy early)    (medium)              (tiny late)
```

**Why this design:**
- **Early layers** need raw token information to build basic features — heavy x0 re-injection helps
- **Late layers** need abstract combinations — less raw input, more built-up state
- `resid_lambda ≈ 1.1` gives mild amplification of the residual stream, helping signal flow in deep networks
- Both are learned — training can tune them away from their initialisations if needed

---

#### Backout — removing mid-layer features before prediction

After all 12 blocks, nanochat **subtracts a fraction of the mid-layer state** from the final state before `lm_head`:

```python
backout_layer = n_layer // 2   # layer 6 in a 12-layer model                 # [NC]

for i, block in enumerate(self.transformer.h):
    x = block(x, ...)
    if i == backout_layer:
        x_backout = x              # [NC] cache the mid-layer state

# After all blocks:
x = x - self.backout_lambda * x_backout   # [NC] subtract mid-layer features
x = norm(x)
logits = self.lm_head(x)
```

`backout_lambda` is a learnable scalar initialised to **0.2**.

**The intuition:**

Transformer layers build up features in roughly this hierarchy:

```
Early layers (0–3):   low-level — token identity, simple syntax, local patterns
Mid layers  (4–7):    intermediate — phrase structure, n-gram patterns
Late layers (8–11):   high-level — semantic meaning, long-range dependencies
```

For next-token prediction, you want the **high-level abstractions** — not the surface features that were useful mid-network but have already served their purpose. Subtracting a fraction of the mid-layer state effectively filters out lower-level signals from the final prediction.

Think of it as a bandpass filter: attenuate the low-frequency (shallow) signal, let through the high-frequency (deep) signal.

**The digit recognition analogy:**
```
Early layers: "this has curves and lines"
Mid layers:   "this has a closed loop on top"
Late layers:  "this is the digit 9"

For prediction you want "digit 9", not "curves and lines."
Backout subtracts a bit of the mid-level state to clean up the final representation.
```

**Initialisation at 0.2 (not 0):** Backout is "opt-out" — assumed to help unless training suggests otherwise. If backout hurts, `backout_lambda` shrinks toward zero during training. This contrasts with smear (`smear_lambda=0`, opt-in) which assumes no effect until proven useful.

---

#### The complete forward pass — all modern techniques assembled

```python
def forward(self, idx, ...):                                           # [NC]
    # 1. Embed and normalise
    x  = self.transformer.wte(idx)                                     # [PT]
    x  = norm(x)                                                       # [PT]

    # 2. Smear — mix previous token's embedding (A.12)
    x  = apply_smear(x)                                                # [NC]

    # 3. Save x0 for re-injection (A.13)
    x0 = x                                                             # [NC]

    # 4. 12 transformer blocks with resid/x0 mixing and backout cache
    x_backout    = None                                                # [NC]
    backout_layer = n_layer // 2                                       # [NC]

    for i, block in enumerate(self.transformer.h):                    # [NC]
        # Re-inject original embedding (A.13)
        x  = resid_lambdas[i] * x + x0_lambdas[i] * x0              # [NC]
        # Get value embedding for this layer if applicable (A.11)
        ve = value_embeds[i](token_ids) if has_ve(i) else None        # [NC]
        # Run the block (Phase 4)
        x  = block(x, ve, ...)                                        # [NC]
        # Cache mid-layer state for backout (A.13)
        if i == backout_layer:                                         # [NC]
            x_backout = x

    # 5. Backout — subtract mid-layer features (A.13)
    if x_backout is not None:                                          # [NC]
        x = x - backout_lambda * x_backout

    # 6. Final norm and lm_head
    x      = norm(x)                                                   # [PT]
    logits = self.lm_head(x)                                           # [PT]
    return logits
```

---

#### Cost and benefit summary

| Technique | Parameters | Initialisation | Mechanism |
|-----------|-----------|---------------|-----------|
| `resid_lambdas` | n_layer scalars (12) | 1.05–1.15 (opt-out) | Scale current state before each block |
| `x0_lambdas` | n_layer scalars (12) | 0.05–0.20 (opt-out) | Re-inject original embedding, decaying with depth |
| `backout_lambda` | 1 scalar | 0.2 (opt-out) | Subtract mid-layer state from final state |
| smear (A.12) | 25 params | 0.0 (opt-in) | Mix previous token into current before blocks |
| value embeds (A.11) | ~150M params | random | Static V lookup by token ID at alternate layers |

**The pattern across all of these:** give the model small learnable knobs to control information flow, with sensible initial values, and let training tune them. Each is tiny individually (a handful of scalars) but they stack — combined with QK norm, ReLU², and RoPE these tricks help nanochat train significantly faster than a vanilla GPT-2.

**Where these come from:** all discovered empirically by the modded-nanoGPT speedrun community racing to train GPT-2 to a target loss in the fewest H100-hours. Karpathy packaged the best of them into nanochat as "modern minimalism."

---

### A.14 — The Output Stage: lm_head to Loss (Every Step Explained)

*The final sequence from hidden states to either a training loss or inference-ready logits. Several small but critical stability tricks live here.*

---

#### The full sequence

```python
softcap = 15

# 1. Project hidden state to vocabulary scores
logits = self.lm_head(x)                           # [PT] (B, T, padded_V)

# 2. Crop padding from vocabulary
logits = logits[..., :self.config.vocab_size]      # [NC] (B, T, V)

# 3. Cast to fp32 for numerical stability
logits = logits.float()                            # [PT] bf16 → fp32

# 4. Softcap — smoothly bound to (-15, +15)
logits = softcap * torch.tanh(logits / softcap)    # [NC]

# 5. Branch on training vs inference
if targets is not None:
    loss = F.cross_entropy(                        # [PT]
        logits.view(-1, logits.size(-1)),          # [NC] (B*T, V)
        targets.view(-1),                          # [NC] (B*T,)
        ignore_index=-1,
        reduction='mean'
    )
    return loss
else:
    return logits   # caller does softmax + sampling
```

---

#### Step 1 — lm_head projection

`lm_head` is `Linear(768, padded_vocab_size)`. Output is `(B, T, padded_V)` — one raw score per vocabulary token at each position. The output uses `padded_V` not `V` because the vocabulary is padded to a multiple of 64 for hardware efficiency.

---

#### Step 2 — Crop vocabulary padding

```python
padded_vocab_size = ceil(vocab_size / 64) * 64
# e.g. vocab_size=32750 → padded to 32768
```

Modern GPU tensor cores run much faster on matrix dimensions that are multiples of 8/16/32/64. A `(768, 32768)` matmul is significantly faster than `(768, 32750)`. The extra slots are fake tokens that don't exist in the real vocabulary. After the matmul, crop them back off so sampling never picks a non-existent token:

```python
logits = logits[..., :self.config.vocab_size]      # [NC] remove padding slots
```

---

#### Step 3 — Cast to fp32

```python
logits = logits.float()    # [PT] bf16 → fp32
```

The entire model runs in bf16 for speed. But the final logit/loss computation needs fp32 precision for two reasons:

**Cross-entropy is numerically sensitive.** It involves `log(softmax(logits))` which requires `exp(logits)`. bf16 has only ~3 decimal digits of precision — large logit values produce overflows or precision loss. fp32 has ~7 decimal digits.

**Logit differences must be preserved.** Softmax depends on differences between logit values. Two logits of `5.301` and `5.299` might both round to `5.3` in bf16, losing the distinction entirely. fp32 preserves it.

Cost is small: logits are computed once per forward pass and immediately consumed (training) or returned (inference). The temporary fp32 tensor is short-lived.

---

#### Step 4 — Softcap (the most interesting step)

```python
logits = 15 * torch.tanh(logits / 15)             # [NC]
```

This smoothly bounds all logits to the range `(-15, +15)` using the tanh function.

**How it works:**

```
tanh maps any real number to (-1, +1):
  tanh(-∞) = -1,  tanh(0) = 0,  tanh(+∞) = +1
  For small x: tanh(x) ≈ x  (linear — passes through unchanged)
  For large x: tanh(x) saturates near ±1

Scaled by 15:
  logit =  +5  →  15 × tanh(5/15)   =  15 × tanh(0.33) ≈  4.9  (barely changed)
  logit = +10  →  15 × tanh(10/15)  =  15 × tanh(0.67) ≈  9.6  (slightly squashed)
  logit = +20  →  15 × tanh(20/15)  =  15 × tanh(1.33) ≈ 13.3  (noticeably squashed)
  logit = +30  →  15 × tanh(30/15)  =  15 × tanh(2.00) ≈ 14.4  (heavily squashed)
```

Normal logits (±5 to ±10) pass through nearly unchanged. Extreme outliers (>20) get gently squeezed toward ±15.

**Why softcap at all:**
- **Training stability** — logits can explode during training. Unbounded logits cause near-infinite gradients through cross-entropy.
- **Better calibration** — models with huge logit gaps are poorly calibrated (very confident but often wrong). Softcap forces more meaningful uncertainty.
- **Numerical safety** — very large values cause overflows in `exp()` inside softmax, even in fp32.

**Why tanh (soft) instead of clamp (hard):**

```python
# Hard clip — gradient is exactly 0 outside ±15:
logits = torch.clamp(logits, -15, 15)
# Once a logit hits +15, gradient = 0, it can never be updated again.

# Soft cap — gradient is small but non-zero even at saturation:
logits = 15 * torch.tanh(logits / 15)
# Logit of +30 can still be nudged — just with a smaller gradient.
```

Soft capping preserves gradients at the boundary. Hard clipping kills them. Same principle as sigmoid vs step functions — smooth, differentiable bounding lets gradient descent keep working.

The value `15` is empirical, popularised by Gemma 2. Large enough that normal logits (±5–10) pass through unchanged, small enough to prevent extreme outliers from destabilising training.

---

#### Step 5a — Training path: cross-entropy loss

```python
loss = F.cross_entropy(                            # [PT]
    logits.view(-1, logits.size(-1)),              # [NC] (B*T, V)
    targets.view(-1),                              # [NC] (B*T,)
    ignore_index=-1,
    reduction='mean'
)
```

**Reshape for cross_entropy:**

`F.cross_entropy` expects `(N, V)` predictions and `(N,)` targets. Our tensors are `(B, T, V)` and `(B, T)`, so `.view(-1, V)` and `.view(-1)` flatten batch and time together: `B*T = 12*1024 = 12,288` independent predictions per forward pass.

**`ignore_index=-1`:**

Any target token with value `-1` is skipped — gradient is not computed for that position. Used in instruction tuning to mask the prompt so the model is only graded on its responses, not on reproducing the input. In plain pretraining, `-1` targets don't appear.

**What cross-entropy computes:**

```
For each of the 12,288 positions:
  loss_t = -log( exp(logits[target]) / Σ_v exp(logits[v]) )
         = -logits[target] + log(Σ_v exp(logits[v]))

Final loss = mean of all 12,288 values → one scalar
```

High probability on the correct token → low loss. Low probability → high loss.

`F.cross_entropy` is **fused** — it computes softmax and log together internally in a numerically stable way. Never apply softmax manually before passing to `cross_entropy`. The fused version avoids precision loss that separate softmax+log would introduce.

---

#### Step 5b — Inference path: return logits

```python
return logits   # (B, T, V) fp32, bounded to ±15
```

No targets → return the bounded fp32 logits. The caller handles sampling:

```python
# In generate():
logits_last = logits[:, -1, :] / temperature      # [NC] last position + temperature
probs = F.softmax(logits_last, dim=-1)             # [PT] convert to probabilities
next_token = torch.multinomial(probs, 1)           # [PT] sample
```

The model hands back scores. The caller decides the sampling strategy (greedy, top-k, temperature, etc.).

---

#### The complete output stage flow

```
Hidden states: (B, T, 768)  bf16
       ↓  lm_head: Linear(768, padded_V)
(B, T, padded_V)  bf16
       ↓  crop [..., :vocab_size]
(B, T, V)  bf16
       ↓  .float()
(B, T, V)  fp32
       ↓  15 × tanh(logits / 15)
(B, T, V)  fp32  bounded to (-15, +15)
       ↓
  ┌────┴─────┐
targets?   no targets
  ↓           ↓
cross_entropy  return logits
scalar loss    caller samples
```

---

#### Why every step matters

| Step | What breaks without it |
|------|----------------------|
| Vocab crop | Sampling picks non-existent padding tokens |
| Cast to fp32 | Numerical overflow/imprecision in cross-entropy log/exp |
| Softcap | Logit explosion → training instability → divergence |
| Soft vs hard cap | Hard clipping kills gradients at boundary → saturated logits frozen |
| Fused cross_entropy | Manual softmax+log loses precision vs fused operation |

These are "polish" not core architecture — but they're what separates a training run that converges smoothly from one that diverges or produces garbage.

> **The key engineering insight:** logits are kept as raw scores (never converted to probabilities) until the very end. cross_entropy's fused softmax+log is more stable than manual conversion. fp32 casting and softcap together handle extreme inputs. Vocab cropping prevents fake token sampling. Small details, large practical impact.

---

### A.15 — The Naive generate() — Autoregressive Sampling Explained

*The simplest possible generation loop. No KV cache, no tricks — just the fundamental "predict next token, append, repeat" loop that all LLM inference builds on.*

---

#### The big picture

This is the reference implementation of generation — intentionally simple, easy to read and verify. It is a Python **generator** (uses `yield`) that streams tokens one at a time as they are produced rather than waiting for the full output.

```python
@torch.inference_mode()                                          # [PT]
def generate(self, tokens, max_tokens, temperature=1.0,
             top_k=None, seed=0):
```

**`@torch.inference_mode()`** — more aggressive than `@torch.no_grad()`. Disables gradient tracking, version counters, and view-tracking machinery. Tensors created inside cannot have gradients enabled later. 10–30% faster than normal mode. Always use this for pure inference.

---

#### Setup

```python
ids = torch.tensor([tokens], dtype=torch.long, device=device)  # [PT] (1, T)
# [tokens] adds a batch dimension → shape (1, T)
# dtype=torch.long = int64, required for token IDs
```

```python
rng = None
if temperature > 0:
    rng = torch.Generator(device=device)   # [PT] dedicated RNG — doesn't affect global seed
    rng.manual_seed(seed)                  # [PT] same seed + same prompt = same output
```

A dedicated `torch.Generator` is used rather than the global PyTorch seed so this function's randomness doesn't interfere with anything else in the program. If `temperature=0` (greedy), no randomness is needed so `rng` stays None.

---

#### The generation loop

```python
for _ in range(max_tokens):                                      # [NC]
```

Each iteration produces exactly one token. Five steps:

**Step 1 — Full forward pass:**

```python
logits = self.forward(ids)       # [NC] (1, T, V) — runs the entire model
```

The entire current sequence is fed through all 12 blocks every iteration. This is the inefficiency — at iteration 100, it recomputes K and V for the first 99 tokens that haven't changed. The KV cache (see Engine.generate) fixes this by caching those computations.

**Step 2 — Extract last position only:**

```python
logits = logits[:, -1, :]        # [NC] (1, T, V) → (1, V) — throw away all but last
```

Only the last position predicts "what comes next." All earlier positions' logits are discarded.

**Step 3 — Top-k filtering (optional):**

```python
if top_k is not None and top_k > 0:                              # [NC]
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))      # [PT] find k largest values
    logits[logits < v[:, [-1]]] = -float('Inf')                  # [NC] zero out the rest
    # v[:, [-1]] = the kth value (threshold)
    # anything below threshold → -inf → exp(-inf)=0 → zero probability after softmax
```

Restricts sampling to only the k most likely tokens. All others get `-inf` so they get exactly zero probability after softmax. Prevents the model from occasionally picking absurdly unlikely tokens.

`v[:, [-1]]` (with brackets) keeps the singleton dimension `(1,1)` for clean broadcasting against `(1,V)`.

**Step 4 — Sample the next token:**

```python
if temperature > 0:                                              # [NC]
    logits = logits / temperature                                # [NC]
    probs  = F.softmax(logits, dim=-1)                          # [PT]
    next_ids = torch.multinomial(probs, num_samples=1,          # [PT]
                                 generator=rng)
else:
    next_ids = torch.argmax(logits, dim=-1, keepdim=True)       # [PT] greedy
```

Temperature effect on the same logits `[5, 4, 3]`:

```
T=0.5:  softmax([10, 8, 6])    = [0.84, 0.11, 0.04]  ← spiky, conservative
T=1.0:  softmax([5, 4, 3])     = [0.66, 0.24, 0.10]  ← model's natural distribution
T=2.0:  softmax([2.5, 2, 1.5]) = [0.49, 0.30, 0.21]  ← flat, creative/random
T=0:    argmax → always pick 5 (index 0)               ← fully deterministic
```

`keepdim=True` on argmax keeps shape `(1,1)` matching the multinomial path.

**Step 5 — Append and yield:**

```python
ids = torch.cat((ids, next_ids), dim=1)   # [PT] (1, T) + (1, 1) → (1, T+1)
token = next_ids.item()                    # [PT] tensor → Python int
yield token                                # [NC] stream to caller
```

`yield` makes this a Python generator — the function pauses here and hands the token to the caller. The caller can print tokens as they arrive rather than waiting for the full generation:

```python
for token in model.generate(prompt, max_tokens=100):
    print(tokenizer.decode([token]), end='', flush=True)   # [NC] live streaming
```

---

#### Concrete walkthrough — generating 3 tokens from prompt [1, 2, 3]

```
Iteration 0:
  ids = [[1, 2, 3]]              (1, 3)
  forward → logits               (1, 3, V)
  last position logits           (1, V)
  sample → next_ids = [[7]]      (1, 1)
  yield 7
  ids = [[1, 2, 3, 7]]          (1, 4)

Iteration 1:
  ids = [[1, 2, 3, 7]]          (1, 4)
  forward → logits               (1, 4, V)  ← recomputes positions 0-3 again!
  last position logits           (1, V)
  sample → next_ids = [[42]]     (1, 1)
  yield 42
  ids = [[1, 2, 3, 7, 42]]      (1, 5)

Iteration 2:
  ids = [[1, 2, 3, 7, 42]]      (1, 5)
  forward → logits               (1, 5, V)  ← recomputes positions 0-4 again!
  sample → next_ids = [[15]]     (1, 1)
  yield 15
```

---

#### The performance problem and the KV cache fix

```
Naive generate() — work per iteration grows with sequence length:
  Iteration 0: process T tokens
  Iteration 1: process T+1 tokens   ← redoes all T
  Iteration 2: process T+2 tokens   ← redoes all T+1
  ...
  Total work: O(N²) for N generated tokens

Engine.generate() with KV cache — work per iteration is constant:
  Iteration 0: process T tokens     (initial prefill, cache K/V for all T)
  Iteration 1: process 1 token      (cache handles positions 0..T)
  Iteration 2: process 1 token      (cache handles positions 0..T+1)
  ...
  Total work: O(N) for N generated tokens
```

The naive version exists as a **correctness reference** — simple enough to read, verify, and debug against. Engine.generate does the same thing efficiently.

---

#### Sampling strategies

| Configuration | Behaviour |
|---------------|-----------|
| `temperature=0` | Greedy — always pick highest logit, fully deterministic |
| `temperature=1, top_k=None` | Pure sampling — most diverse, no filtering |
| `temperature=1, top_k=50` | Top-k — limit to 50 most likely tokens |
| `temperature=0.7, top_k=50` | Common production setting — balanced |
| `temperature=2.0` | Wild — very random, often nonsensical |

Practical guidance: `temperature=0.7–1.0` with `top_k=50` for creative tasks. `temperature=0.0–0.3` for factual/code generation.

---

#### TL;DR

Generation is "predict next token, append, repeat." Every optimisation in LLM inference (KV caches, speculative decoding, batched inference) is built on top of this fundamental loop. Understanding this naive version teaches you the essence of how all LLM inference works.

The naive version is `O(N²)` — each iteration recomputes K and V for the entire growing sequence. The KV cache makes it `O(N)` by caching those computations. The maths is identical; only the order of computation changes.
