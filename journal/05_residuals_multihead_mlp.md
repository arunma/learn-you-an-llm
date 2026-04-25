---

## Residual Connections, MLP/FFN, and Gradient Flow
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
