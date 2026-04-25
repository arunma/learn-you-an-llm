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
