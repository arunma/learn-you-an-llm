"""Generate an SVG visualization of tensor shape transformations through the nanochat pipeline."""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------- data ----------

STEPS = [
    # (step, operation, shape_tuple, label)
    (0,  "get_batch()",      (1, 1),                "(B, T)"),
    (1,  "wte(idx)",         (1, 1, 768),            "(B, T, 768)"),
    (2,  "norm(x)",          (1, 1, 768),            "(B, T, 768)"),
    (3,  "smear",            (1, 1, 768),            "(B, T, 768)"),
    (4,  "c_q/c_k/c_v",     (1, 1, 768),            "(B, T, 768)"),
    (5,  ".view(B,T,6,128)", (1, 1, 6, 128),         "(B, T, 6, 128)"),
    (6,  "RoPE",             (1, 1, 6, 128),         "(B, T, 6, 128)"),
    (7,  "QK norm",          (1, 1, 6, 128),         "(B, T, 6, 128)"),
    (8,  "Flash Attn",       (1, 1, 6, 128),         "(B, T, 6, 128)"),
    (9,  ".view(B,T,-1)",    (1, 1, 768),            "(B, T, 768)"),
    (10, "c_proj",           (1, 1, 768),            "(B, T, 768)"),
    (11, "+residual",        (1, 1, 768),            "(B, T, 768)"),
    (12, "MLP expand",       (1, 1, 3072),           "(B, T, 3072)"),
    (13, "relu\u00b2",       (1, 1, 3072),           "(B, T, 3072)"),
    (14, "MLP compress",     (1, 1, 768),            "(B, T, 768)"),
    (15, "+residual",        (1, 1, 768),            "(B, T, 768)"),
    (16, "\u00d712 blocks",  (1, 1, 768),            "(B, T, 768)"),
    (17, "backout+norm",     (1, 1, 768),            "(B, T, 768)"),
    (18, "lm_head",          (1, 1, 32768),          "(B, T, 32768)"),
    (19, "softcap",          (1, 1, 32768),          "(B, T, 32768)"),
    (20, "cross_entropy",    (1,),                   "scalar"),
]

# Use log-scale for the total element count so the tiny (B,T) and scalar bars
# are still visible next to the 32768-wide vocab bars.
def bar_height(shape):
    total = math.prod(shape)
    if total <= 1:
        return 0.3          # give scalar a visible sliver
    return math.log2(total)  # log2 keeps proportions readable

MAX_H = bar_height((1, 1, 32768))  # normalizing reference

# Color palette per semantic dimension
C_BATCH  = "#4A90D9"   # blue
C_SEQ    = "#50B86C"   # green
C_EMBED  = "#E8943A"   # orange
C_VOCAB  = "#D94A4A"   # red
C_HEAD   = "#9B59B6"   # purple  (heads dimension)
C_SCALAR = "#888888"   # grey

def color_for_step(step_idx, shape):
    """Return the dominant color based on which new dimension drives the shape."""
    if step_idx == 0:
        return C_SEQ        # just B, T
    if step_idx == 20:
        return C_SCALAR
    if step_idx in (18, 19):
        return C_VOCAB
    if step_idx in (5, 6, 7, 8):
        return C_HEAD
    if step_idx in (12, 13):
        return C_EMBED      # 4x expansion still embedding-flavored
    return C_EMBED

# Highlight steps where a NEW dimension appears
HIGHLIGHT_STEPS = {
    1:  "C appears",
    5:  "heads appear",
    18: "V appears",
}

# ---------- plot ----------

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

bar_width = 0.7
xs = np.arange(len(STEPS))

for i, (step, op, shape, label) in enumerate(STEPS):
    h = bar_height(shape) / MAX_H  # normalize to [0, 1] range
    c = color_for_step(step, shape)

    # Highlighted steps get a gold border
    is_highlight = step in HIGHLIGHT_STEPS
    edge_color = "#FFD700" if is_highlight else "#333333"
    edge_width = 2.5 if is_highlight else 0.8

    bar = ax.bar(
        i, h, width=bar_width,
        color=c, edgecolor=edge_color, linewidth=edge_width,
        alpha=0.88, zorder=3,
    )

    # Shape annotation above bar
    ax.text(
        i, h + 0.03, label,
        ha="center", va="bottom", fontsize=5.5,
        fontfamily="monospace", color="#222222", rotation=45,
    )

    # Highlight callout
    if is_highlight:
        ax.annotate(
            HIGHLIGHT_STEPS[step],
            xy=(i, h + 0.02),
            xytext=(i, h + 0.30),
            ha="center", fontsize=7, fontweight="bold",
            color="#B8860B",
            arrowprops=dict(arrowstyle="->", color="#B8860B", lw=1.2),
            zorder=5,
        )

# X-axis: operation names
ax.set_xticks(xs)
ax.set_xticklabels(
    [f"{s[0]}. {s[1]}" for s in STEPS],
    rotation=55, ha="right", fontsize=6.5,
)

ax.set_ylabel("Relative tensor size (log scale)", fontsize=9)
ax.set_title(
    "Nanochat: Tensor Shape Flow Through the Forward Pass",
    fontsize=13, fontweight="bold", pad=14,
)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yticks([])

# Legend
legend_items = [
    mpatches.Patch(facecolor=C_BATCH,  edgecolor="#333", label="Batch (B)"),
    mpatches.Patch(facecolor=C_SEQ,    edgecolor="#333", label="Sequence (T)"),
    mpatches.Patch(facecolor=C_EMBED,  edgecolor="#333", label="Embedding (C)"),
    mpatches.Patch(facecolor=C_HEAD,   edgecolor="#333", label="Heads (n_head)"),
    mpatches.Patch(facecolor=C_VOCAB,  edgecolor="#333", label="Vocab (V)"),
    mpatches.Patch(facecolor=C_SCALAR, edgecolor="#333", label="Scalar"),
    mpatches.Patch(facecolor="white",  edgecolor="#FFD700", linewidth=2,
                   label="Dimension change"),
]
ax.legend(
    handles=legend_items, loc="upper left", fontsize=7,
    framealpha=0.9, edgecolor="#CCCCCC",
)

# Connecting arrows between bars to show flow
for i in range(len(STEPS) - 1):
    h_cur = bar_height(STEPS[i][2]) / MAX_H
    h_nxt = bar_height(STEPS[i + 1][2]) / MAX_H
    ax.annotate(
        "",
        xy=(i + 1 - bar_width / 2, h_nxt / 2),
        xytext=(i + bar_width / 2, h_cur / 2),
        arrowprops=dict(
            arrowstyle="->,head_width=0.15,head_length=0.1",
            color="#AAAAAA", lw=0.6,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )

plt.tight_layout()

out_path = "/Users/argon/projects/learn_you_a_llm/journal/diagrams/tensor_shape_flow.svg"
fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")
