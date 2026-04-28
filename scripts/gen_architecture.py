"""Generate an SVG diagram of the nanochat transformer architecture pipeline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Colors ──────────────────────────────────────────────────────────────────
BLUE   = "#2196F3"  # embedding / projection
GREEN  = "#4CAF50"  # attention
ORANGE = "#FF9800"  # MLP
PURPLE = "#9C27B0"  # normalization
RED    = "#F44336"  # loss / output
GRAY   = "#607D8B"  # modern additions
BG     = "#FAFAFA"
ARROW  = "#37474F"
TEXT   = "#212121"
SHAPE  = "#9E9E9E"

# ── Layout constants ────────────────────────────────────────────────────────
FIG_W, FIG_H = 8, 14
BOX_W = 4.8
BOX_H = 0.42
CENTER_X = FIG_W / 2
MARGIN_TOP = 13.3
STEP = 0.58        # vertical spacing between boxes
SMALL_STEP = 0.48  # tighter spacing inside the block

# ── Helpers ─────────────────────────────────────────────────────────────────

def draw_box(ax, y, label, color, shape_text=None, width=BOX_W, height=BOX_H,
             fontsize=9, alpha=0.18, bold=False, cx=None):
    """Draw a rounded rectangle with centered label and optional shape annotation."""
    cx = cx or CENTER_X
    x = cx - width / 2
    # Draw fill with low alpha, then edge at full alpha
    fill = patches.FancyBboxPatch(
        (x, y - height / 2), width, height,
        boxstyle="round,pad=0.08", linewidth=0,
        edgecolor="none", facecolor=color, alpha=alpha,
    )
    border = patches.FancyBboxPatch(
        (x, y - height / 2), width, height,
        boxstyle="round,pad=0.08", linewidth=1.6,
        edgecolor=color, facecolor="none", alpha=0.85,
    )
    ax.add_patch(fill)
    ax.add_patch(border)
    weight = "bold" if bold else "normal"
    ax.text(cx, y, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT, fontweight=weight, zorder=5)
    if shape_text:
        ax.text(cx + width / 2 + 0.08, y, shape_text,
                ha="left", va="center", fontsize=6.5, color=SHAPE,
                fontstyle="italic", zorder=5)
    return y


def draw_arrow(ax, y_from, y_to, color=ARROW):
    """Draw a downward arrow between two boxes."""
    ax.annotate(
        "", xy=(CENTER_X, y_to + BOX_H / 2 + 0.02),
        xytext=(CENTER_X, y_from - BOX_H / 2 - 0.02),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                        shrinkA=0, shrinkB=0),
        zorder=3,
    )


def draw_bracket(ax, y_top, y_bot, label, side="left"):
    """Draw a bracket on the side with a label."""
    if side == "left":
        bx = CENTER_X - BOX_W / 2 - 0.35
        ha = "right"
        tx = bx - 0.08
    else:
        bx = CENTER_X + BOX_W / 2 + 0.35
        ha = "left"
        tx = bx + 0.08
    ax.plot([bx, bx], [y_top, y_bot], color=GRAY, lw=1.2, alpha=0.6, zorder=2)
    ax.plot([bx, bx + 0.08 * (1 if side == "left" else -1)], [y_top, y_top],
            color=GRAY, lw=1.2, alpha=0.6, zorder=2)
    ax.plot([bx, bx + 0.08 * (1 if side == "left" else -1)], [y_bot, y_bot],
            color=GRAY, lw=1.2, alpha=0.6, zorder=2)
    mid = (y_top + y_bot) / 2
    ax.text(tx, mid, label, ha=ha, va="center", fontsize=7,
            color=GRAY, rotation=90 if side == "left" else 270, zorder=5)


# ── Build figure ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Title
ax.text(CENTER_X, FIG_H - 0.35, "nanochat Architecture Pipeline",
        ha="center", va="center", fontsize=15, fontweight="bold", color=TEXT)
ax.text(CENTER_X, FIG_H - 0.7, "GPT-2 backbone  ·  32 768 vocab  ·  768 dim  ·  12 heads  ·  12 layers",
        ha="center", va="center", fontsize=7.5, color=SHAPE)

y = MARGIN_TOP

# ── Token IDs ───────────────────────────────────────────────────────────────
draw_box(ax, y, "Token IDs", BLUE, "(B, T)", bold=True)
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── wte Embedding ───────────────────────────────────────────────────────────
draw_box(ax, y, "wte  Token Embedding", BLUE, "(B, T, 768)")
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── RMSNorm ─────────────────────────────────────────────────────────────────
draw_box(ax, y, "RMSNorm (pre-block)", PURPLE, "(B, T, 768)")
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── Smear ───────────────────────────────────────────────────────────────────
draw_box(ax, y, "Smear — bigram injection", GRAY, "(B, T, 768)")
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── Transformer Block banner ────────────────────────────────────────────────
block_top_y = y + BOX_H / 2 + 0.12

# resid_lambda + x0 re-injection label
draw_box(ax, y, "resid_lambda · x₀ re-injection", GRAY,
         width=BOX_W, height=0.30, fontsize=7.5, alpha=0.10)
y_prev = y
y -= SMALL_STEP
draw_arrow(ax, y_prev, y)

# ── Attention sub-block ─────────────────────────────────────────────────────
draw_box(ax, y, "RMSNorm", PURPLE, "(B, T, 768)", fontsize=8)
y_prev = y
y -= SMALL_STEP
draw_arrow(ax, y_prev, y)

draw_box(ax, y, "CausalSelfAttention", GREEN,
         width=BOX_W, bold=True)
y_prev = y
y -= 0.35
# Sub-details inside attention
ax.text(CENTER_X, y + 0.06,
        "Q / K / V projections  →  RoPE  →  Flash Attention  →  Value Embeddings",
        ha="center", va="center", fontsize=6.5, color=GREEN, zorder=5)
y -= 0.18
draw_arrow(ax, y_prev, y)

draw_box(ax, y, "Residual Add", GRAY, "(B, T, 768)",
         height=0.30, fontsize=7.5, alpha=0.10)
y_prev = y
y -= SMALL_STEP
draw_arrow(ax, y_prev, y)

# ── MLP sub-block ───────────────────────────────────────────────────────────
draw_box(ax, y, "RMSNorm", PURPLE, "(B, T, 768)", fontsize=8)
y_prev = y
y -= SMALL_STEP
draw_arrow(ax, y_prev, y)

draw_box(ax, y, "MLP  (relu² activation)", ORANGE, "(B, T, 768)", bold=True)
y_prev = y
y -= SMALL_STEP
draw_arrow(ax, y_prev, y)

draw_box(ax, y, "Residual Add", GRAY, "(B, T, 768)",
         height=0.30, fontsize=7.5, alpha=0.10)

# Cache label at layer 6
ax.text(CENTER_X + BOX_W / 2 + 0.12, y, "← cache at layer 6",
        ha="left", va="center", fontsize=6.5, color=GRAY,
        fontstyle="italic", zorder=5)

block_bot_y = y - BOX_H / 2 - 0.08

# ── Bracket around transformer block ───────────────────────────────────────
draw_bracket(ax, block_top_y, block_bot_y, "×12 Transformer Blocks", side="left")

# Draw a subtle background for the block region
block_bg = patches.FancyBboxPatch(
    (CENTER_X - BOX_W / 2 - 0.15, block_bot_y),
    BOX_W + 0.30, block_top_y - block_bot_y,
    boxstyle="round,pad=0.1", linewidth=0.8,
    edgecolor=GRAY, facecolor=GRAY, alpha=0.04, linestyle="--",
    zorder=1,
)
ax.add_patch(block_bg)

y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── Backout subtraction ─────────────────────────────────────────────────────
draw_box(ax, y, "Backout Subtraction (mid-layer state)", GRAY, "(B, T, 768)")
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── Final RMSNorm ───────────────────────────────────────────────────────────
draw_box(ax, y, "RMSNorm (final)", PURPLE, "(B, T, 768)")
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── lm_head ────────────────────────────────────────────────────────────────
draw_box(ax, y, "lm_head  Linear Projection", BLUE, "(B, T, 32768)", bold=True)
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── Softcap + crop ─────────────────────────────────────────────────────────
draw_box(ax, y, "Vocab Crop  +  fp32 Cast  +  Softcap (30.0)", GRAY, "(B, T, V)")
y_prev = y
y -= STEP
draw_arrow(ax, y_prev, y)

# ── Branching: training vs inference ────────────────────────────────────────
branch_y = y
branch_w = 2.1
branch_h = BOX_H

# Training branch (left)
lx = CENTER_X - 1.4
draw_box(ax, branch_y, "cross_entropy loss", RED,
         width=branch_w, bold=True, cx=lx)
ax.text(lx, branch_y - branch_h / 2 - 0.12, "Training",
        ha="center", va="center", fontsize=7, color=RED, fontweight="bold")

# Inference branch (right)
rx = CENTER_X + 1.4
fill_r = patches.FancyBboxPatch(
    (rx - branch_w / 2, branch_y - branch_h / 2), branch_w, branch_h,
    boxstyle="round,pad=0.08", linewidth=0,
    edgecolor="none", facecolor=RED, alpha=0.18,
)
border_r = patches.FancyBboxPatch(
    (rx - branch_w / 2, branch_y - branch_h / 2), branch_w, branch_h,
    boxstyle="round,pad=0.08", linewidth=1.6,
    edgecolor=RED, facecolor="none", alpha=0.85,
)
ax.add_patch(fill_r)
ax.add_patch(border_r)
ax.text(rx, branch_y, "softmax → sample", ha="center", va="center",
        fontsize=9, color=TEXT, fontweight="bold", zorder=5)
ax.text(rx, branch_y - branch_h / 2 - 0.12, "Inference",
        ha="center", va="center", fontsize=7, color=RED, fontweight="bold")

# Forking arrows from last box to the two branches
fork_y_from = y_prev - BOX_H / 2 - 0.02
fork_y_to = branch_y + branch_h / 2 + 0.02

ax.annotate(
    "", xy=(lx, fork_y_to),
    xytext=(CENTER_X, fork_y_from),
    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.2,
                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.15"),
    zorder=3,
)
ax.annotate(
    "", xy=(rx, fork_y_to),
    xytext=(CENTER_X, fork_y_from),
    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.2,
                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=-0.15"),
    zorder=3,
)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_items = [
    (BLUE,   "Embedding / Projection"),
    (GREEN,  "Attention"),
    (ORANGE, "MLP"),
    (PURPLE, "Normalization"),
    (RED,    "Loss / Output"),
    (GRAY,   "Modern Additions"),
]
lx_start = 0.5
ly = 0.45
for i, (color, label) in enumerate(legend_items):
    xi = lx_start + i * 1.2
    sq = patches.FancyBboxPatch(
        (xi, ly - 0.06), 0.12, 0.12,
        boxstyle="round,pad=0.02", linewidth=0.8,
        edgecolor=color, facecolor=color, alpha=0.35,
    )
    ax.add_patch(sq)
    ax.text(xi + 0.17, ly, label, ha="left", va="center",
            fontsize=5.5, color=TEXT, zorder=5)

plt.tight_layout(pad=0.3)

out = "/Users/argon/projects/learn_you_a_llm/journal/diagrams/architecture_pipeline.svg"
fig.savefig(out, format="svg", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved → {out}")
