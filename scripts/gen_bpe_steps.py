"""Generate an SVG diagram showing BPE tokenization steps for 'lowest'."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Colors ──────────────────────────────────────────────────────────────────
BG     = "#FAFAFA"
TEXT   = "#212121"
ARROW  = "#37474F"

# Distinct token colors — assigned per unique token, persists through merges
TOKEN_COLORS = {
    "l":   "#2196F3",  # blue
    "o":   "#4CAF50",  # green
    "w":   "#FF9800",  # orange
    "e":   "#9C27B0",  # purple
    "s":   "#F44336",  # red
    "t":   "#00BCD4",  # teal
    "es":  "#E91E63",  # pink (merge of e+s)
    "est": "#795548",  # brown (merge of es+t)
    "lo":  "#3F51B5",  # indigo (merge of l+o)
    "low": "#009688",  # dark teal (merge of lo+w)
}

# ── Data ────────────────────────────────────────────────────────────────────
steps = [
    {"label": "Start: individual characters",
     "tokens": ["l", "o", "w", "e", "s", "t"],
     "merge": None, "freq": None},
    {"label": "Step 1: merge most frequent pair ('e','s')",
     "tokens": ["l", "o", "w", "es", "t"],
     "merge": (3, 4), "freq": "freq=7"},  # indices in PREVIOUS row
    {"label": "Step 2: merge ('es','t')",
     "tokens": ["l", "o", "w", "est"],
     "merge": (3, 4), "freq": "freq=4"},
    {"label": "Step 3: merge ('l','o')",
     "tokens": ["lo", "w", "est"],
     "merge": (0, 1), "freq": "freq=3"},
    {"label": "Step 4: merge ('lo','w')",
     "tokens": ["low", "est"],
     "merge": (0, 1), "freq": "freq=2"},
]

# ── Layout ──────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 8
ROW_H = 1.2           # vertical space per step
BOX_H = 0.55
BOX_PAD = 0.15        # horizontal gap between token boxes
CHAR_W = 0.42         # width per character in a token
MIN_BOX_W = 0.6
TOP_Y = FIG_H - 1.0
LEFT_MARGIN = 2.8     # x where token boxes start

# ── Helpers ─────────────────────────────────────────────────────────────────

def token_box_width(tok):
    return max(MIN_BOX_W, len(tok) * CHAR_W + 0.3)


def draw_token_boxes(ax, tokens, cx_start, y):
    """Draw a row of colored token boxes. Returns list of (cx, w) per token."""
    positions = []
    x = cx_start
    for tok in tokens:
        w = token_box_width(tok)
        color = TOKEN_COLORS[tok]
        fill = patches.FancyBboxPatch(
            (x, y - BOX_H / 2), w, BOX_H,
            boxstyle="round,pad=0.06", linewidth=0,
            edgecolor="none", facecolor=color, alpha=0.20,
        )
        border = patches.FancyBboxPatch(
            (x, y - BOX_H / 2), w, BOX_H,
            boxstyle="round,pad=0.06", linewidth=1.8,
            edgecolor=color, facecolor="none", alpha=0.85,
        )
        ax.add_patch(fill)
        ax.add_patch(border)
        ax.text(x + w / 2, y, f"'{tok}'", ha="center", va="center",
                fontsize=11, color=TEXT, fontweight="bold", fontfamily="monospace",
                zorder=5)
        positions.append((x + w / 2, w))
        x += w + BOX_PAD
    return positions


# ── Build figure ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Title
ax.text(FIG_W / 2, FIG_H - 0.35,
        "BPE Tokenization: 'lowest' \u2192 ['low', 'est']",
        ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)

# Draw each step
prev_positions = None
for i, step in enumerate(steps):
    y = TOP_Y - i * ROW_H

    # Step label on the left
    ax.text(0.2, y, step["label"],
            ha="left", va="center", fontsize=9, color="#546E7A", zorder=5)

    # Token boxes
    positions = draw_token_boxes(ax, step["tokens"], LEFT_MARGIN, y)

    # Token count on the right
    total_w = sum(token_box_width(t) for t in step["tokens"]) + BOX_PAD * (len(step["tokens"]) - 1)
    count_x = LEFT_MARGIN + total_w + 0.3
    ax.text(count_x, y, f"{len(step['tokens'])} tokens",
            ha="left", va="center", fontsize=9, color="#78909C",
            fontstyle="italic", zorder=5)

    # Merge arrows from previous row
    if step["merge"] is not None and prev_positions is not None:
        m_left, m_right = step["merge"]
        # Arrow from each merging token in the previous row down to the merged token in this row
        # Find which token in this row is the merged result
        # The merge indices refer to positions in the PREVIOUS row's tokens
        prev_step = steps[i - 1]
        merged_text = prev_step["tokens"][m_left] + prev_step["tokens"][m_right]

        # Find merged token position in current row
        merged_idx = step["tokens"].index(merged_text)
        target_cx = positions[merged_idx][0]
        target_y_top = y + BOX_H / 2 + 0.02

        for m_idx in [m_left, m_right]:
            src_cx = prev_positions[m_idx][0]
            src_y_bot = (y + ROW_H) - BOX_H / 2 - 0.02
            ax.annotate(
                "", xy=(target_cx, target_y_top),
                xytext=(src_cx, src_y_bot),
                arrowprops=dict(
                    arrowstyle="-|>", color=TOKEN_COLORS[merged_text],
                    lw=1.5, shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=3,
            )

        # Frequency label near the arrow midpoint
        if step["freq"]:
            mid_x = (prev_positions[m_left][0] + prev_positions[m_right][0]) / 2
            mid_y = y + ROW_H / 2
            ax.text(mid_x, mid_y + 0.12, step["freq"],
                    ha="center", va="center", fontsize=7.5,
                    color=TOKEN_COLORS[merged_text], fontweight="bold",
                    fontstyle="italic", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.8))

    prev_positions = positions

# Final annotation
final_y = TOP_Y - (len(steps) - 1) * ROW_H - 0.65
ax.text(FIG_W / 2, final_y,
        "Final vocabulary: 2 tokens cover the entire word",
        ha="center", va="center", fontsize=11, color="#37474F",
        fontweight="bold", fontstyle="italic")

plt.tight_layout(pad=0.3)

out = "/Users/argon/projects/learn_you_a_llm/journal/diagrams/bpe_merge_steps.svg"
fig.savefig(out, format="svg", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved \u2192 {out}")
