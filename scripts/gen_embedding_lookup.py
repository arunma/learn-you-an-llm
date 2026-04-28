"""Generate an SVG diagram showing nn.Embedding as a lookup table."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ── Colors ──────────────────────────────────────────────────────────────────
BG       = "#FAFAFA"
TEXT     = "#212121"
ARROW    = "#37474F"
BLUE     = "#2196F3"
GREEN    = "#4CAF50"
TEAL     = "#009688"
GRAY     = "#607D8B"
LIGHT_BG = "#ECEFF1"

# Row colors for the three tokens
ROW_COLORS = ["#1565C0", "#2E7D32", "#00838F"]  # deep blue, deep green, deep teal
ROW_FILLS  = ["#BBDEFB", "#C8E6C9", "#B2DFDB"]  # light fills

# ── Layout ──────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 6

# ── Data ────────────────────────────────────────────────────────────────────
tokens = [
    (142,   "The"),
    (8417,  "cat"),
    (1923,  "sat"),
]

VOCAB_SIZE = 32768
EMBED_DIM  = 768

# ── Build figure ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# Title
ax.text(FIG_W / 2, FIG_H - 0.3,
        "nn.Embedding: Lookup Table, Not Computation",
        ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)

# ── Left: Token ID column ──────────────────────────────────────────────────
token_col_x = 0.8
token_col_top = 4.6
token_row_h = 0.65
token_box_w = 1.6
token_box_h = 0.50

ax.text(token_col_x + token_box_w / 2, token_col_top + 0.5,
        "Token IDs", ha="center", va="center",
        fontsize=10, fontweight="bold", color=GRAY)
ax.text(token_col_x + token_box_w / 2, token_col_top + 0.2,
        "shape: (3,)", ha="center", va="center",
        fontsize=8, color=GRAY, fontstyle="italic")

token_centers = []
for i, (tid, word) in enumerate(tokens):
    y = token_col_top - i * token_row_h
    color = ROW_COLORS[i]
    fill_c = ROW_FILLS[i]

    fill = patches.FancyBboxPatch(
        (token_col_x, y - token_box_h / 2), token_box_w, token_box_h,
        boxstyle="round,pad=0.06", linewidth=0,
        edgecolor="none", facecolor=fill_c, alpha=0.6,
    )
    border = patches.FancyBboxPatch(
        (token_col_x, y - token_box_h / 2), token_box_w, token_box_h,
        boxstyle="round,pad=0.06", linewidth=1.8,
        edgecolor=color, facecolor="none", alpha=0.85,
    )
    ax.add_patch(fill)
    ax.add_patch(border)
    ax.text(token_col_x + token_box_w / 2, y,
            f'{tid}  "{word}"', ha="center", va="center",
            fontsize=10, color=TEXT, fontweight="bold", fontfamily="monospace",
            zorder=5)
    token_centers.append((token_col_x + token_box_w, y))

# ── Center: Embedding table ────────────────────────────────────────────────
table_x = 3.6
table_y_top = 5.0
table_w = 2.8
table_h = 4.2
cell_h = 0.22

# Table background
table_bg = patches.FancyBboxPatch(
    (table_x, table_y_top - table_h), table_w, table_h,
    boxstyle="round,pad=0.05", linewidth=1.5,
    edgecolor=GRAY, facecolor=LIGHT_BG, alpha=0.4,
)
ax.add_patch(table_bg)

# Table header
ax.text(table_x + table_w / 2, table_y_top + 0.35,
        "Embedding Table", ha="center", va="center",
        fontsize=10, fontweight="bold", color=GRAY)
ax.text(table_x + table_w / 2, table_y_top + 0.1,
        f"nn.Embedding({VOCAB_SIZE}, {EMBED_DIM})",
        ha="center", va="center", fontsize=8, color=GRAY, fontfamily="monospace")

# Draw abbreviated rows with "..." gaps
# Row layout: row 0..2 at top, then "...", row 142 highlighted,
#   "...", row 1923 highlighted, "...", row 8417 highlighted, "...", row 32767

# We'll draw specific rows and ellipses
rng = np.random.RandomState(42)

def draw_table_row(y, label, highlight_idx=None):
    """Draw a single row in the embedding table."""
    row_y = y
    if highlight_idx is not None:
        color = ROW_COLORS[highlight_idx]
        fill_c = ROW_FILLS[highlight_idx]
        # Highlighted row
        row_fill = patches.FancyBboxPatch(
            (table_x + 0.05, row_y - cell_h / 2), table_w - 0.10, cell_h,
            boxstyle="round,pad=0.02", linewidth=1.2,
            edgecolor=color, facecolor=fill_c, alpha=0.5,
        )
        ax.add_patch(row_fill)
        ax.text(table_x + table_w / 2, row_y, label,
                ha="center", va="center", fontsize=7, color=color,
                fontweight="bold", fontfamily="monospace", zorder=5)
    else:
        ax.text(table_x + table_w / 2, row_y, label,
                ha="center", va="center", fontsize=6.5, color="#90A4AE",
                fontfamily="monospace", zorder=5)
    return row_y


# Layout rows from top to bottom
row_y = table_y_top - 0.25
draw_table_row(row_y, "row 0:     [0.12, -0.34, 0.56, ...]")
row_y -= cell_h
draw_table_row(row_y, "row 1:     [0.78, 0.02, -0.91, ...]")
row_y -= cell_h
draw_table_row(row_y, "row 2:     [-0.15, 0.67, 0.23, ...]")
row_y -= cell_h * 0.8
ax.text(table_x + table_w / 2, row_y, "\u22ee", ha="center", va="center",
        fontsize=10, color="#90A4AE")

# Row 142 ("The")
row_y -= cell_h * 0.8
r142_y = draw_table_row(row_y, "row 142:   [0.41, -0.87, 0.13, ...]", highlight_idx=0)
row_y -= cell_h * 0.8
ax.text(table_x + table_w / 2, row_y, "\u22ee", ha="center", va="center",
        fontsize=10, color="#90A4AE")

# Row 1923 ("sat")
row_y -= cell_h * 0.8
r1923_y = draw_table_row(row_y, "row 1923:  [-0.29, 0.54, -0.71, ...]", highlight_idx=2)
row_y -= cell_h * 0.8
ax.text(table_x + table_w / 2, row_y, "\u22ee", ha="center", va="center",
        fontsize=10, color="#90A4AE")

# Row 8417 ("cat")
row_y -= cell_h * 0.8
r8417_y = draw_table_row(row_y, "row 8417:  [0.63, 0.11, -0.45, ...]", highlight_idx=1)
row_y -= cell_h * 0.8
ax.text(table_x + table_w / 2, row_y, "\u22ee", ha="center", va="center",
        fontsize=10, color="#90A4AE")

# Bottom of table
row_y -= cell_h * 0.8
draw_table_row(row_y, "row 32767: [-0.08, 0.93, 0.17, ...]")

# Dimension label at bottom of table
ax.annotate("", xy=(table_x + table_w - 0.05, table_y_top - table_h - 0.15),
            xytext=(table_x + 0.05, table_y_top - table_h - 0.15),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
ax.text(table_x + table_w / 2, table_y_top - table_h - 0.30,
        f"{EMBED_DIM} dimensions", ha="center", va="center",
        fontsize=7, color=GRAY, fontstyle="italic")

# Vocab size label on left side of table
ax.annotate("", xy=(table_x - 0.15, table_y_top - table_h),
            xytext=(table_x - 0.15, table_y_top),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.0))
ax.text(table_x - 0.3, table_y_top - table_h / 2,
        f"{VOCAB_SIZE}\nrows", ha="center", va="center",
        fontsize=7, color=GRAY, fontstyle="italic", rotation=90)

# ── Arrows from token IDs to table rows ────────────────────────────────────
# Token 0 (142, "The") -> row 142
highlighted_rows = {0: r142_y, 1: r8417_y, 2: r1923_y}

for i, (src_x, src_y) in enumerate(token_centers):
    target_y = highlighted_rows[i]
    ax.annotate(
        "", xy=(table_x + 0.05, target_y),
        xytext=(src_x + 0.05, src_y),
        arrowprops=dict(
            arrowstyle="-|>", color=ROW_COLORS[i], lw=1.8,
            shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=0.1" if i != 1 else "arc3,rad=-0.05",
        ),
        zorder=4,
    )

# ── Right: Output vectors ──────────────────────────────────────────────────
out_x = 7.2
out_top = 4.6
out_box_w = 2.4
out_box_h = 0.50

ax.text(out_x + out_box_w / 2, out_top + 0.5,
        "Output Embeddings", ha="center", va="center",
        fontsize=10, fontweight="bold", color=GRAY)
ax.text(out_x + out_box_w / 2, out_top + 0.2,
        "shape: (3, 768)", ha="center", va="center",
        fontsize=8, color=GRAY, fontstyle="italic")

output_centers = []
for i, (tid, word) in enumerate(tokens):
    y = out_top - i * token_row_h
    color = ROW_COLORS[i]
    fill_c = ROW_FILLS[i]

    fill = patches.FancyBboxPatch(
        (out_x, y - out_box_h / 2), out_box_w, out_box_h,
        boxstyle="round,pad=0.06", linewidth=0,
        edgecolor="none", facecolor=fill_c, alpha=0.6,
    )
    border = patches.FancyBboxPatch(
        (out_x, y - out_box_h / 2), out_box_w, out_box_h,
        boxstyle="round,pad=0.06", linewidth=1.8,
        edgecolor=color, facecolor="none", alpha=0.85,
    )
    ax.add_patch(fill)
    ax.add_patch(border)
    ax.text(out_x + out_box_w / 2, y,
            f'"{word}" \u2192 [{EMBED_DIM} floats]',
            ha="center", va="center", fontsize=9, color=TEXT,
            fontweight="bold", fontfamily="monospace", zorder=5)
    output_centers.append((out_x, y))

# ── Arrows from table rows to output ───────────────────────────────────────
for i in range(len(tokens)):
    target_y = highlighted_rows[i]
    out_y = output_centers[i][1]
    ax.annotate(
        "", xy=(out_x - 0.05, out_y),
        xytext=(table_x + table_w - 0.05, target_y),
        arrowprops=dict(
            arrowstyle="-|>", color=ROW_COLORS[i], lw=1.8,
            shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=-0.1" if i != 1 else "arc3,rad=0.05",
        ),
        zorder=4,
    )

# ── Shape transformation note ──────────────────────────────────────────────
ax.text(FIG_W / 2, 0.55,
        "(3,)  \u2500\u2500  lookup  \u2500\u2500\u25b6  (3, 768)",
        ha="center", va="center", fontsize=11, color=TEXT,
        fontweight="bold", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                  edgecolor=BLUE, alpha=0.6, linewidth=1.2))

ax.text(FIG_W / 2, 0.12,
        "No computation \u2014 just row indexing. The 768 numbers per token are the learned parameters.",
        ha="center", va="center", fontsize=8, color=GRAY, fontstyle="italic")

plt.tight_layout(pad=0.3)

out = "/Users/argon/projects/learn_you_a_llm/journal/diagrams/embedding_lookup.svg"
fig.savefig(out, format="svg", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved \u2192 {out}")
