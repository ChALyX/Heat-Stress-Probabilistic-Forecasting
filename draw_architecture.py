"""Generate a publication-quality model architecture diagram using matplotlib.

Strict top-to-bottom centered layout for clarity.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helpers ─────────────────────────────────────────────────────────────

def box(x, y, w, h, title, sub="", fc="#fff", ec="#333", fs=10.5, sub_fs=8,
        bold=True, ls="-", lw=1.5):
    """Draw a rounded box centered at (x+w/2, y+h/2)."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           linestyle=ls, zorder=2)
    ax.add_patch(patch)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + 0.13, title, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)
        ax.text(cx, cy - 0.17, sub, ha="center", va="center",
                fontsize=sub_fs, color="#555", zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)


def arrow(x1, y1, x2, y2, c="#444", lw=1.3, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 color=c, linewidth=lw, connectionstyle=cs,
                                 mutation_scale=13, zorder=1))

def darrow(x1, y1, x2, y2, c="#888", lw=1.0, cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 color=c, linewidth=lw, linestyle="--",
                                 connectionstyle=cs, mutation_scale=10, zorder=1))

# Center X for the main column
CX = 5.0  # center of diagram

# ── Title ───────────────────────────────────────────────────────────────

ax.text(CX, 13.55, "Probabilistic GRU Architecture",
        ha="center", fontsize=15, fontweight="bold", color="#1a1a1a")
ax.text(CX, 13.2, "for Coastal Heat-Stress Forecasting",
        ha="center", fontsize=13, fontweight="bold", color="#1a1a1a")
ax.text(CX, 12.85, "~328K params  |  72h lookback → 24h forecast  |  2 targets (HI, WBGT-like)",
        ha="center", fontsize=8, color="#777")

# ── Row 0: Input  (y ≈ 12.0) ───────────────────────────────────────────

box(2.5, 11.9, 5.0, 0.65, "Input",
    "X ∈ ℝ  (B × 72 × 24 features)", "#EEF2FF", "#4A6FA5")

# ── Row 1: Feature Projection (y ≈ 10.9) ───────────────────────────────

arrow(CX, 11.9, CX, 11.6)

box(2.0, 10.85, 6.0, 0.65, "Feature Projection",
    "Linear(24→64) → LayerNorm → GELU → Dropout(0.2)", "#E8F5E9", "#388E3C")

# ── Row 2: GRU Encoder (y ≈ 9.8) ───────────────────────────────────────

arrow(CX, 10.85, CX, 10.55)

box(2.0, 9.75, 6.0, 0.7, "GRU Encoder",
    "2-layer, hidden_size=128, batch_first=True", "#E3F2FD", "#1565C0")

# ── Row 3: Split — h_T and encoder_outputs (y ≈ 8.85) ──────────────────

# Two output boxes side by side
arrow(3.7, 9.75, 3.2, 9.45, c="#F57C00")  # left: h_T
arrow(6.3, 9.75, 6.8, 9.45, c="#1565C0")  # right: encoder_outputs

box(1.5, 8.85, 3.0, 0.5, "h_T  (Query)", "[B, 128]",
    "#FFF3E0", "#F57C00", fs=10, sub_fs=7.5)

box(5.5, 8.85, 3.0, 0.5, "Encoder Outputs (K, V)", "[B, 72, 128]",
    "#E3F2FD", "#1565C0", fs=10, sub_fs=7.5)

# ── Row 4: Multi-Head Attention (y ≈ 7.7) ──────────────────────────────

arrow(3.0, 8.85, 4.0, 8.45, c="#F57C00")   # h_T → MHA
arrow(7.0, 8.85, 6.0, 8.45, c="#1565C0")   # enc_out → MHA

box(2.5, 7.7, 5.0, 0.65, "Multi-Head Attention",
    "4 heads, dropout=0.2      ⟵ ablation: optional",
    "#FCE4EC", "#C62828", ls="--", lw=1.8)

# Q and K,V labels near arrows
ax.text(3.2, 8.6, "Q", fontsize=8, color="#F57C00", fontweight="bold")
ax.text(6.7, 8.6, "K, V", fontsize=8, color="#1565C0", fontweight="bold")

# ── Row 5: Concat & LayerNorm (y ≈ 6.55) ───────────────────────────────

arrow(CX, 7.7, CX, 7.25)

box(2.0, 6.5, 6.0, 0.65, "Concat [ h_T ; context ] & LayerNorm",
    "→ summary ∈ ℝ^{B×256}  (ℝ^{B×128} without attention)",
    "#F3E5F5", "#6A1B9A")

# Skip connection: h_T directly to Concat (bypassing attention)
darrow(1.8, 8.85, 2.3, 7.15, c="#F57C00", cs="arc3,rad=0.35")
ax.text(1.15, 8.0, "skip", fontsize=7.5, color="#F57C00", style="italic")

# ── Row 6: Repeat & Cat with Horizon Embedding (y ≈ 5.2) ───────────────

arrow(CX, 6.5, CX, 6.2)

# Horizon Embedding — to the left
box(0.3, 5.2, 3.0, 0.65, "Horizon Embedding",
    "nn.Embedding(24, 128)  ⟵ ablation: optional",
    "#FFFDE7", "#F9A825", fs=10, ls="--", lw=1.8)

# h_ids label
ax.text(1.8, 6.0, "ids = [0, 1, ..., 23]", fontsize=7, color="#999", ha="center")
arrow(1.8, 5.93, 1.8, 5.85, c="#F9A825")

# Repeat & Cat — main column
box(2.5, 4.95, 5.0, 0.65, "Repeat(×24) & Concatenate",
    "[B,24,256] ⊕ [B,24,128] → [B, 24, 384]",
    "#E8EAF6", "#303F9F")

# Arrow: Horizon Embedding → Repeat&Cat
arrow(3.3, 5.52, 3.5, 5.38, c="#F9A825", cs="arc3,rad=-0.1")

# ── Row 7: Residual Decoder (y ≈ 3.7) ──────────────────────────────────

arrow(CX, 4.95, CX, 4.65)

box(2.0, 3.7, 6.0, 0.85, "Residual Decoder",
    "Linear(384→128)\nBlock ×2 : Linear → GELU → Dropout(0.2) + Residual + LayerNorm",
    "#E0F7FA", "#00695C")

# ── Row 8: Output Heads (y ≈ 2.5) ──────────────────────────────────────

arrow(3.7, 3.7, 3.2, 3.35, c="#2E7D32")   # → Mean Head
arrow(6.3, 3.7, 6.8, 3.35, c="#E65100")   # → LogVar Head

box(1.5, 2.55, 3.0, 0.7, "Mean Head", "Linear(128→2)",
    "#E8F5E9", "#2E7D32")
box(5.5, 2.55, 3.0, 0.7, "LogVar Head", "Linear(128→2), clamp[−8, 6]",
    "#FFF3E0", "#E65100")

# ── Row 9: Outputs (y ≈ 1.5) ───────────────────────────────────────────

arrow(3.0, 2.55, 3.0, 2.25, c="#2E7D32")
arrow(7.0, 2.55, 7.0, 2.25, c="#E65100")

box(1.8, 1.65, 2.4, 0.5, "μ ∈ ℝ^{B×24×2}", "",
    "#C8E6C9", "#2E7D32", fs=11, bold=True)
box(5.8, 1.65, 2.4, 0.5, "log σ² ∈ ℝ^{B×24×2}", "",
    "#FFE0B2", "#E65100", fs=11, bold=True)

# Target label
ax.text(CX, 1.35, "Targets:  Heat Index (HI)  &  WBGT-like",
        ha="center", fontsize=8, color="#666")

# ── Annotation boxes (bottom) ──────────────────────────────────────────

box(0.5, 0.2, 4.0, 0.9, "Horizon-Weighted Composite Loss",
    "ℒ = Σ wh · NLL(μ,σ²,y) + 0.1·MSE + 10⁻⁴·Var_penalty\n"
    "wh : 1.0 (h=1) → 2.0 (h=24)",
    "#FAFAFA", "#BDBDBD", fs=9.5, sub_fs=7.5, ls="--", lw=1.0)

box(5.5, 0.2, 4.0, 0.9, "Inference: MC Dropout",
    "30 forward passes with dropout enabled\n"
    "σ²_total = σ²_aleatoric + σ²_epistemic",
    "#FAFAFA", "#BDBDBD", fs=9.5, sub_fs=7.5, ls="--", lw=1.0)

# ── Section labels (left margin) ───────────────────────────────────────

sections = [
    (12.2, "ENCODER", "#1565C0"),
    (8.1,  "ATTENTION", "#C62828"),
    (6.8,  "CONTEXT", "#6A1B9A"),
    (5.5,  "HORIZON", "#F9A825"),
    (4.3,  "DECODER", "#00695C"),
    (2.9,  "OUTPUT", "#BF360C"),
]
for sy, sname, sc in sections:
    ax.text(0.05, sy, sname, fontsize=7, fontweight="bold",
            color=sc, style="italic", rotation=0, va="center")

# ── Legend ──────────────────────────────────────────────────────────────

ax.plot([3.8, 4.3], [1.35, 1.35], linestyle="--", color="#C62828", linewidth=1.5)
ax.text(4.4, 1.35, "= ablation-optional", fontsize=7, va="center", color="#666")

plt.tight_layout(pad=0.3)
fig.savefig("results/model_architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig("results/model_architecture.svg", bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: results/model_architecture.png")
print("Saved: results/model_architecture.svg")
