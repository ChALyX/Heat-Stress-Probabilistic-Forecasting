"""Generate a publication-quality model architecture diagram using matplotlib."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helper functions ────────────────────────────────────────────────────

def draw_box(x, y, w, h, label, sublabel, fc, ec, fontsize=10, sublabel_size=7.5,
             bold=True, linestyle="-", linewidth=1.5, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=fc, edgecolor=ec, linewidth=linewidth,
                         linestyle=linestyle, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2 + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight=weight, zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                ha="center", va="center", fontsize=sublabel_size, color="#555", zorder=3)

def draw_arrow(x1, y1, x2, y2, color="#444", lw=1.3, style="->", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, linewidth=lw,
                            connectionstyle=connectionstyle, zorder=1,
                            mutation_scale=12)
    ax.add_patch(arrow)

def draw_dashed_arrow(x1, y1, x2, y2, color="#888", lw=1.0, connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", color=color, linewidth=lw,
                            linestyle="--", connectionstyle=connectionstyle,
                            zorder=1, mutation_scale=10)
    ax.add_patch(arrow)

# ── Title ───────────────────────────────────────────────────────────────

ax.text(7, 8.65, "Probabilistic GRU Architecture for Coastal Heat-Stress Forecasting",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1a1a1a")
ax.text(7, 8.35, "~328K parameters  |  72h lookback → 24h probabilistic forecast  |  2 targets (HI, WBGT-like)",
        ha="center", va="center", fontsize=8.5, color="#666")

# ── Section labels ──────────────────────────────────────────────────────

ax.text(0.3, 7.9, "ENCODER", fontsize=9, fontweight="bold", color="#1565C0", style="italic")
ax.text(0.3, 4.55, "CONTEXT", fontsize=9, fontweight="bold", color="#6A1B9A", style="italic")
ax.text(0.3, 2.85, "DECODER", fontsize=9, fontweight="bold", color="#00695C", style="italic")
ax.text(0.3, 0.95, "OUTPUT", fontsize=9, fontweight="bold", color="#BF360C", style="italic")

# ── Row 1: Encoder ──────────────────────────────────────────────────────

# Input
draw_box(0.5, 7.0, 2.2, 0.7, "Input", "X ∈ ℝ^{B×72×24}", "#F0F4FF", "#4A6FA5")

# Feature Projection
draw_box(3.5, 7.0, 3.0, 0.7, "Feature Projection",
         "Linear(24→64) + LayerNorm + GELU + Dropout", "#E8F5E9", "#43A047")

# GRU Encoder
draw_box(7.3, 7.0, 2.8, 0.7, "GRU Encoder",
         "2-layer, hidden=128, batch_first", "#E3F2FD", "#1976D2")

# Arrows
draw_arrow(2.7, 7.35, 3.5, 7.35)
draw_arrow(6.5, 7.35, 7.3, 7.35)

# GRU outputs
draw_box(11.0, 7.25, 2.5, 0.35, "encoder_outputs", "[B, 72, 128]",
         "#E3F2FD", "#1976D2", fontsize=8.5, sublabel_size=7, bold=False)
draw_box(11.0, 6.8, 2.5, 0.35, "last hidden h_T", "[B, 128]",
         "#FFF3E0", "#F57C00", fontsize=8.5, sublabel_size=7, bold=False)

draw_arrow(10.1, 7.5, 11.0, 7.42, color="#1976D2")
draw_arrow(10.1, 7.2, 11.0, 6.98, color="#F57C00")

# ── Row 2: Attention (optional) ─────────────────────────────────────────

# Multi-Head Attention
draw_box(4.5, 5.4, 4.0, 0.8, "Multi-Head Attention",
         "4 heads, Q=h_T, K=V=encoder_outputs, dropout=0.2",
         "#FCE4EC", "#C62828", linestyle="--", linewidth=1.5)

# "(optional — ablation)" label
ax.text(8.6, 6.05, "(ablation: optional)", fontsize=7.5, color="#C62828", style="italic")

# Arrows from GRU outputs to MHA
draw_arrow(12.25, 7.25, 7.5, 6.2, color="#1976D2", connectionstyle="arc3,rad=-0.15")
draw_arrow(12.25, 6.8, 5.5, 6.2, color="#F57C00", connectionstyle="arc3,rad=0.15")

# K,V and Q labels
ax.text(10.0, 6.65, "K, V", fontsize=7.5, color="#1976D2", fontweight="bold")
ax.text(8.5, 6.55, "Q", fontsize=7.5, color="#F57C00", fontweight="bold")

# ── Row 3: Context ──────────────────────────────────────────────────────

# Concat + LayerNorm
draw_box(4.0, 4.0, 5.0, 0.7, "Concat [h_T ; attn_context] & LayerNorm",
         "→ context ∈ ℝ^{B×256}  (or ℝ^{B×128} w/o attention)",
         "#F3E5F5", "#7B1FA2")

# Arrow: MHA → Concat
draw_arrow(6.5, 5.4, 6.5, 4.7)

# Skip connection h_T → Concat
draw_dashed_arrow(5.5, 5.4, 5.0, 4.7, color="#F57C00", connectionstyle="arc3,rad=0.2")
ax.text(4.55, 5.1, "skip", fontsize=7, color="#F57C00", style="italic")

# ── Horizon Embedding ───────────────────────────────────────────────────

draw_box(0.5, 3.3, 2.8, 0.7, "Horizon Embedding",
         "nn.Embedding(24, 128)  (ablation: optional)",
         "#FFFDE7", "#F9A825", linestyle="--", linewidth=1.5)

ax.text(1.9, 4.15, "h_ids = [0,1,...,23]", fontsize=7, color="#888", ha="center")
draw_arrow(1.9, 4.05, 1.9, 4.0, color="#F9A825")

# ── Row 4: Decoder ──────────────────────────────────────────────────────

# Repeat & Cat
draw_box(3.5, 2.7, 4.5, 0.65, "Repeat(×24) & Concatenate",
         "[B,24,256] ⊕ [B,24,128] → [B, 24, 384]",
         "#E8EAF6", "#3F51B5")

# Arrow: Concat → Repeat
draw_arrow(6.5, 4.0, 5.75, 3.35)
# Arrow: Horizon Emb → Repeat
draw_arrow(2.9, 3.3, 3.5, 3.02, color="#F9A825", connectionstyle="arc3,rad=0.1")

# Residual Decoder
draw_box(3.5, 1.7, 4.5, 0.75, "Residual Decoder",
         "Linear(384→128)\nBlock×2: [Linear→GELU→Dropout] + Residual + LayerNorm",
         "#E0F7FA", "#00838F")

# Arrow: Repeat → Decoder
draw_arrow(5.75, 2.7, 5.75, 2.45)

# ── Row 5: Output Heads ────────────────────────────────────────────────

# Mean Head
draw_box(3.5, 0.5, 2.0, 0.65, "Mean Head", "Linear(128→2)",
         "#E8F5E9", "#2E7D32")

# LogVar Head
draw_box(6.5, 0.5, 2.0, 0.65, "LogVar Head", "Linear(128→2), clamp[−8, 6]",
         "#FFF3E0", "#E65100")

# Arrows
draw_arrow(4.8, 1.7, 4.5, 1.15, color="#2E7D32")
draw_arrow(6.7, 1.7, 7.5, 1.15, color="#E65100")

# Output tensors
draw_box(3.7, 0.0, 1.6, 0.35, "μ ∈ ℝ^{B×24×2}", "",
         "#C8E6C9", "#2E7D32", fontsize=9.5, bold=True)
draw_box(6.7, 0.0, 1.6, 0.35, "log σ² ∈ ℝ^{B×24×2}", "",
         "#FFE0B2", "#E65100", fontsize=9.5, bold=True)

draw_arrow(4.5, 0.5, 4.5, 0.35, color="#2E7D32")
draw_arrow(7.5, 0.5, 7.5, 0.35, color="#E65100")

# ── Annotation boxes ───────────────────────────────────────────────────

# Loss function
draw_box(9.5, 1.5, 4.0, 0.9, "Horizon-Weighted Composite Loss",
         "ℒ = Σ w_h · NLL(μ,σ²,y) + 0.1·MSE + 10⁻⁴·Var_penalty\nw_h: 1.0 (h=1) → 2.0 (h=24)",
         "#FAFAFA", "#9E9E9E", fontsize=9.5, sublabel_size=7.5, linestyle="--", linewidth=1.0)

# MC Dropout
draw_box(9.5, 0.3, 4.0, 0.9, "Inference: MC Dropout",
         "30 forward passes → epistemic uncertainty\nσ²_total = σ²_aleatoric + σ²_epistemic",
         "#FAFAFA", "#9E9E9E", fontsize=9.5, sublabel_size=7.5, linestyle="--", linewidth=1.0)

# ── Legend for dashed boxes ─────────────────────────────────────────────

ax.plot([10.0, 10.5], [2.65, 2.65], linestyle="--", color="#C62828", linewidth=1.5)
ax.text(10.6, 2.65, "= ablation-optional component", fontsize=7.5, va="center", color="#555")

plt.tight_layout(pad=0.5)
fig.savefig("results/model_architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig("results/model_architecture.svg", bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: results/model_architecture.png")
print("Saved: results/model_architecture.svg")
