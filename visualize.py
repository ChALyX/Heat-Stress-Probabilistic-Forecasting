"""
Cross-experiment visualization script.

Generates publication-quality figures that compare results across:
- Training curves (loss history) for all model variants
- Cross-site performance comparison
- Ablation study bar charts
- Seasonal performance breakdown
- Feature importance analysis

Usage:
    python visualize.py                          # generate all figures
    python visualize.py --results-dir results    # specify results directory
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TARGET_DISPLAY = {"heat_index_c": "Heat Index (°C)", "wbgt_like_c": "WBGT-like (°C)"}
SITE_DISPLAY = {"qingdao": "Qingdao", "dubai": "Dubai", "singapore": "Singapore", "miami": "Miami"}
VARIANT_DISPLAY = {
    "full": "Full Model",
    "noattn": "w/o Attention",
    "nohorizon": "w/o Horizon Embed",
    "deterministic": "Deterministic",
}

PALETTE = [
    "#E63946",  # red
    "#2A9D8F",  # teal
    "#E9A820",  # amber
    "#264653",  # dark blue-gray
    "#8B5CF6",  # violet
    "#F97316",  # orange
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#84CC16",  # lime
]


def load_train_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_eval_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── 1. Training Curves ──────────────────────────────────────────────────────


def plot_training_curves(results_dir: Path, output_dir: Path) -> None:
    """Plot training/validation loss curves for all model variants."""

    # Collect all train JSON files
    train_files = sorted(results_dir.glob("*_train.json"))
    if not train_files:
        print("No training history files found.")
        return

    # Group by site
    site_groups: dict[str, list[tuple[str, dict]]] = {}
    for f in train_files:
        data = load_train_json(f)
        if data is None or "history" not in data:
            continue
        name = f.stem.replace("_train", "")
        # Parse variant and site from name like "full_qingdao" or "noattn_qingdao"
        parts = name.split("_", 1)
        variant = parts[0]
        site = parts[1] if len(parts) > 1 else "unknown"
        site_groups.setdefault(site, []).append((variant, data))

    # Plot all variants together
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    all_variants = []
    for site, entries in site_groups.items():
        for variant, data in entries:
            label = f"{VARIANT_DISPLAY.get(variant, variant)} ({SITE_DISPLAY.get(site, site)})"
            all_variants.append((label, data["history"]))

    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    for idx, (title, ylabel) in enumerate([("Training Loss", "Loss"), ("Validation Loss", "Loss")]):
        ax = axes[idx]
        key = "train_loss" if idx == 0 else "val_loss"
        for i, (label, history) in enumerate(all_variants):
            epochs = [h["epoch"] for h in history]
            values = [h[key] for h in history]
            color = PALETTE[i % len(PALETTE)]
            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]
            ax.plot(epochs, values, color=color, linewidth=1.8, label=label,
                    alpha=0.9, linestyle=ls, marker=mk, markersize=3,
                    markevery=max(1, len(epochs) // 8))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out = output_dir / "training_curves_all.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # Also plot per-site if multiple variants exist for same site
    for site, entries in site_groups.items():
        if len(entries) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Training Curves — {SITE_DISPLAY.get(site, site)} (Ablation)", fontsize=14, fontweight="bold")
        for idx, (title, key) in enumerate([("Training Loss", "train_loss"), ("Validation Loss", "val_loss")]):
            ax = axes[idx]
            for i, (variant, data) in enumerate(entries):
                epochs = [h["epoch"] for h in data["history"]]
                values = [h[key] for h in data["history"]]
                ax.plot(epochs, values, color=PALETTE[i % len(PALETTE)], linewidth=2,
                        label=VARIANT_DISPLAY.get(variant, variant))
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        plt.tight_layout()
        out = output_dir / f"training_curves_{site}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")


# ── 2. Cross-Site Comparison ────────────────────────────────────────────────


def plot_cross_site_comparison(results_dir: Path, output_dir: Path) -> None:
    """Grouped bar chart comparing full model performance across sites."""

    sites = ["qingdao", "dubai", "singapore", "miami"]
    metrics_keys = ["rmse", "mae", "crps", "coverage_90", "skill_score"]
    metric_labels = ["RMSE (°C)", "MAE (°C)", "CRPS", "Coverage@90%", "Skill Score"]

    # Load eval JSONs for full model at each site
    site_data = {}
    for site in sites:
        path = results_dir / f"eval_full_{site}.json"
        data = load_eval_json(path)
        if data:
            site_data[site] = data

    if len(site_data) < 2:
        print("Not enough cross-site data for comparison.")
        return

    targets = ["heat_index_c", "wbgt_like_c"]
    fig, axes = plt.subplots(2, len(metrics_keys), figsize=(18, 8))
    fig.suptitle("Cross-Site Performance Comparison (Full Model)", fontsize=14, fontweight="bold")

    bar_width = 0.18
    x = np.arange(len(site_data))

    for row, target in enumerate(targets):
        for col, (mk, ml) in enumerate(zip(metrics_keys, metric_labels)):
            ax = axes[row, col]
            values = []
            labels = []
            for site in site_data:
                if mk == "skill_score":
                    val = site_data[site].get("persistence_skill", {}).get(target, {}).get("skill_score", 0)
                elif mk == "coverage_90":
                    val = site_data[site].get(target, {}).get("coverage_90", 0)
                else:
                    val = site_data[site].get(target, {}).get(mk, 0)
                values.append(val)
                labels.append(SITE_DISPLAY.get(site, site))

            colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
            bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.85, edgecolor="white")

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel(ml)
            if row == 0:
                ax.set_title(ml)
            if col == 0:
                ax.set_ylabel(f"{TARGET_DISPLAY[target]}\n{ml}")

    plt.tight_layout()
    out = output_dir / "cross_site_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # Radar chart
    _plot_radar_chart(site_data, targets, output_dir)


def _plot_radar_chart(site_data: dict, targets: list[str], output_dir: Path) -> None:
    """Radar chart comparing sites on normalized metrics."""

    metrics_for_radar = ["rmse", "mae", "crps", "coverage_90"]
    labels = ["RMSE", "MAE", "CRPS", "Coverage@90%"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection="polar"))
    fig.suptitle("Cross-Site Radar Comparison (Full Model)", fontsize=14, fontweight="bold", y=1.02)

    for tidx, target in enumerate(targets):
        ax = axes[tidx]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        for sidx, (site, data) in enumerate(site_data.items()):
            values = []
            for mk in metrics_for_radar:
                val = data.get(target, {}).get(mk, 0)
                # For coverage, invert so that higher = better = further from center
                # For error metrics, use 1/(1+val) so lower error = further from center
                if mk == "coverage_90":
                    values.append(val)
                else:
                    values.append(1.0 / (1.0 + val))
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, color=PALETTE[sidx % len(PALETTE)],
                    label=SITE_DISPLAY.get(site, site), markersize=4)
            ax.fill(angles, values, alpha=0.08, color=PALETTE[sidx % len(PALETTE)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(TARGET_DISPLAY[target], pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    out = output_dir / "cross_site_radar.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Ablation Study Bar Chart ────────────────────────────────────────────


def plot_ablation_comparison(results_dir: Path, output_dir: Path) -> None:
    """Bar chart comparing ablation variants on Qingdao."""

    variants = ["full", "noattn", "nohorizon", "deterministic"]
    metrics_keys = ["rmse", "mae", "crps", "winkler_90"]
    metric_labels = ["RMSE (°C)", "MAE (°C)", "CRPS", "Winkler@90%"]
    targets = ["heat_index_c", "wbgt_like_c"]

    variant_data = {}
    for v in variants:
        path = results_dir / f"eval_{v}_qingdao.json"
        data = load_eval_json(path)
        if data:
            variant_data[v] = data

    if len(variant_data) < 2:
        print("Not enough ablation data.")
        return

    fig, axes = plt.subplots(2, len(metrics_keys), figsize=(16, 8))
    fig.suptitle("Ablation Study — Qingdao", fontsize=14, fontweight="bold")

    x = np.arange(len(variant_data))
    for row, target in enumerate(targets):
        for col, (mk, ml) in enumerate(zip(metrics_keys, metric_labels)):
            ax = axes[row, col]
            values = []
            labels = []
            for v in variant_data:
                val = variant_data[v].get(target, {}).get(mk, 0)
                values.append(val)
                labels.append(VARIANT_DISPLAY.get(v, v))

            colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
            bars = ax.bar(x, values, width=0.6, color=colors, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, fontsize=8)
            if row == 0:
                ax.set_title(ml)
            if col == 0:
                ax.set_ylabel(TARGET_DISPLAY[target])

    plt.tight_layout()
    out = output_dir / "ablation_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # Skill score comparison
    _plot_ablation_skill(variant_data, output_dir)


def _plot_ablation_skill(variant_data: dict, output_dir: Path) -> None:
    """Bar chart of skill scores across ablation variants."""

    targets = ["heat_index_c", "wbgt_like_c"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Persistence Skill Score — Ablation Comparison", fontsize=14, fontweight="bold")

    x = np.arange(len(variant_data))
    bar_width = 0.35

    for tidx, target in enumerate(targets):
        values = []
        for v in variant_data:
            ps = variant_data[v].get("persistence_skill", {}).get(target, {})
            values.append(ps.get("skill_score", 0))
        offset = (tidx - 0.5) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width, color=PALETTE[tidx],
                      alpha=0.85, label=TARGET_DISPLAY[target], edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_DISPLAY.get(v, v) for v in variant_data], rotation=15)
    ax.set_ylabel("Skill Score (1 - MSE_model / MSE_persist)")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend()

    plt.tight_layout()
    out = output_dir / "ablation_skill_score.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3b. Probabilistic vs Deterministic Comparison ─────────────────────────


def plot_prob_vs_det(results_dir: Path, output_dir: Path) -> None:
    """Focused comparison between probabilistic (full) and deterministic models on Qingdao."""

    prob_data = load_eval_json(results_dir / "eval_full_qingdao.json")
    det_data = load_eval_json(results_dir / "eval_deterministic_qingdao.json")
    if not prob_data or not det_data:
        print("Missing probabilistic or deterministic eval data for Qingdao.")
        return

    targets = ["heat_index_c", "wbgt_like_c"]
    model_labels = ["Probabilistic", "Deterministic"]
    model_colors = [PALETTE[1], PALETTE[0]]  # teal vs red

    # ── Figure 1: Overall metrics side-by-side ──
    metrics = ["rmse", "mae", "crps", "winkler_90", "gaussian_nll"]
    metric_labels = ["RMSE (°C)", "MAE (°C)", "CRPS", "Winkler@90%", "Gaussian NLL"]

    fig, axes = plt.subplots(2, len(metrics), figsize=(20, 8))
    fig.suptitle("Probabilistic vs Deterministic — Qingdao", fontsize=14, fontweight="bold")

    for row, target in enumerate(targets):
        for col, (mk, ml) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            vals = [
                prob_data.get(target, {}).get(mk, 0),
                det_data.get(target, {}).get(mk, 0),
            ]
            bars = ax.bar([0, 1], vals, width=0.5, color=model_colors, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(model_labels, fontsize=9)
            if row == 0:
                ax.set_title(ml)
            if col == 0:
                ax.set_ylabel(TARGET_DISPLAY[target])

    plt.tight_layout()
    out = output_dir / "prob_vs_det_overall.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 2: Horizon-wise RMSE & CRPS curves ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Probabilistic vs Deterministic — Per-Horizon Comparison", fontsize=14, fontweight="bold")

    for row, target in enumerate(targets):
        prob_horizon = prob_data.get("horizon_metrics", {}).get(target, [])
        det_horizon = det_data.get("horizon_metrics", {}).get(target, [])
        if not prob_horizon or not det_horizon:
            continue

        hours_p = [h["horizon_hour"] for h in prob_horizon]
        hours_d = [h["horizon_hour"] for h in det_horizon]

        # RMSE by horizon
        ax = axes[row, 0]
        ax.plot(hours_p, [h["rmse"] for h in prob_horizon],
                "o-", color=model_colors[0], linewidth=2, markersize=4, label="Probabilistic")
        ax.plot(hours_d, [h["rmse"] for h in det_horizon],
                "s--", color=model_colors[1], linewidth=2, markersize=4, label="Deterministic")
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("RMSE (°C)")
        ax.set_title(f"{TARGET_DISPLAY[target]} — RMSE")
        ax.legend()

        # CRPS by horizon
        ax = axes[row, 1]
        ax.plot(hours_p, [h["crps"] for h in prob_horizon],
                "o-", color=model_colors[0], linewidth=2, markersize=4, label="Probabilistic")
        ax.plot(hours_d, [h["crps"] for h in det_horizon],
                "s--", color=model_colors[1], linewidth=2, markersize=4, label="Deterministic")
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("CRPS")
        ax.set_title(f"{TARGET_DISPLAY[target]} — CRPS")
        ax.legend()

    plt.tight_layout()
    out = output_dir / "prob_vs_det_horizon.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 3: Uncertainty quality — predictive std & coverage ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Uncertainty Quality — Probabilistic vs Deterministic", fontsize=14, fontweight="bold")

    for tidx, target in enumerate(targets):
        ax = axes[tidx]
        prob_h = prob_data.get("horizon_metrics", {}).get(target, [])
        det_h = det_data.get("horizon_metrics", {}).get(target, [])
        if not prob_h or not det_h:
            continue

        hours = [h["horizon_hour"] for h in prob_h]

        # Predictive std
        ax.plot(hours, [h["avg_predictive_std"] for h in prob_h],
                "o-", color=model_colors[0], linewidth=2, markersize=4, label="Prob. Pred Std")
        ax.plot(hours, [h["avg_predictive_std"] for h in det_h],
                "s--", color=model_colors[1], linewidth=2, markersize=4, label="Det. Pred Std")

        # Actual RMSE as reference
        ax.plot(hours, [h["rmse"] for h in prob_h],
                "^:", color=PALETTE[4], linewidth=1.5, markersize=3, label="Actual RMSE")

        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("Std / Error (°C)")
        ax.set_title(f"{TARGET_DISPLAY[target]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "prob_vs_det_uncertainty.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 4. Architecture Comparison (GRU vs LSTM vs Transformer) ───────────────


ARCH_CONFIGS = {
    "GRU": "eval_full_qingdao.json",
    "LSTM": "eval_lstm_qingdao.json",
    "Transformer": "eval_transformer_qingdao.json",
}
ARCH_COLORS = [PALETTE[1], PALETTE[0], PALETTE[4]]  # teal, red, violet
ARCH_MARKERS = ["o", "s", "^"]
ARCH_LINESTYLES = ["-", "--", "-."]


def plot_architecture_comparison(results_dir: Path, output_dir: Path) -> None:
    """Generate comparison figures for GRU vs LSTM vs Transformer on Qingdao."""

    arch_data: dict[str, dict] = {}
    for label, fname in ARCH_CONFIGS.items():
        data = load_eval_json(results_dir / fname)
        if data:
            arch_data[label] = data
    if len(arch_data) < 2:
        print("Need at least 2 architecture eval files for comparison. Skipping.")
        return

    targets = ["heat_index_c", "wbgt_like_c"]
    arch_labels = list(arch_data.keys())

    # ── Figure 1: Overall metrics grouped bar chart ──
    metrics = ["rmse", "mae", "crps", "winkler_90", "coverage_90"]
    metric_labels = ["RMSE (°C)", "MAE (°C)", "CRPS", "Winkler@90%", "Coverage@90%"]

    fig, axes = plt.subplots(2, len(metrics), figsize=(22, 8))
    fig.suptitle("Encoder Architecture Comparison — Qingdao", fontsize=14, fontweight="bold")

    bar_width = 0.25
    for row, target in enumerate(targets):
        for col, (mk, ml) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            for i, label in enumerate(arch_labels):
                val = arch_data[label].get(target, {}).get(mk, 0)
                bar = ax.bar(i * bar_width, val, width=bar_width, color=ARCH_COLORS[i],
                             alpha=0.85, edgecolor="white", label=label if row == 0 and col == 0 else "")
                ax.text(i * bar_width, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks([i * bar_width for i in range(len(arch_labels))])
            ax.set_xticklabels(arch_labels, fontsize=9)
            if row == 0:
                ax.set_title(ml)
            if col == 0:
                ax.set_ylabel(TARGET_DISPLAY[target])
            if mk == "coverage_90":
                ax.axhline(0.9, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out = output_dir / "arch_comparison_overall.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 2: Per-horizon RMSE and CRPS ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Horizon Comparison — GRU vs LSTM vs Transformer", fontsize=14, fontweight="bold")

    for row, target in enumerate(targets):
        for col, (mk, ml) in enumerate([("rmse", "RMSE (°C)"), ("crps", "CRPS")]):
            ax = axes[row, col]
            for i, label in enumerate(arch_labels):
                horizon = arch_data[label].get("horizon_metrics", {}).get(target, [])
                if not horizon:
                    continue
                hours = [h["horizon_hour"] for h in horizon]
                values = [h[mk] for h in horizon]
                ax.plot(hours, values, marker=ARCH_MARKERS[i], linestyle=ARCH_LINESTYLES[i],
                        color=ARCH_COLORS[i], linewidth=2, markersize=4, label=label)
            ax.set_xlabel("Forecast Horizon (hours)")
            ax.set_ylabel(ml)
            ax.set_title(f"{TARGET_DISPLAY[target]} — {ml.split(' ')[0]}")
            ax.legend()

    plt.tight_layout()
    out = output_dir / "arch_comparison_horizon.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 3: Uncertainty decomposition per horizon ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Uncertainty Decomposition — GRU vs LSTM vs Transformer", fontsize=14, fontweight="bold")

    for row, target in enumerate(targets):
        # Aleatoric std
        ax = axes[row, 0]
        for i, label in enumerate(arch_labels):
            horizon = arch_data[label].get("horizon_metrics", {}).get(target, [])
            if not horizon:
                continue
            hours = [h["horizon_hour"] for h in horizon]
            ax.plot(hours, [h["avg_aleatoric_std"] for h in horizon],
                    marker=ARCH_MARKERS[i], linestyle=ARCH_LINESTYLES[i],
                    color=ARCH_COLORS[i], linewidth=2, markersize=4, label=label)
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("Aleatoric Std (°C)")
        ax.set_title(f"{TARGET_DISPLAY[target]} — Aleatoric Uncertainty")
        ax.legend()

        # Epistemic std
        ax = axes[row, 1]
        for i, label in enumerate(arch_labels):
            horizon = arch_data[label].get("horizon_metrics", {}).get(target, [])
            if not horizon:
                continue
            hours = [h["horizon_hour"] for h in horizon]
            ax.plot(hours, [h["avg_epistemic_std"] for h in horizon],
                    marker=ARCH_MARKERS[i], linestyle=ARCH_LINESTYLES[i],
                    color=ARCH_COLORS[i], linewidth=2, markersize=4, label=label)
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("Epistemic Std (°C)")
        ax.set_title(f"{TARGET_DISPLAY[target]} — Epistemic Uncertainty")
        ax.legend()

    plt.tight_layout()
    out = output_dir / "arch_comparison_uncertainty.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 4: Skill score comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Persistence Skill Score — GRU vs LSTM vs Transformer", fontsize=14, fontweight="bold")

    for tidx, target in enumerate(targets):
        ax = axes[tidx]
        for i, label in enumerate(arch_labels):
            horizon = arch_data[label].get("horizon_metrics", {}).get(target, [])
            if not horizon:
                continue
            hours = [h["horizon_hour"] for h in horizon]
            # Compute per-horizon skill from RMSE vs persistence
            persist_rmse = arch_data[label]["persistence_skill"][target]["persistence_rmse"]
            skills = [1.0 - (h["rmse"] ** 2) / (persist_rmse ** 2) for h in horizon]
            ax.plot(hours, skills, marker=ARCH_MARKERS[i], linestyle=ARCH_LINESTYLES[i],
                    color=ARCH_COLORS[i], linewidth=2, markersize=4, label=label)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("Skill Score")
        ax.set_title(TARGET_DISPLAY[target])
        ax.legend()
        ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    out = output_dir / "arch_comparison_skill.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 5: Radar chart ──
    overall_metrics = ["rmse", "mae", "crps", "coverage_90", "gaussian_nll"]
    radar_labels = ["RMSE", "MAE", "CRPS", "Cov@90%", "NLL"]
    n_metrics = len(overall_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"polar": True})
    fig.suptitle("Architecture Radar — Qingdao", fontsize=14, fontweight="bold")

    for tidx, target in enumerate(targets):
        ax = axes[tidx]
        # Collect raw values and normalize (lower is better except coverage)
        raw_vals = {label: [arch_data[label][target].get(m, 0) for m in overall_metrics]
                    for label in arch_labels}
        # Normalize each metric to [0, 1] range across architectures
        for m_idx in range(n_metrics):
            col_vals = [raw_vals[l][m_idx] for l in arch_labels]
            mn, mx = min(col_vals), max(col_vals)
            rng = mx - mn if mx != mn else 1.0
            for l in arch_labels:
                if overall_metrics[m_idx] == "coverage_90":
                    # Closer to 0.9 is better
                    raw_vals[l][m_idx] = 1.0 - abs(raw_vals[l][m_idx] - 0.9) / max(rng, 0.05)
                else:
                    # Lower is better -> invert
                    raw_vals[l][m_idx] = 1.0 - (raw_vals[l][m_idx] - mn) / rng

        for i, label in enumerate(arch_labels):
            vals = raw_vals[label] + [raw_vals[label][0]]
            ax.plot(angles, vals, marker=ARCH_MARKERS[i], linestyle=ARCH_LINESTYLES[i],
                    color=ARCH_COLORS[i], linewidth=2, markersize=5, label=label)
            ax.fill(angles, vals, color=ARCH_COLORS[i], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_title(TARGET_DISPLAY[target], pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    out = output_dir / "arch_comparison_radar.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 5. Seasonal Performance Analysis ─────────────────────────────────────


def plot_seasonal_analysis(results_dir: Path, output_dir: Path) -> None:
    """Analyze and plot model performance broken down by season.

    This requires re-running inference with seasonal splits, so we read the
    per-horizon table CSVs and the raw data timestamps to assign seasons.
    We use a simpler approach: load the eval JSON which has horizon_metrics,
    and correlate with the test set timestamps from the checkpoint.
    """

    import csv
    import torch
    from train import (
        TrainConfig, load_era5_table, build_feature_target_arrays,
        split_boundaries, build_sample_indices, SequenceDataset,
        build_model, predict_distribution, set_seed, TARGET_NAMES, EPS,
    )
    from torch.utils.data import DataLoader

    sites = ["qingdao", "dubai", "singapore", "miami"]
    targets = list(TARGET_NAMES)

    for site in sites:
        cp_path = Path(f"checkpoints/full_{site}.pt")
        data_path = Path(f"data/era5_{site}.csv")
        if not cp_path.exists() or not data_path.exists():
            continue

        set_seed(42)
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        config = TrainConfig(**checkpoint["config"])
        table = load_era5_table(str(data_path))
        features, targets_arr, feature_names, timestamps = build_feature_target_arrays(table)

        feature_mean = checkpoint["feature_mean"]
        feature_std = checkpoint["feature_std"]
        target_mean = checkpoint["target_mean"]
        target_std = checkpoint["target_std"]

        features_norm = ((features - feature_mean) / feature_std).astype(np.float32)
        targets_norm = ((targets_arr - target_mean) / target_std).astype(np.float32)

        boundaries = split_boundaries(len(features_norm), config.train_ratio, config.val_ratio)
        test_indices = build_sample_indices(
            start=boundaries["test"][0], end=boundaries["test"][1],
            lookback=config.lookback, horizon=config.horizon, max_samples=0,
        )
        test_dataset = SequenceDataset(
            features=features_norm, targets=targets_norm,
            indices=test_indices, lookback=config.lookback, horizon=config.horizon,
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        model = build_model(config, len(feature_names))
        model.load_state_dict(checkpoint["model_state_dict"])

        pred_mean, total_var, aleatoric_var, targets_true = predict_distribution(
            model=model, loader=test_loader, device=torch.device("cpu"),
            target_mean=target_mean, target_std=target_std, mc_samples=30,
        )

        # Get month for each test sample (based on the start of the forecast window)
        sample_months = np.array([timestamps[idx].month for idx in test_indices])

        # Define seasons
        season_map = {
            "Spring (MAM)": [3, 4, 5],
            "Summer (JJA)": [6, 7, 8],
            "Autumn (SON)": [9, 10, 11],
            "Winter (DJF)": [12, 1, 2],
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Seasonal Performance — {SITE_DISPLAY.get(site, site)}", fontsize=14, fontweight="bold")

        for tidx, target_name in enumerate(TARGET_NAMES):
            # RMSE by season
            ax_rmse = axes[tidx, 0]
            # Coverage by season
            ax_cov = axes[tidx, 1]

            season_names = []
            rmse_vals = []
            mae_vals = []
            cov_vals = []
            n_samples = []

            z_90 = 1.6448536269514722
            for season_name, months in season_map.items():
                mask = np.isin(sample_months, months)
                if mask.sum() == 0:
                    continue
                pred_s = pred_mean[mask, :, tidx]
                true_s = targets_true[mask, :, tidx]
                var_s = np.maximum(total_var[mask, :, tidx], EPS)
                std_s = np.sqrt(var_s)

                rmse = float(np.sqrt(np.mean((pred_s - true_s) ** 2)))
                mae = float(np.mean(np.abs(pred_s - true_s)))
                lower = pred_s - z_90 * std_s
                upper = pred_s + z_90 * std_s
                cov = float(np.mean((true_s >= lower) & (true_s <= upper)))

                season_names.append(season_name)
                rmse_vals.append(rmse)
                mae_vals.append(mae)
                cov_vals.append(cov)
                n_samples.append(int(mask.sum()))

            x = np.arange(len(season_names))

            # RMSE + MAE grouped bars
            w = 0.35
            bars1 = ax_rmse.bar(x - w / 2, rmse_vals, w, color=PALETTE[0], alpha=0.85, label="RMSE")
            bars2 = ax_rmse.bar(x + w / 2, mae_vals, w, color=PALETTE[1], alpha=0.85, label="MAE")
            for bar, val in zip(bars1, rmse_vals):
                ax_rmse.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            for bar, val in zip(bars2, mae_vals):
                ax_rmse.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax_rmse.set_xticks(x)
            ax_rmse.set_xticklabels(season_names, fontsize=8)
            ax_rmse.set_ylabel("Error (°C)")
            ax_rmse.set_title(f"{TARGET_DISPLAY[target_name]} — Error by Season")
            ax_rmse.legend()

            # Coverage bar
            bars_cov = ax_cov.bar(x, cov_vals, width=0.5, color=PALETTE[2], alpha=0.85)
            ax_cov.axhline(y=0.9, color="red", linestyle="--", linewidth=1, label="Target 90%")
            for bar, val, ns in zip(bars_cov, cov_vals, n_samples):
                ax_cov.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}\n(n={ns})", ha="center", va="bottom", fontsize=7)
            ax_cov.set_xticks(x)
            ax_cov.set_xticklabels(season_names, fontsize=8)
            ax_cov.set_ylabel("Coverage")
            ax_cov.set_title(f"{TARGET_DISPLAY[target_name]} — 90% PI Coverage by Season")
            ax_cov.set_ylim(0.7, 1.0)
            ax_cov.legend()

        plt.tight_layout()
        out = output_dir / f"seasonal_analysis_{site}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")


# ── 5. Feature Importance (Permutation) ─────────────────────────────────────


def plot_feature_importance(results_dir: Path, output_dir: Path) -> None:
    """Compute and plot permutation feature importance for the full Qingdao model."""

    import torch
    from train import (
        TrainConfig, load_era5_table, build_feature_target_arrays,
        split_boundaries, build_sample_indices, SequenceDataset,
        build_model, set_seed, TARGET_NAMES, EPS,
    )
    from torch.utils.data import DataLoader

    cp_path = Path("checkpoints/full_qingdao.pt")
    data_path = Path("data/era5_qingdao.csv")
    if not cp_path.exists() or not data_path.exists():
        print("Qingdao checkpoint/data not found for feature importance.")
        return

    set_seed(42)
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    table = load_era5_table(str(data_path))
    features, targets_arr, feature_names, timestamps = build_feature_target_arrays(table)

    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]

    features_norm = ((features - feature_mean) / feature_std).astype(np.float32)
    targets_norm = ((targets_arr - target_mean) / target_std).astype(np.float32)

    boundaries = split_boundaries(len(features_norm), config.train_ratio, config.val_ratio)
    test_indices = build_sample_indices(
        start=boundaries["test"][0], end=boundaries["test"][1],
        lookback=config.lookback, horizon=config.horizon, max_samples=2000,
    )
    test_dataset = SequenceDataset(
        features=features_norm, targets=targets_norm,
        indices=test_indices, lookback=config.lookback, horizon=config.horizon,
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = build_model(config, len(feature_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Baseline MSE
    def compute_mse(loader: DataLoader) -> np.ndarray:
        """Return per-target MSE."""
        all_mse = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                mean, _ = model(x_batch)
                mean_phys = mean.numpy() * target_std.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
                true_phys = y_batch.numpy() * target_std.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
                all_mse.append(np.mean((mean_phys - true_phys) ** 2, axis=(0, 1)))
        return np.mean(np.stack(all_mse), axis=0)

    baseline_mse = compute_mse(test_loader)

    # Permutation importance: shuffle each feature across the time dimension
    importance = {target: [] for target in TARGET_NAMES}
    n_features = len(feature_names)
    print(f"Computing permutation importance for {n_features} features...")

    for feat_idx in range(n_features):
        # Create a copy of features with this feature shuffled
        features_perm = features_norm.copy()
        rng = np.random.RandomState(42 + feat_idx)
        perm_idx = rng.permutation(len(features_perm))
        features_perm[:, feat_idx] = features_perm[perm_idx, feat_idx]

        perm_dataset = SequenceDataset(
            features=features_perm, targets=targets_norm,
            indices=test_indices, lookback=config.lookback, horizon=config.horizon,
        )
        perm_loader = DataLoader(perm_dataset, batch_size=256, shuffle=False)
        perm_mse = compute_mse(perm_loader)

        for tidx, target in enumerate(TARGET_NAMES):
            # Importance = increase in MSE when feature is permuted
            importance[target].append(float(perm_mse[tidx] - baseline_mse[tidx]))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Permutation Feature Importance — Qingdao (Full Model)", fontsize=14, fontweight="bold")

    for tidx, target in enumerate(TARGET_NAMES):
        ax = axes[tidx]
        imp = np.array(importance[target])
        sorted_idx = np.argsort(imp)[::-1]
        top_k = min(15, len(sorted_idx))
        top_idx = sorted_idx[:top_k]

        y_pos = np.arange(top_k)
        ax.barh(y_pos, imp[top_idx], color=PALETTE[tidx], alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ MSE (increase when feature permuted)")
        ax.set_title(f"{TARGET_DISPLAY[target]}")

        # Add value labels
        for pos, idx in zip(y_pos, top_idx):
            ax.text(imp[idx], pos, f" {imp[idx]:.4f}", va="center", fontsize=7)

    plt.tight_layout()
    out = output_dir / "feature_importance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")

    # Save CSV
    csv_path = output_dir / "feature_importance.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["feature"] + [f"importance_{t}" for t in TARGET_NAMES])
        for i, fname in enumerate(feature_names):
            writer.writerow([fname] + [f"{importance[t][i]:.6f}" for t in TARGET_NAMES])
    print(f"Saved: {csv_path}")


# ── 6. Uncertainty Calibration Summary ──────────────────────────────────────


def plot_uncertainty_summary(results_dir: Path, output_dir: Path) -> None:
    """Summary plot of aleatoric vs epistemic uncertainty across all sites."""

    sites = ["qingdao", "dubai", "singapore", "miami"]
    targets = ["heat_index_c", "wbgt_like_c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Uncertainty Decomposition Across Sites", fontsize=14, fontweight="bold")

    for tidx, target in enumerate(targets):
        ax = axes[tidx]
        aleatoric_vals = []
        epistemic_vals = []
        site_labels = []

        for site in sites:
            path = results_dir / f"eval_full_{site}.json"
            data = load_eval_json(path)
            if data is None:
                continue
            ub = data.get("uncertainty_breakdown", {}).get(target, {})
            aleatoric_vals.append(ub.get("mean_aleatoric_std", 0))
            epistemic_vals.append(ub.get("mean_epistemic_std", 0))
            site_labels.append(SITE_DISPLAY.get(site, site))

        x = np.arange(len(site_labels))
        w = 0.35
        ax.bar(x - w / 2, aleatoric_vals, w, color=PALETTE[3], alpha=0.85, label="Aleatoric")
        ax.bar(x + w / 2, epistemic_vals, w, color=PALETTE[4], alpha=0.85, label="Epistemic")
        ax.set_xticks(x)
        ax.set_xticklabels(site_labels)
        ax.set_ylabel("Std Dev (°C)")
        ax.set_title(TARGET_DISPLAY[target])
        ax.legend()

    plt.tight_layout()
    out = output_dir / "uncertainty_decomposition_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-experiment comparison figures.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--skip-seasonal", action="store_true", help="Skip seasonal analysis (slow)")
    parser.add_argument("--skip-importance", action="store_true", help="Skip feature importance (slow)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating cross-experiment comparison figures")
    print("=" * 60)

    plot_training_curves(results_dir, output_dir)
    plot_cross_site_comparison(results_dir, output_dir)
    plot_ablation_comparison(results_dir, output_dir)
    plot_prob_vs_det(results_dir, output_dir)
    plot_architecture_comparison(results_dir, output_dir)
    plot_uncertainty_summary(results_dir, output_dir)

    if not args.skip_seasonal:
        print("\n--- Seasonal Analysis (requires model inference) ---")
        plot_seasonal_analysis(results_dir, output_dir)

    if not args.skip_importance:
        print("\n--- Feature Importance (requires model inference) ---")
        plot_feature_importance(results_dir, output_dir)

    print("\n" + "=" * 60)
    print("All figures generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
