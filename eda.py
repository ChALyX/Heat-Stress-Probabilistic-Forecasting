"""
Exploratory Data Analysis (EDA) for ERA5 coastal heat-stress data.

Generates publication-quality figures covering:
1. Data quality: missing values, outliers
2. Feature distributions across sites
3. Correlation analysis
4. Target variable (Heat Index, WBGT-like) distributions
5. Temporal patterns: diurnal cycles, seasonal trends, autocorrelation
6. Cross-site comparison

Usage:
    python eda.py                        # analyze all sites
    python eda.py --site qingdao dubai   # analyze selected sites
    python eda.py --output-dir results   # specify output directory
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────────
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

PALETTE = ["#E63946", "#2A9D8F", "#E9A820", "#264653", "#8B5CF6", "#F97316"]
SITE_COLORS = {
    "qingdao": "#E63946",
    "dubai": "#2A9D8F",
    "singapore": "#E9A820",
    "miami": "#264653",
}
SITE_DISPLAY = {
    "qingdao": "Qingdao",
    "dubai": "Dubai",
    "singapore": "Singapore",
    "miami": "Miami",
}

RAW_COLUMNS = [
    "u100", "v100", "u10", "v10", "fg10", "d2m", "t2m",
    "msl", "sst", "skt", "sp", "ssrd", "strd", "tp",
]

COLUMN_LABELS = {
    "u100": "100m U-wind (m/s)",
    "v100": "100m V-wind (m/s)",
    "u10": "10m U-wind (m/s)",
    "v10": "10m V-wind (m/s)",
    "fg10": "10m Gust (m/s)",
    "d2m": "2m Dewpoint (K)",
    "t2m": "2m Temperature (K)",
    "msl": "Mean Sea Level Pressure (Pa)",
    "sst": "Sea Surface Temp (K)",
    "skt": "Skin Temperature (K)",
    "sp": "Surface Pressure (Pa)",
    "ssrd": "Solar Radiation Down (J/m²)",
    "strd": "Thermal Radiation Down (J/m²)",
    "tp": "Total Precipitation (m)",
}


# ── Helper functions (matching train.py) ─────────────────────────────────────

def saturation_vapour_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def relative_humidity_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    e_t = saturation_vapour_pressure_hpa(temp_c)
    e_td = saturation_vapour_pressure_hpa(dewpoint_c)
    rh = 100.0 * (e_td / np.maximum(e_t, 1e-6))
    return np.clip(rh, 1.0, 100.0)


def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    hi_simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
    hi_full = (
        -42.379 + 2.04901523 * temp_f + 10.14333127 * rh
        - 0.22475541 * temp_f * rh - 0.00683783 * temp_f**2
        - 0.05481717 * rh**2 + 0.00122874 * temp_f**2 * rh
        + 0.00085282 * temp_f * rh**2 - 0.00000199 * temp_f**2 * rh**2
    )
    adj_low = ((13.0 - rh) / 4.0) * np.sqrt(np.maximum(0.0, (17.0 - np.abs(temp_f - 95.0)) / 17.0))
    adj_high = ((rh - 85.0) / 10.0) * ((87.0 - temp_f) / 5.0)
    hi_full = np.where((rh < 13.0) & (80.0 <= temp_f) & (temp_f <= 112.0), hi_full - adj_low, hi_full)
    hi_full = np.where((rh > 85.0) & (80.0 <= temp_f) & (temp_f <= 87.0), hi_full + adj_high, hi_full)
    hi_f = np.where(hi_simple < 80.0, hi_simple, hi_full)
    return (hi_f - 32.0) * 5.0 / 9.0


def wet_bulb_stull_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    rh = np.clip(rh, 1.0, 100.0)
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh) - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035
    )


def wbgt_like_celsius(temp_c, rh, wind_speed, solar_wm2, thermal_wm2):
    wet_bulb = wet_bulb_stull_celsius(temp_c, rh)
    globe_like = temp_c + 0.0025 * solar_wm2 + 0.0012 * thermal_wm2 - 0.35 * wind_speed
    wbgt = 0.7 * wet_bulb + 0.2 * globe_like + 0.1 * temp_c
    return np.maximum(wbgt, wet_bulb)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_site(site_name: str, data_dir: str = "data") -> pd.DataFrame:
    """Load ERA5 CSV for a site into a pandas DataFrame with derived columns."""
    path = Path(data_dir) / f"era5_{site_name}.csv"
    df = pd.read_csv(path, parse_dates=["valid_time"])
    df["site"] = site_name

    # Derived features (matching train.py)
    df["t2m_c"] = df["t2m"] - 273.15
    df["d2m_c"] = df["d2m"] - 273.15
    df["skt_c"] = df["skt"] - 273.15
    df["sst_c"] = df["sst"] - 273.15
    df["rh"] = relative_humidity_from_dewpoint(df["t2m_c"].values, df["d2m_c"].values)
    df["wind10"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
    df["wind100"] = np.sqrt(df["u100"]**2 + df["v100"]**2)
    df["ssrd_wm2"] = np.clip(df["ssrd"] / 3600.0, 0, None)
    df["strd_wm2"] = np.clip(df["strd"] / 3600.0, 0, None)
    df["tp_mm"] = df["tp"] * 1000.0

    df["heat_index_c"] = heat_index_celsius(df["t2m_c"].values, df["rh"].values)
    df["wbgt_like_c"] = wbgt_like_celsius(
        df["t2m_c"].values, df["rh"].values, df["wind10"].values,
        df["ssrd_wm2"].values, df["strd_wm2"].values,
    )

    df["hour"] = df["valid_time"].dt.hour
    df["month"] = df["valid_time"].dt.month
    df["year"] = df["valid_time"].dt.year
    df["day_of_year"] = df["valid_time"].dt.dayofyear
    return df


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_missing_values(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Bar chart of missing-value percentages per column per site."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sites = list(dfs.keys())
    cols = RAW_COLUMNS
    x = np.arange(len(cols))
    width = 0.8 / len(sites)

    for i, site in enumerate(sites):
        df = dfs[site]
        pcts = [(df[c].isna().sum() / len(df)) * 100 for c in cols]
        ax.bar(x + i * width, pcts, width, label=SITE_DISPLAY[site],
               color=SITE_COLORS[site], alpha=0.85)

    ax.set_xticks(x + width * (len(sites) - 1) / 2)
    ax.set_xticklabels([COLUMN_LABELS.get(c, c) for c in cols], rotation=45, ha="right")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Value Percentage by Feature and Site")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_missing_values.png")
    plt.close(fig)
    print("  [+] eda_missing_values.png")


def plot_basic_stats_table(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save a CSV summary of basic statistics per site."""
    records = []
    for site, df in dfs.items():
        for col in RAW_COLUMNS:
            s = df[col].dropna()
            records.append({
                "site": site,
                "feature": col,
                "count": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "25%": s.quantile(0.25),
                "50%": s.quantile(0.50),
                "75%": s.quantile(0.75),
                "max": s.max(),
                "skewness": s.skew(),
                "kurtosis": s.kurtosis(),
            })
    stats_df = pd.DataFrame(records)
    stats_df.to_csv(output_dir / "eda_basic_stats.csv", index=False, float_format="%.4f")
    print("  [+] eda_basic_stats.csv")


def plot_feature_distributions(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """KDE overlay of key features across sites."""
    features = ["t2m_c", "d2m_c", "rh", "wind10", "ssrd_wm2", "tp_mm"]
    labels = {
        "t2m_c": "2m Temperature (°C)",
        "d2m_c": "2m Dewpoint (°C)",
        "rh": "Relative Humidity (%)",
        "wind10": "10m Wind Speed (m/s)",
        "ssrd_wm2": "Solar Radiation (W/m²)",
        "tp_mm": "Precipitation (mm)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flat, features):
        for site, df in dfs.items():
            vals = df[feat].dropna()
            ax.hist(vals, bins=80, density=True, alpha=0.3, color=SITE_COLORS[site])
            # KDE
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(vals)
                xg = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xg, kde(xg), color=SITE_COLORS[site], lw=1.5, label=SITE_DISPLAY[site])
            except Exception:
                pass
        ax.set_title(labels.get(feat, feat))
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.suptitle("Feature Distributions Across Sites", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_feature_distributions.png")
    plt.close(fig)
    print("  [+] eda_feature_distributions.png")


def plot_correlation_matrix(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Correlation heatmap of engineered features for each site."""
    eng_features = [
        "t2m_c", "d2m_c", "rh", "skt_c", "sst_c", "wind10", "wind100",
        "ssrd_wm2", "strd_wm2", "tp_mm", "heat_index_c", "wbgt_like_c",
    ]
    eng_labels = [
        "T2m", "Dewpoint", "RH", "Skin T", "SST", "Wind10",
        "Wind100", "Solar↓", "Thermal↓", "Precip", "Heat Idx", "WBGT",
    ]

    n_sites = len(dfs)
    fig, axes = plt.subplots(1, n_sites, figsize=(5 * n_sites, 5))
    if n_sites == 1:
        axes = [axes]

    for ax, (site, df) in zip(axes, dfs.items()):
        corr = df[eng_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6},
            xticklabels=eng_labels, yticklabels=eng_labels,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{SITE_DISPLAY[site]}", fontsize=12)
    fig.suptitle("Feature Correlation Matrix", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_correlation_matrix.png")
    plt.close(fig)
    print("  [+] eda_correlation_matrix.png")


def plot_target_distributions(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Distribution of the two target variables across sites."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets = [("heat_index_c", "Heat Index (°C)"), ("wbgt_like_c", "WBGT-like (°C)")]

    for ax, (col, label) in zip(axes, targets):
        for site, df in dfs.items():
            vals = df[col].dropna()
            ax.hist(vals, bins=80, density=True, alpha=0.35, color=SITE_COLORS[site],
                    label=SITE_DISPLAY[site])
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution")
        ax.legend()

    fig.suptitle("Target Variable Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_target_distributions.png")
    plt.close(fig)
    print("  [+] eda_target_distributions.png")


def plot_target_boxplot(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Monthly box plots of targets to show seasonal patterns."""
    combined = pd.concat(dfs.values(), ignore_index=True)
    targets = [("heat_index_c", "Heat Index (°C)"), ("wbgt_like_c", "WBGT-like (°C)")]

    fig, axes = plt.subplots(len(targets), 1, figsize=(14, 5 * len(targets)))
    for ax, (col, label) in zip(axes, targets):
        data_to_plot = []
        positions = []
        colors_list = []
        sites = list(dfs.keys())
        n_sites = len(sites)

        for month in range(1, 13):
            for j, site in enumerate(sites):
                df = dfs[site]
                vals = df.loc[df["month"] == month, col].dropna().values
                data_to_plot.append(vals)
                positions.append(month * (n_sites + 1) + j)
                colors_list.append(SITE_COLORS[site])

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1))
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # X-axis labels
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        tick_pos = [m * (n_sites + 1) + (n_sites - 1) / 2 for m in range(1, 13)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(month_names)
        ax.set_ylabel(label)
        ax.set_title(f"Monthly {label} by Site")

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, fc=SITE_COLORS[s], alpha=0.7) for s in sites]
        ax.legend(handles, [SITE_DISPLAY[s] for s in sites], loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "eda_seasonal_boxplot.png")
    plt.close(fig)
    print("  [+] eda_seasonal_boxplot.png")


def plot_diurnal_cycle(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Average diurnal cycle for key variables."""
    features = [
        ("t2m_c", "Temperature (°C)"),
        ("rh", "Relative Humidity (%)"),
        ("ssrd_wm2", "Solar Radiation (W/m²)"),
        ("heat_index_c", "Heat Index (°C)"),
        ("wbgt_like_c", "WBGT-like (°C)"),
        ("wind10", "Wind Speed (m/s)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (feat, label) in zip(axes.flat, features):
        for site, df in dfs.items():
            hourly = df.groupby("hour")[feat].agg(["mean", "std"])
            ax.plot(hourly.index, hourly["mean"], color=SITE_COLORS[site],
                    lw=1.5, label=SITE_DISPLAY[site])
            ax.fill_between(hourly.index,
                            hourly["mean"] - hourly["std"],
                            hourly["mean"] + hourly["std"],
                            color=SITE_COLORS[site], alpha=0.12)
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel(label)
        ax.set_title(f"Diurnal Cycle: {label}")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=7)

    fig.suptitle("Average Diurnal Cycles (mean ± 1σ)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_diurnal_cycle.png")
    plt.close(fig)
    print("  [+] eda_diurnal_cycle.png")


def plot_annual_trend(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Monthly mean time series to show annual and inter-annual trends."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    targets = [("heat_index_c", "Heat Index (°C)"), ("wbgt_like_c", "WBGT-like (°C)")]

    for ax, (col, label) in zip(axes, targets):
        for site, df in dfs.items():
            monthly = df.set_index("valid_time").resample("ME")[col].mean()
            ax.plot(monthly.index, monthly.values, color=SITE_COLORS[site],
                    lw=1, alpha=0.8, label=SITE_DISPLAY[site])
        ax.set_ylabel(label)
        ax.set_title(f"Monthly Mean {label}")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Inter-annual Trends of Target Variables", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_annual_trend.png")
    plt.close(fig)
    print("  [+] eda_annual_trend.png")


def plot_autocorrelation(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Autocorrelation of target variables up to 168h (1 week)."""
    max_lag = 168
    targets = [("heat_index_c", "Heat Index"), ("wbgt_like_c", "WBGT-like")]

    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 4 * len(targets)))
    for ax, (col, label) in zip(axes, targets):
        for site, df in dfs.items():
            series = df[col].dropna().values
            n = len(series)
            mean = series.mean()
            var = np.var(series)
            acf = np.array([
                np.mean((series[:n - lag] - mean) * (series[lag:] - mean)) / var
                for lag in range(max_lag + 1)
            ])
            ax.plot(range(max_lag + 1), acf, color=SITE_COLORS[site],
                    lw=1.2, label=SITE_DISPLAY[site])

        # Mark lookback window (72h) and forecast horizon (24h)
        ax.axvline(72, color="gray", ls="--", lw=1, alpha=0.7, label="Lookback (72h)")
        ax.axvline(24, color="gray", ls=":", lw=1, alpha=0.7, label="Horizon (24h)")
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation: {label}")
        ax.legend(fontsize=7, ncol=3)
        ax.set_xlim(0, max_lag)

    fig.tight_layout()
    fig.savefig(output_dir / "eda_autocorrelation.png")
    plt.close(fig)
    print("  [+] eda_autocorrelation.png")


def plot_feature_target_scatter(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Scatter plots of key features vs target variables (sampled for speed)."""
    features = [("t2m_c", "Temperature (°C)"), ("rh", "RH (%)"),
                ("ssrd_wm2", "Solar Rad (W/m²)"), ("wind10", "Wind (m/s)")]
    targets = [("heat_index_c", "Heat Index (°C)"), ("wbgt_like_c", "WBGT-like (°C)")]

    fig, axes = plt.subplots(len(targets), len(features), figsize=(4 * len(features), 4 * len(targets)))

    for i, (tcol, tlabel) in enumerate(targets):
        for j, (fcol, flabel) in enumerate(features):
            ax = axes[i, j]
            for site, df in dfs.items():
                sample = df.sample(min(3000, len(df)), random_state=42)
                ax.scatter(sample[fcol], sample[tcol], s=1, alpha=0.2,
                           color=SITE_COLORS[site], label=SITE_DISPLAY[site])
            ax.set_xlabel(flabel)
            if j == 0:
                ax.set_ylabel(tlabel)
            if i == 0:
                ax.set_title(flabel)
            if i == 0 and j == len(features) - 1:
                ax.legend(fontsize=6, markerscale=5)

    fig.suptitle("Feature vs Target Scatter Plots", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_feature_target_scatter.png")
    plt.close(fig)
    print("  [+] eda_feature_target_scatter.png")


def plot_cross_site_summary(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Radar chart comparing site characteristics."""
    metrics = ["t2m_c", "rh", "wind10", "ssrd_wm2", "heat_index_c", "wbgt_like_c"]
    metric_labels = ["Temp", "RH", "Wind", "Solar", "Heat Idx", "WBGT"]

    # Normalize each metric to [0, 1] across sites
    site_means = {}
    for site, df in dfs.items():
        site_means[site] = [df[m].mean() for m in metrics]

    all_vals = np.array(list(site_means.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for site, vals in site_means.items():
        normed = [(v - mn) / r for v, mn, r in zip(vals, mins, ranges)]
        normed += normed[:1]
        ax.plot(angles, normed, "o-", color=SITE_COLORS[site], lw=1.5,
                label=SITE_DISPLAY[site], markersize=4)
        ax.fill(angles, normed, color=SITE_COLORS[site], alpha=0.08)

    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-Site Climate Profile Comparison", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "eda_cross_site_radar.png")
    plt.close(fig)
    print("  [+] eda_cross_site_radar.png")


def plot_stationarity_check(dfs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Rolling mean and std to visually check stationarity of targets."""
    window = 24 * 30  # 30-day rolling window

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    targets = [("heat_index_c", "Heat Index"), ("wbgt_like_c", "WBGT-like")]

    for col_idx, (tcol, tlabel) in enumerate(targets):
        for site, df in dfs.items():
            ts = df.set_index("valid_time")[tcol]
            roll_mean = ts.rolling(window, min_periods=window // 2).mean()
            roll_std = ts.rolling(window, min_periods=window // 2).std()

            axes[0, col_idx].plot(roll_mean.index, roll_mean.values,
                                  color=SITE_COLORS[site], lw=0.8,
                                  label=SITE_DISPLAY[site])
            axes[1, col_idx].plot(roll_std.index, roll_std.values,
                                  color=SITE_COLORS[site], lw=0.8,
                                  label=SITE_DISPLAY[site])

        axes[0, col_idx].set_title(f"{tlabel} — 30-day Rolling Mean")
        axes[0, col_idx].set_ylabel("Mean (°C)")
        axes[0, col_idx].legend(fontsize=7)
        axes[1, col_idx].set_title(f"{tlabel} — 30-day Rolling Std")
        axes[1, col_idx].set_ylabel("Std (°C)")
        axes[1, col_idx].legend(fontsize=7)

    fig.suptitle("Stationarity Check (30-day Rolling Statistics)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_stationarity_check.png")
    plt.close(fig)
    print("  [+] eda_stationarity_check.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for ERA5 heat-stress data.")
    parser.add_argument("--site", nargs="+", default=list(SITE_COLORS.keys()),
                        choices=list(SITE_COLORS.keys()), help="Sites to analyze.")
    parser.add_argument("--data-dir", default="data", help="Data directory.")
    parser.add_argument("--output-dir", default="results", help="Output directory for figures.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    dfs: Dict[str, pd.DataFrame] = {}
    for site in args.site:
        print(f"  Loading {site}...")
        dfs[site] = load_site(site, args.data_dir)
        print(f"    {len(dfs[site]):,} rows, {dfs[site]['valid_time'].min()} → {dfs[site]['valid_time'].max()}")

    print(f"\nGenerating EDA figures → {output_dir}/")

    plot_missing_values(dfs, output_dir)
    plot_basic_stats_table(dfs, output_dir)
    plot_feature_distributions(dfs, output_dir)
    plot_correlation_matrix(dfs, output_dir)
    plot_target_distributions(dfs, output_dir)
    plot_target_boxplot(dfs, output_dir)
    plot_diurnal_cycle(dfs, output_dir)
    plot_annual_trend(dfs, output_dir)
    plot_autocorrelation(dfs, output_dir)
    plot_feature_target_scatter(dfs, output_dir)
    plot_cross_site_summary(dfs, output_dir)
    plot_stationarity_check(dfs, output_dir)

    print("\nEDA complete!")


if __name__ == "__main__":
    main()
