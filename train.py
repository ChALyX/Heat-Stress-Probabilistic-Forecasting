"""
Training script for probabilistic coastal heat-stress forecasting.

This script supports multiple coastal sites using ERA5 single-point hourly
reanalysis data. It builds two biologically meaningful targets:

1. Heat Index (HI): temperature-humidity discomfort indicator.
2. WBGT-like proxy: a radiation- and wind-adjusted wet-bulb formulation that
   approximates outdoor heat stress when full black-globe measurements are not
   available in ERA5.

The model is intentionally probabilistic rather than deterministic. For every
future hour and every target, it predicts a Gaussian mean and variance. During
evaluation we can additionally enable Monte-Carlo dropout to estimate epistemic
uncertainty on top of the aleatoric uncertainty learned by the model.

Ablation flags allow disabling individual components for systematic evaluation:
  --no-attention       Disable temporal attention (use last hidden state only)
  --no-horizon-embed   Disable horizon embedding
  --deterministic      Train a deterministic baseline (MSE only, no variance)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


EPS = 1e-6
TARGET_NAMES = ("heat_index_c", "wbgt_like_c")

SITES: Dict[str, Dict[str, float]] = {
    "qingdao": {"latitude": 36.25, "longitude": 120.5},
    "dubai": {"latitude": 25.25, "longitude": 55.25},
    "singapore": {"latitude": 1.25, "longitude": 103.75},
    "miami": {"latitude": 25.75, "longitude": -80.25},
}


@dataclass
class TrainConfig:
    """Container for experiment hyperparameters."""

    data_path: str = "data/era5_qingdao.csv"
    checkpoint_path: str = "checkpoints/full_qingdao.pt"
    metrics_path: str = "results/full_qingdao_train.json"
    site: str = "qingdao"
    lookback: int = 72
    horizon: int = 24
    batch_size: int = 256
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    feature_proj_size: int = 64
    decoder_hidden_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 60
    patience: int = 10
    grad_clip: float = 1.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    mc_dropout_samples: int = 30
    max_samples_per_split: int = 0
    device: str = "cpu"
    no_attention: bool = False
    no_horizon_embed: bool = False
    deterministic: bool = False
    num_attention_heads: int = 4
    horizon_weight_ratio: float = 2.0


def set_seed(seed: int) -> None:
    """Make training reproducible across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def saturation_vapour_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    """Return saturation vapour pressure in hPa using the Magnus formula."""

    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def relative_humidity_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Estimate relative humidity (%) from air temperature and dew point."""

    e_t = saturation_vapour_pressure_hpa(temp_c)
    e_td = saturation_vapour_pressure_hpa(dewpoint_c)
    rh = 100.0 * (e_td / np.maximum(e_t, EPS))
    return np.clip(rh, 1.0, 100.0)


def heat_index_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Compute the NOAA heat index in Celsius.

    Follows the full NOAA algorithm:
    1. Compute the Steadman simple formula as a first estimate.
    2. If the simple estimate < 80 F, use it directly.
    3. Otherwise apply the Rothfusz regression with low-RH and high-RH adjustments.
    """

    temp_f = temp_c * 9.0 / 5.0 + 32.0

    # Step 1: Steadman simple formula (NOAA initial estimate).
    hi_simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)

    # Step 2: Rothfusz regression for the full heat-index calculation.
    hi_full = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * temp_f**2 * rh
        + 0.00085282 * temp_f * rh**2
        - 0.00000199 * temp_f**2 * rh**2
    )

    # Step 3: Low-RH and high-RH adjustments (applied only to full regression).
    adjustment_low = ((13.0 - rh) / 4.0) * np.sqrt(np.maximum(0.0, (17.0 - np.abs(temp_f - 95.0)) / 17.0))
    adjustment_high = ((rh - 85.0) / 10.0) * ((87.0 - temp_f) / 5.0)
    hi_full = np.where((rh < 13.0) & (80.0 <= temp_f) & (temp_f <= 112.0), hi_full - adjustment_low, hi_full)
    hi_full = np.where((rh > 85.0) & (80.0 <= temp_f) & (temp_f <= 87.0), hi_full + adjustment_high, hi_full)

    # Step 4: Use simple formula when its estimate < 80 F, otherwise use full regression.
    hi_f = np.where(hi_simple < 80.0, hi_simple, hi_full)
    return (hi_f - 32.0) * 5.0 / 9.0


def wet_bulb_stull_celsius(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Approximate wet-bulb temperature with the Stull (2011) closed form."""

    rh = np.clip(rh, 1.0, 100.0)
    return (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def wbgt_like_celsius(
    temp_c: np.ndarray,
    rh: np.ndarray,
    wind_speed: np.ndarray,
    solar_flux_wm2: np.ndarray,
    thermal_flux_wm2: np.ndarray,
) -> np.ndarray:
    """
    Build a WBGT-like proxy from ERA5 variables.

    This is not a strict ISO WBGT reconstruction because ERA5 does not directly
    provide black-globe temperature or direct solar geometry here. Instead, we
    form a physically motivated proxy that combines wet-bulb temperature,
    radiation loading, and wind-driven cooling.
    """

    wet_bulb = wet_bulb_stull_celsius(temp_c, rh)
    globe_like = temp_c + 0.0025 * solar_flux_wm2 + 0.0012 * thermal_flux_wm2 - 0.35 * wind_speed
    wbgt = 0.7 * wet_bulb + 0.2 * globe_like + 0.1 * temp_c
    return np.maximum(wbgt, wet_bulb)


def parse_timestamp(value: str) -> datetime:
    """Parse ERA5 timestamp formats (both old and new CDS API styles)."""

    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {value!r}")


def load_era5_table(csv_path: str) -> Dict[str, np.ndarray]:
    """Load the ERA5 CSV file into NumPy arrays."""

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows were found in {csv_path}.")

    timestamps = [parse_timestamp(row["valid_time"]) for row in rows]
    skip_keys = {"valid_time", "latitude", "longitude"}
    raw_keys = [key for key in rows[0].keys() if key not in skip_keys]
    # Process skt before sst so skt is available as a fallback for missing sst.
    numeric_keys = sorted(raw_keys, key=lambda k: (k == "sst", k))
    table: Dict[str, np.ndarray] = {
        "timestamp": np.array(timestamps),
    }
    for key in numeric_keys:
        values = []
        for row in rows:
            val = row[key]
            values.append(float(val) if val != "" else float("nan"))
        arr = np.array(values, dtype=np.float32)
        if np.isnan(arr).any():
            nan_pct = np.isnan(arr).mean() * 100
            if nan_pct > 50:
                # Column is mostly missing (e.g. SST over land) — fill with
                # a physically reasonable fallback: skin temperature if
                # available, otherwise column mean.
                fallback = table.get("skt", None)
                if fallback is not None and key == "sst":
                    arr = np.where(np.isnan(arr), fallback, arr)
                else:
                    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
            else:
                # Sparse missing — forward-fill then back-fill.
                mask = np.isnan(arr)
                for i in range(1, len(arr)):
                    if mask[i]:
                        arr[i] = arr[i - 1]
                mask = np.isnan(arr)
                for i in range(len(arr) - 2, -1, -1):
                    if mask[i]:
                        arr[i] = arr[i + 1]
        table[key] = arr
    return table


def build_feature_target_arrays(table: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Create engineered features and the two probabilistic forecasting targets."""

    t2m_c = table["t2m"] - 273.15
    d2m_c = table["d2m"] - 273.15
    skt_c = table["skt"] - 273.15
    sst_c = table["sst"] - 273.15
    msl_hpa = table["msl"] / 100.0
    sp_hpa = table["sp"] / 100.0
    ssrd_wm2 = np.clip(table["ssrd"] / 3600.0, 0.0, None)
    strd_wm2 = np.clip(table["strd"] / 3600.0, 0.0, None)
    tp_mm = table["tp"] * 1000.0
    wind10 = np.sqrt(table["u10"] ** 2 + table["v10"] ** 2)
    wind100 = np.sqrt(table["u100"] ** 2 + table["v100"] ** 2)
    rh = relative_humidity_from_dewpoint(t2m_c, d2m_c)
    heat_index = heat_index_celsius(t2m_c, rh)
    wbgt_like = wbgt_like_celsius(t2m_c, rh, wind10, ssrd_wm2, strd_wm2)

    timestamps = table["timestamp"]
    hours = np.array([ts.hour for ts in timestamps], dtype=np.float32)
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps], dtype=np.float32)

    feature_map = {
        "t2m_c": t2m_c,
        "d2m_c": d2m_c,
        "relative_humidity": rh,
        "skt_c": skt_c,
        "sst_c": sst_c,
        "sea_air_temp_gap_c": sst_c - t2m_c,
        "skin_air_temp_gap_c": skt_c - t2m_c,
        "dewpoint_depression_c": t2m_c - d2m_c,
        "msl_hpa": msl_hpa,
        "sp_hpa": sp_hpa,
        "ssrd_wm2": ssrd_wm2,
        "strd_wm2": strd_wm2,
        "tp_mm": tp_mm,
        "u10": table["u10"],
        "v10": table["v10"],
        "u100": table["u100"],
        "v100": table["v100"],
        "wind10_speed": wind10,
        "wind100_speed": wind100,
        "gust10_speed": table["fg10"],
        "hour_sin": np.sin(2.0 * math.pi * hours / 24.0),
        "hour_cos": np.cos(2.0 * math.pi * hours / 24.0),
        "doy_sin": np.sin(2.0 * math.pi * day_of_year / 366.0),
        "doy_cos": np.cos(2.0 * math.pi * day_of_year / 366.0),
    }

    feature_names = list(feature_map.keys())
    features = np.stack([feature_map[name] for name in feature_names], axis=1).astype(np.float32)
    targets = np.stack([heat_index, wbgt_like], axis=1).astype(np.float32)
    return features, targets, feature_names, timestamps


def split_boundaries(total_length: int, train_ratio: float, val_ratio: float) -> Dict[str, Tuple[int, int]]:
    """Create contiguous time-based split boundaries."""

    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_length),
    }


def build_sample_indices(start: int, end: int, lookback: int, horizon: int, max_samples: int) -> np.ndarray:
    """Create valid rolling-window start indices for one split."""

    first = start + lookback
    last = end - horizon
    if last <= first:
        raise ValueError("The split is too short for the chosen lookback/horizon configuration.")
    indices = np.arange(first, last, dtype=np.int64)
    if max_samples > 0 and len(indices) > max_samples:
        stride = max(1, len(indices) // max_samples)
        indices = indices[::stride][:max_samples]
    return indices


class SequenceDataset(Dataset):
    """Windowed dataset for multi-step probabilistic forecasting."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> None:
        self.features = features
        self.targets = targets
        self.indices = indices
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = self.features[idx - self.lookback : idx]
        y = self.targets[idx : idx + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


class ProbabilisticGRU(nn.Module):
    """GRU encoder with optional multi-head attention, residual decoder, horizon embedding, and Gaussian heads."""

    def __init__(
        self,
        input_size: int,
        target_size: int,
        horizon: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        feature_proj_size: int = 64,
        decoder_hidden_size: int = 128,
        use_attention: bool = True,
        use_horizon_embed: bool = True,
        deterministic: bool = False,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.target_size = target_size
        self.horizon = horizon
        self.use_attention = use_attention
        self.use_horizon_embed = use_horizon_embed
        self.deterministic = deterministic

        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, feature_proj_size),
            nn.LayerNorm(feature_proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=feature_proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        context_size = hidden_size * 2 if use_attention else hidden_size
        if use_attention:
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.context_norm = nn.LayerNorm(context_size)

        decoder_in = context_size + (hidden_size if use_horizon_embed else 0)
        if use_horizon_embed:
            self.horizon_embedding = nn.Embedding(horizon, hidden_size)

        # Residual decoder: project to decoder_hidden_size, then two residual blocks.
        self.decoder_proj = nn.Linear(decoder_in, decoder_hidden_size)
        self.decoder_block1 = nn.Sequential(
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder_norm1 = nn.LayerNorm(decoder_hidden_size)
        self.decoder_block2 = nn.Sequential(
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder_norm2 = nn.LayerNorm(decoder_hidden_size)

        self.mean_head = nn.Linear(decoder_hidden_size, target_size)
        if not deterministic:
            self.logvar_head = nn.Linear(decoder_hidden_size, target_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.feature_proj(x)
        encoder_outputs, hidden = self.encoder(projected)
        last_hidden = hidden[-1]

        if self.use_attention:
            query = last_hidden.unsqueeze(1)
            context, _ = self.multihead_attention(query, encoder_outputs, encoder_outputs)
            context = context.squeeze(1)
            summary = self.context_norm(torch.cat([last_hidden, context], dim=-1))
        else:
            summary = self.context_norm(last_hidden)

        repeated_summary = summary.unsqueeze(1).expand(-1, self.horizon, -1)

        if self.use_horizon_embed:
            horizon_ids = torch.arange(self.horizon, device=x.device)
            horizon_embed = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(x.size(0), -1, -1)
            decoder_input = torch.cat([repeated_summary, horizon_embed], dim=-1)
        else:
            decoder_input = repeated_summary

        # Residual decoder
        h = self.decoder_proj(decoder_input)
        h = self.decoder_norm1(h + self.decoder_block1(h))
        h = self.decoder_norm2(h + self.decoder_block2(h))

        mean = self.mean_head(h)

        if self.deterministic:
            logvar = torch.zeros_like(mean)
        else:
            logvar = self.logvar_head(h)
            logvar = torch.clamp(logvar, min=-8.0, max=6.0)
        return mean, logvar


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a diagonal Gaussian."""

    var = torch.exp(logvar)
    return 0.5 * torch.mean(math.log(2.0 * math.pi) + logvar + ((target - mean) ** 2) / (var + EPS))


def composite_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Blend Gaussian NLL with a small mean-regression term.

    The extra RMSE-like term helps the mean forecast converge faster in early
    training, while the NLL keeps the model probabilistic.
    """

    nll = gaussian_nll(mean, logvar, target)
    mse = torch.mean((target - mean) ** 2)
    variance_penalty = 1e-4 * torch.mean(torch.exp(logvar))
    return nll + 0.1 * mse + variance_penalty


def build_horizon_weights(horizon: int, ratio: float = 2.0) -> torch.Tensor:
    """Build linearly increasing weights for each forecast horizon step.

    *ratio* controls the ratio between the last and first weight.
    A ratio of 2.0 means the 24-h step is weighted twice as much as the 1-h step.
    The weights are normalised so that their mean is 1.0.
    """

    weights = torch.linspace(1.0, ratio, horizon)
    return weights / weights.mean()


def horizon_weighted_composite_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Composite loss with optional per-horizon weighting.

    The horizon weighting encourages the model to pay more attention to
    longer-range forecasts, which are typically harder to predict.
    """

    var = torch.exp(logvar)
    # Per-element NLL: shape [B, H, T]
    nll_elem = 0.5 * (math.log(2.0 * math.pi) + logvar + ((target - mean) ** 2) / (var + EPS))
    if horizon_weights is not None:
        # horizon_weights shape: [H] -> broadcast to [1, H, 1]
        nll_elem = nll_elem * horizon_weights.view(1, -1, 1)
    nll = torch.mean(nll_elem)
    mse = torch.mean((target - mean) ** 2)
    variance_penalty = 1e-4 * torch.mean(var)
    return nll + 0.1 * mse + variance_penalty


def deterministic_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE loss for deterministic baseline."""

    return torch.mean((target - mean) ** 2)


def inverse_standardize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Undo target normalization."""

    return array * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def compute_metrics(
    pred_mean: np.ndarray,
    pred_var: np.ndarray,
    target: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute forecast quality and uncertainty calibration metrics."""

    results: Dict[str, Dict[str, float]] = {}
    z_90 = 1.6448536269514722
    for target_idx, name in enumerate(TARGET_NAMES):
        mean_i = pred_mean[:, :, target_idx]
        var_i = np.maximum(pred_var[:, :, target_idx], EPS)
        target_i = target[:, :, target_idx]
        std_i = np.sqrt(var_i)
        lower = mean_i - z_90 * std_i
        upper = mean_i + z_90 * std_i
        nll = 0.5 * np.mean(np.log(2.0 * math.pi * var_i) + ((target_i - mean_i) ** 2) / var_i)
        rmse = float(np.sqrt(np.mean((target_i - mean_i) ** 2)))
        mae = float(np.mean(np.abs(target_i - mean_i)))
        coverage_90 = float(np.mean((target_i >= lower) & (target_i <= upper)))
        avg_std = float(np.mean(std_i))
        results[name] = {
            "rmse": rmse,
            "mae": mae,
            "gaussian_nll": float(nll),
            "coverage_90": coverage_90,
            "avg_predictive_std": avg_std,
        }
    return results


def predict_distribution(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    mc_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict targets and decompose uncertainty.

    Returns
    -------
    pred_mean : predictive mean in physical units
    total_var : aleatoric + epistemic variance in physical units
    aleatoric_var : data uncertainty in physical units
    targets : ground truth in physical units
    """

    means_all: List[np.ndarray] = []
    total_vars_all: List[np.ndarray] = []
    aleatoric_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    model.train(mc_samples > 1)
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            sample_means = []
            sample_vars = []
            for _ in range(mc_samples):
                mean_norm, logvar_norm = model(x_batch)
                sample_means.append(mean_norm.cpu().numpy())
                sample_vars.append(np.exp(logvar_norm.cpu().numpy()))

            mean_samples = np.stack(sample_means, axis=0)
            var_samples = np.stack(sample_vars, axis=0)
            mean_norm = np.mean(mean_samples, axis=0)
            epistemic_norm = np.var(mean_samples, axis=0)
            aleatoric_norm = np.mean(var_samples, axis=0)
            total_norm = aleatoric_norm + epistemic_norm

            scale = target_std.reshape(1, 1, -1)
            pred_mean = inverse_standardize(mean_norm, target_mean, target_std)
            total_var = total_norm * (scale**2)
            aleatoric_var = aleatoric_norm * (scale**2)
            targets = inverse_standardize(y_batch.cpu().numpy(), target_mean, target_std)

            means_all.append(pred_mean)
            total_vars_all.append(total_var)
            aleatoric_all.append(aleatoric_var)
            targets_all.append(targets)

    return (
        np.concatenate(means_all, axis=0),
        np.concatenate(total_vars_all, axis=0),
        np.concatenate(aleatoric_all, axis=0),
        np.concatenate(targets_all, axis=0),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    loss_fn: Callable = composite_loss,
    horizon_weights: torch.Tensor | None = None,
) -> float:
    """Run one training epoch."""

    model.train()
    losses = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        mean, logvar = model(x_batch)
        if horizon_weights is not None:
            loss = loss_fn(mean, logvar, y_batch, horizon_weights)
        else:
            loss = loss_fn(mean, logvar, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Callable = composite_loss,
    horizon_weights: torch.Tensor | None = None,
) -> float:
    """Evaluate average loss on normalized targets."""

    model.eval()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mean, logvar = model(x_batch)
            if horizon_weights is not None:
                losses.append(float(loss_fn(mean, logvar, y_batch, horizon_weights).item()))
            else:
                losses.append(float(loss_fn(mean, logvar, y_batch).item()))
    return float(np.mean(losses))


def prepare_datasets(config: TrainConfig) -> Tuple[Dict[str, SequenceDataset], Dict[str, np.ndarray], List[str]]:
    """Load raw data, engineer features, normalize them, and build datasets."""

    table = load_era5_table(config.data_path)
    features, targets, feature_names, timestamps = build_feature_target_arrays(table)
    boundaries = split_boundaries(len(features), config.train_ratio, config.val_ratio)

    train_start, train_end = boundaries["train"]
    feature_mean = features[train_start:train_end].mean(axis=0)
    feature_std = features[train_start:train_end].std(axis=0) + EPS
    target_mean = targets[train_start:train_end].mean(axis=0)
    target_std = targets[train_start:train_end].std(axis=0) + EPS

    features_norm = ((features - feature_mean) / feature_std).astype(np.float32)
    targets_norm = ((targets - target_mean) / target_std).astype(np.float32)

    datasets: Dict[str, SequenceDataset] = {}
    split_indices: Dict[str, np.ndarray] = {}
    for split_name, (start, end) in boundaries.items():
        split_indices[split_name] = build_sample_indices(
            start=start,
            end=end,
            lookback=config.lookback,
            horizon=config.horizon,
            max_samples=config.max_samples_per_split,
        )
        datasets[split_name] = SequenceDataset(
            features=features_norm,
            targets=targets_norm,
            indices=split_indices[split_name],
            lookback=config.lookback,
            horizon=config.horizon,
        )

    metadata = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "boundaries": np.array(
            [
                boundaries["train"][0],
                boundaries["train"][1],
                boundaries["val"][0],
                boundaries["val"][1],
                boundaries["test"][0],
                boundaries["test"][1],
            ],
            dtype=np.int64,
        ),
        "timestamps": timestamps,
    }
    return datasets, metadata, feature_names


def collate_loaders(datasets: Dict[str, SequenceDataset], batch_size: int) -> Dict[str, DataLoader]:
    """Build PyTorch dataloaders for all splits."""

    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    config: TrainConfig,
    metadata: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    best_val_loss: float,
) -> None:
    """Persist model weights and all preprocessing metadata."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_names": list(feature_names),
        "target_names": list(TARGET_NAMES),
        "feature_mean": metadata["feature_mean"],
        "feature_std": metadata["feature_std"],
        "target_mean": metadata["target_mean"],
        "target_std": metadata["target_std"],
        "boundaries": metadata["boundaries"],
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def model_variant_label(config: TrainConfig) -> str:
    """Return a human-readable label for the model variant being trained."""

    if config.deterministic:
        return "Deterministic GRU"
    parts = ["Probabilistic GRU"]
    if config.no_attention:
        parts.append("w/o Attention")
    if config.no_horizon_embed:
        parts.append("w/o Horizon Embed")
    return " ".join(parts) if len(parts) == 1 else " (".join(parts[:1]) + ", ".join(parts[1:]) + ")"


def parse_args() -> TrainConfig:
    """Parse command-line arguments into a TrainConfig instance."""

    parser = argparse.ArgumentParser(description="Train a probabilistic GRU for coastal heat-stress forecasting.")
    parser.add_argument("--data-path", default=TrainConfig.data_path)
    parser.add_argument("--checkpoint-path", default=TrainConfig.checkpoint_path)
    parser.add_argument("--metrics-path", default=TrainConfig.metrics_path)
    parser.add_argument("--site", default=TrainConfig.site, choices=list(SITES.keys()))
    parser.add_argument("--lookback", type=int, default=TrainConfig.lookback)
    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--feature-proj-size", type=int, default=TrainConfig.feature_proj_size)
    parser.add_argument("--decoder-hidden-size", type=int, default=TrainConfig.decoder_hidden_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--train-ratio", type=float, default=TrainConfig.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--mc-dropout-samples", type=int, default=TrainConfig.mc_dropout_samples)
    parser.add_argument("--max-samples-per-split", type=int, default=TrainConfig.max_samples_per_split)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--no-attention", action="store_true", help="Disable temporal attention")
    parser.add_argument("--no-horizon-embed", action="store_true", help="Disable horizon embedding")
    parser.add_argument("--deterministic", action="store_true", help="Train deterministic baseline (MSE only)")
    parser.add_argument("--num-attention-heads", type=int, default=TrainConfig.num_attention_heads)
    parser.add_argument("--horizon-weight-ratio", type=float, default=TrainConfig.horizon_weight_ratio)
    args = parser.parse_args()
    return TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})


def build_model(config: TrainConfig, input_size: int) -> ProbabilisticGRU:
    """Build a model from config, respecting ablation flags."""

    return ProbabilisticGRU(
        input_size=input_size,
        target_size=len(TARGET_NAMES),
        horizon=config.horizon,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        feature_proj_size=config.feature_proj_size,
        decoder_hidden_size=config.decoder_hidden_size,
        use_attention=not config.no_attention,
        use_horizon_embed=not config.no_horizon_embed,
        deterministic=config.deterministic,
        num_attention_heads=config.num_attention_heads,
    )


def main() -> None:
    """Main training entry point."""

    config = parse_args()
    set_seed(config.seed)
    device = torch.device(config.device)

    # Select loss function and build horizon weights.
    if config.deterministic:
        loss_fn = deterministic_loss
        h_weights = None
    else:
        loss_fn = horizon_weighted_composite_loss
        h_weights = build_horizon_weights(config.horizon, config.horizon_weight_ratio).to(device)

    print(f"Model variant: {model_variant_label(config)}")
    print(f"Site: {config.site}")

    datasets, metadata, feature_names = prepare_datasets(config)
    loaders = collate_loaders(datasets, config.batch_size)

    model = build_model(config, len(feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device, config.grad_clip, loss_fn, h_weights)
        val_loss = evaluate_epoch(model, loaders["val"], device, loss_fn, h_weights)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint state.")

    model.load_state_dict(best_state)
    save_checkpoint(config.checkpoint_path, model, config, metadata, feature_names, best_val)

    mc_samples = 1 if config.deterministic else max(1, config.mc_dropout_samples)
    pred_mean, total_var, aleatoric_var, targets = predict_distribution(
        model=model,
        loader=loaders["test"],
        device=device,
        target_mean=metadata["target_mean"],
        target_std=metadata["target_std"],
        mc_samples=mc_samples,
    )
    metrics = compute_metrics(pred_mean, total_var, targets)
    metrics["summary"] = {
        "model_variant": model_variant_label(config),
        "site": config.site,
        "best_val_loss": best_val,
        "num_train_samples": len(datasets["train"]),
        "num_val_samples": len(datasets["val"]),
        "num_test_samples": len(datasets["test"]),
        "lookback_hours": config.lookback,
        "forecast_horizon_hours": config.horizon,
        "mc_dropout_samples": mc_samples,
        "feature_count": len(feature_names),
        "deterministic": config.deterministic,
        "no_attention": config.no_attention,
        "no_horizon_embed": config.no_horizon_embed,
        "mean_aleatoric_std": {
            TARGET_NAMES[idx]: float(np.mean(np.sqrt(np.maximum(aleatoric_var[:, :, idx], EPS))))
            for idx in range(len(TARGET_NAMES))
        },
    }
    metrics["history"] = history

    Path(config.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config.metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Checkpoint saved to: {config.checkpoint_path}")
    print(f"Training metrics saved to: {config.metrics_path}")


if __name__ == "__main__":
    main()
