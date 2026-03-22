---
theme: default
title: Probabilistic Heat Stress Forecasting
info: |
  Probabilistic Deep Learning for Heat Stress Assessment
  over Global Coastal Cities Using ERA5 Reanalysis Data
colorSchema: light
fonts:
  sans: Inter
  mono: Fira Code
  provider: google
drawings:
  enabled: false
transition: slide-left
---

# Probabilistic Heat Stress Forecasting

## over Global Coastal Cities

<br>

ERA5 Reanalysis Data (2010-2025) | Probabilistic GRU | Uncertainty Quantification

<br>

<div class="abs-bl m-6 text-sm opacity-60">
COMP0197 Applied Deep Learning
</div>

<!--
Opening slide. This project builds a probabilistic deep learning system for forecasting heat stress indices over 4 coastal cities using 16 years of ERA5 reanalysis data.
-->

---
layout: two-cols
---

# Background & Motivation

<v-clicks>

- Climate change intensifies **urban heat stress**
- Coastal cities face compound heat-humidity risks
- Traditional NWP models: computationally expensive
- Need: **fast, uncertainty-aware** local forecasts
- Key gap: most ML approaches are **deterministic** — no confidence intervals

</v-clicks>

::right::

<br>
<br>

### Research Questions

<v-clicks>

1. Can a lightweight GRU produce **calibrated probabilistic** heat stress forecasts?
2. How do predictions vary across **climate zones**?
3. Which architectural components matter most?

</v-clicks>

<!--
Heat stress is a growing public health concern. We need fast, interpretable forecasts with uncertainty quantification.
-->

---

# Study Sites

Four coastal cities spanning distinct climate zones

<br>

| City | Climate | Lat/Lon | Characteristics |
|------|---------|---------|-----------------|
| **Qingdao** | Temperate monsoon | 36.25N, 120.50E | Humid summers, strong seasonality |
| **Dubai** | Arid subtropical | 25.25N, 55.25E | Extreme dry heat |
| **Singapore** | Tropical | 1.25N, 103.75E | Year-round high humidity |
| **Miami** | Subtropical | 25.75N, 80.25W | Hurricane-prone coast |

<br>

<v-click>

**Why these cities?** — They represent 4 fundamentally different heat stress profiles, testing model generalization across climate regimes.

</v-click>

<!--
We deliberately chose cities with very different climates to test whether a single architecture generalizes well.
-->

---

# Data: ERA5 Reanalysis

**16 years** of hourly data (2010-2025) from ECMWF ERA5

<br>

<div class="grid grid-cols-2 gap-8">
<div>

### 24 Input Features

- 2m temperature, dewpoint, skin/SST
- 10m & 100m wind components, gusts
- Shortwave & longwave radiation
- Surface & sea-level pressure
- Precipitation
- **Engineered:** RH, wind speed, temp gaps
- **Temporal:** hour & day-of-year (sin/cos)

</div>
<div>

### Task Setup

- **Lookback:** 72 hours
- **Forecast horizon:** 24 hours
- **Split:** 70% train / 15% val / 15% test
- **Targets:**
  - Heat Index (HI) — NOAA formula
  - WBGT-like — radiation & wind aware

<br>

> Temporal split avoids data leakage

</div>
</div>

<!--
ERA5 provides global, consistent reanalysis data at hourly resolution. We engineer additional features that are physically meaningful for heat stress.
-->

---
layout: section
---

# Exploratory Data Analysis

---

# Feature Distributions Across Sites

<img src="/images/eda_feature_distributions.png" class="h-100 mx-auto" />

<!--
Climate differences are striking: Singapore temperature spans only 5°C while Qingdao spans over 45°C. Precipitation is extremely right-skewed — near-zero most of the time. These distributions justify using a nonlinear model and highlight the diversity of our study sites.
-->

---
layout: two-cols
---

# Feature Correlations

<img src="/images/eda_correlation_matrix.png" class="h-90" />

::right::

<br>
<br>

### Key Observations

<v-clicks>

- **t2m ↔ Heat Index**: r = 0.98-1.00 — temperature is the dominant driver
- **Wind10 ↔ Wind100**: r = 0.89-0.96 — redundant but model handles it
- **Solar ↔ Thermal radiation**: negatively correlated in tropics
- Singapore: overall **weaker** inter-feature correlations (narrow temp range)

</v-clicks>

<br>

<v-click>

> Correlation structure varies by climate zone — **one model must adapt** to different regimes

</v-click>

<!--
The correlation matrices reveal that feature relationships are climate-dependent. For example, radiation variables are strongly correlated in Qingdao but less so in Singapore. This means the model must learn different feature interactions for each site.
-->

---

# Target Variable Distributions

<img src="/images/eda_target_distributions.png" class="h-80 mx-auto" />

<v-click>

**Non-Gaussian targets**: Qingdao/Miami are bimodal (winter-summer split), Singapore WBGT is extremely concentrated at 25-28°C. Despite this, our Gaussian output works well because the **conditional distribution** $P(y|x_{1:72})$ is unimodal — the bimodality comes from mixing summer/winter samples.

</v-click>

<!--
This is an important finding: the marginal distributions are clearly non-Gaussian with bimodal shapes, but the model predicts conditional distributions given 72h of context. Conditioning removes the seasonal mixing, making the Gaussian assumption empirically valid — confirmed by our PIT and calibration analyses.
-->

---
layout: two-cols
---

# Temporal Patterns

### Diurnal Cycles

<img src="/images/eda_diurnal_cycle.png" class="h-70" />

::right::

<br>

### Autocorrelation Analysis

<img src="/images/eda_autocorrelation.png" class="h-70" />

<v-click>

**72h lookback** captures 3 full diurnal cycles — autocorrelation still >0.3 at this lag

</v-click>

<!--
Left: Clear 24-hour periodicity in all variables. Dubai has the strongest diurnal temperature swing. Right: The autocorrelation analysis validates our 72-hour lookback window. The dashed line at 72h shows correlation is still meaningful, capturing 3 complete day-night cycles. Singapore decays fastest because its day-to-day variation is minimal.
-->

---

# EDA Summary

<div class="grid grid-cols-2 gap-8">
<div>

### Data Quality ✓

- **No missing values** except Dubai SST (land grid point, handled via skin temperature fallback)
- ~140K hourly samples per site (16 years)
- Stationary after deseasonalization — no long-term drift

</div>
<div>

### Design Implications

<v-clicks>

- **Nonlinear model needed**: temperature vs Heat Index is exponential at high-T
- **72h lookback justified**: autocorrelation validates the window choice
- **Gaussian conditional output works**: despite non-Gaussian marginals
- **Climate diversity validated**: 4 sites span extreme ranges in all features

</v-clicks>

</div>
</div>

<br>

<v-click>

> EDA confirms our architectural and hyperparameter choices are **data-driven, not arbitrary**

</v-click>

<!--
This EDA validates key design decisions: the lookback window, the Gaussian output assumption, the choice of nonlinear GRU, and the selection of diverse study sites. These are not arbitrary choices but data-informed decisions.
-->

---

# Model Architecture

**Probabilistic GRU** — ~328K trainable parameters

<img src="/images/model_architecture.png" class="h-96 mx-auto" />

<!--
The model flows from raw ERA5 input through feature projection, bidirectional GRU, optional attention, horizon embedding, residual decoder, to probabilistic output heads producing mean and variance.
-->

---

# Architecture Details

<div class="grid grid-cols-2 gap-6">
<div>

### Encoder

<v-clicks>

- **Feature Projection**: Linear 24→64 + LayerNorm + GELU
- **Bidirectional GRU**: 2 layers, hidden=128
- **Multi-Head Attention**: 4 heads (optional)

</v-clicks>

</div>
<div>

### Decoder

<v-clicks>

- **Horizon Embedding**: Learnable per-step embeddings
- **Residual MLP**: 2 blocks with LayerNorm
- **Dual Output Heads**: $\mu_t$ and $\log\sigma^2_t$

</v-clicks>

</div>
</div>

<br>

<v-click>

### Probabilistic Output

$$p(y_t | x) = \mathcal{N}(\mu_t, \sigma_t^2) \quad \text{for each horizon } h = 1, \ldots, 24$$

**Epistemic uncertainty** via MC Dropout (30 samples at inference)

</v-click>

<!--
The key innovation is the dual-head probabilistic output that predicts both the mean and variance of a Gaussian distribution, enabling full uncertainty quantification.
-->

---

# Training Strategy

<div class="grid grid-cols-2 gap-8">
<div>

### Loss Function

**Horizon-Weighted Composite Loss:**

$$\mathcal{L} = \sum_{h=1}^{24} w_h \left[ \text{NLL}_h + 0.1 \cdot \text{MSE}_h + 10^{-4} \cdot \sigma_h^2 \right]$$

<v-clicks>

- **NLL**: Gaussian negative log-likelihood
- **MSE**: Stabilizes early training
- **Variance penalty**: Prevents runaway uncertainty
- **Horizon weights**: $w_h = 1.0 \to 2.0$ (linear)

</v-clicks>

</div>
<div>

### Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Scheduler | CosineAnnealing |
| Batch size | 256 |
| Max epochs | 60 |
| Early stopping | patience=10 |
| Gradient clip | 1.0 |
| Dropout | 0.2 |

</div>
</div>

<!--
The horizon-weighted loss encourages the model to focus on harder long-range predictions. The variance penalty prevents the model from simply inflating uncertainty to improve NLL.
-->

---

# Training Curves

<img src="/images/training_curves_all.png" class="h-100 mx-auto" />

<!--
All 7 model variants (4 sites + 3 ablation) converge smoothly. Early stopping triggers around epoch 30-40 for most models.
-->

---
layout: section
---

# Results

---

# Cross-Site Performance

<img src="/images/cross_site_comparison.png" class="h-100 mx-auto" />

<!--
This shows RMSE, MAE, CRPS, Coverage, and Skill Score across all 4 sites. Singapore has lowest errors; Dubai has highest skill scores.
-->

---
layout: two-cols
---

# Key Findings

<v-clicks>

- **Singapore**: Lowest RMSE (1.19C HI) — tropical regularity
- **Dubai**: Highest Skill Score (0.82) — arid predictability
- **Qingdao**: Highest errors — complex monsoon dynamics
- **Miami**: Balanced performance — subtropical patterns
- **All sites**: Skill Score > 0.55 (beats persistence)

</v-clicks>

<br>

<v-click>

| Site | HI RMSE | HI CRPS | Skill |
|------|---------|---------|-------|
| Qingdao | 1.83C | 0.96 | 0.63 |
| Dubai | 1.68C | 0.86 | **0.82** |
| Singapore | **1.19C** | **0.65** | 0.55 |
| Miami | 1.47C | 0.76 | 0.74 |

</v-click>

::right::

<img src="/images/cross_site_radar.png" class="h-90 mt-8" />

<!--
The radar chart provides a multi-dimensional view of model performance. Note how different sites excel on different metrics.
-->

---
layout: section
---

# Ablation Study

---

# Ablation: Which Components Matter?

Tested on **Qingdao** (most challenging site)

<img src="/images/ablation_comparison.png" class="h-80 mx-auto" />

<!--
We ablate three components: attention, horizon embedding, and probabilistic output. The results reveal surprising findings about what matters.
-->

---

# Ablation Results

<div class="grid grid-cols-2 gap-6">
<div>

| Variant | HI RMSE | HI CRPS |
|---------|---------|---------|
| Full Model | 1.831 | 0.960 |
| w/o Attention | 1.798 | 0.937 |
| w/o Horizon | 2.502 | 1.390 |
| Deterministic | 1.823 | 2.710 |

</div>
<div>

<v-clicks>

### Three Key Insights

1. **Horizon embedding is critical** — removing it causes **+37% RMSE**

2. **Attention adds no value** — w/o attention is *slightly better* (GRU alone suffices for single-point time series)

3. **Probabilistic output is essential** — deterministic CRPS 2.71 vs 0.96

</v-clicks>

</div>
</div>

<br>

<v-click>

> **Takeaway:** Invest in output-side design (horizon awareness, probabilistic heads), not encoder complexity.

</v-click>

<!--
This is perhaps the most important result. The horizon embedding that tells the model "how far ahead are you predicting" is far more valuable than attention. And probabilistic modeling is essential for uncertainty quantification.
-->

---

# Skill Score vs Persistence

<img src="/images/ablation_skill_score.png" class="h-80 mx-auto" />

<v-click>

Model advantage **grows with forecast horizon**: ~20% at 1h → ~70% at 24h

</v-click>

<!--
The model's relative advantage over simple persistence grows dramatically at longer horizons, showing it learns genuine temporal correlations.
-->

---
layout: section
---

# Uncertainty Quantification

---

# Uncertainty Decomposition

<img src="/images/uncertainty_decomposition_summary.png" class="h-80 mx-auto" />

<v-click>

**Aleatoric uncertainty dominates** (3-5x larger than epistemic) — prediction errors are driven by weather chaos, not model limitations.

</v-click>

<!--
This tells us that further model improvements alone won't dramatically reduce errors. We need better input data or ensemble approaches.
-->

---
layout: two-cols
---

# Calibration

**Well-calibrated predictions**: 90% intervals cover ~90% of true values

<img src="/images/eval_full_qingdao_calibration.png" class="h-70" />

::right::

<br>

### Uncertainty Bands

<img src="/images/eval_full_qingdao_sample_decomposition.png" class="h-70" />

<!--
Left: The calibration curve closely follows the identity line, meaning our predicted confidence levels match realized coverage. Right: A sample forecast showing the predicted mean with aleatoric and epistemic uncertainty bands.
-->

---

# Feature Importance

<img src="/images/feature_importance.png" class="h-90 mx-auto" />

<!--
Permutation importance aligns with physical intuition: temperature and humidity variables dominate. Radiation matters more for WBGT than Heat Index.
-->

---

# Seasonal Analysis

<img src="/images/seasonal_analysis_qingdao.png" class="h-90 mx-auto" />

<!--
Performance varies by season — summer (JJA) is typically harder to predict due to more extreme conditions and convective weather.
-->

---
layout: two-cols
---

# Summary

<v-clicks>

- **Probabilistic GRU** forecasts heat stress with calibrated uncertainty across 4 climate zones
- RMSE: **1.19-1.83C** for Heat Index
- Skill Scores: **0.55-0.82** (all beat persistence)
- **Horizon embedding** is the most critical component (+37% error without it)
- Attention provides **no benefit** for single-point time series
- **Aleatoric > Epistemic**: weather chaos dominates model uncertainty

</v-clicks>

::right::

<br>

### Limitations & Future Work

<v-clicks>

- Single-point ERA5 (no spatial modeling)
- WBGT-like is a proxy, not ISO standard
- 24h horizon only
- Independent site training

<br>

**Future directions:**
- Spatial GNN for multi-point forecasting
- Transfer learning across climate zones
- Conformal prediction for distribution-free intervals
- Integration with NWP ensembles

</v-clicks>

<!--
Our work shows that a lightweight probabilistic GRU can produce well-calibrated heat stress forecasts. The key takeaway: invest in output design, not encoder complexity.
-->

---
layout: center
class: text-center
---

# Thank You

<br>

Probabilistic Heat Stress Forecasting over Global Coastal Cities

<br>

**4 sites** | **16 years** | **~328K params** | **Calibrated uncertainty**

<br>

<div class="opacity-60">

COMP0197 Applied Deep Learning

</div>

<!--
Questions?
-->
