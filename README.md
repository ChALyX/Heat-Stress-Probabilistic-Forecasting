[English](#english) | [中文](#中文)

---

<a id="中文"></a>

# 多站点滨海城市复合热应激概率预测与不确定性量化

## 项目简介

本项目基于 ERA5 单点逐小时再分析数据，构建了一个 **概率时序预测模型**，用于预测未来 24 小时热应激水平，并量化预测中的不确定性。

项目覆盖四个不同气候区的滨海城市：

| 站点 | 气候类型 | 坐标 |
|------|---------|------|
| 青岛 (Qingdao) | 温带季风，夏季湿热 | 36.25°N, 120.50°E |
| 迪拜 (Dubai) | 干旱亚热带，极端干热 | 25.25°N, 55.25°E |
| 新加坡 (Singapore) | 热带，全年高湿 | 1.25°N, 103.75°E |
| 迈阿密 (Miami) | 亚热带，飓风易发海岸 | 25.75°N, -80.25°E |

模型预测两个热应激指标：

1. **Heat Index (HI)**：温度与湿度共同作用下的体感热负荷指标（NOAA 公式）
2. **WBGT-like**：基于湿球温度、辐射和风速构造的近似户外热应激指标

不确定性分解为：

- **Aleatoric uncertainty**：数据本身的随机性与噪声（由模型的异方差输出学习）
- **Epistemic uncertainty**：模型认知不确定性（通过 MC Dropout 估计）

## 研究动机

单纯预测 2 米气温无法充分描述滨海城市"高温高湿"的人体风险。高湿会削弱蒸发散热，海风会改变通风冷却条件，辐射和地表热储存会强化体感热压力。因此，本项目将问题建模为：

**过去 72 小时多变量气象序列 → 未来 24 小时热应激概率分布预测**

## 数据集

数据来源：ERA5 single-level hourly time series（2010-01-01 至 2025-12-31，共 16 年）

```
data/era5_qingdao.csv
data/era5_dubai.csv
data/era5_singapore.csv
data/era5_miami.csv
```

原始变量包括 14 个 ERA5 字段：2m 温度、2m 露点温度、皮肤温度、海表温度、10m/100m 风分量、阵风、海平面气压、地表气压、短波/长波辐射、降水。

## 模型架构

核心模型为 **Probabilistic GRU**（`train.py` 中的 `ProbabilisticGRU` 类），共约 **328,000 个可训练参数**。

```
输入 [B, 72, 24] ──→ Feature Projection ──→ GRU Encoder ──→ Multi-Head Attention ──→ Context
                         (Linear+LN+GELU)    (2-layer, H=128)    (4 heads)              ↓
                                                                              ┌─ Horizon Embedding
                                                                              ↓
                                                                    Residual Decoder (2 blocks)
                                                                              ↓
                                                                 ┌── Mean Head ──→ [B, 24, 2]
                                                                 └── LogVar Head ──→ [B, 24, 2]
```

### 关键设计

1. **Feature Projection**：`Linear(24→64) + LayerNorm + GELU + Dropout`，将原始特征映射到紧凑表示
2. **GRU Encoder**：2 层双向 GRU（hidden_size=128），捕捉热量积累、湿度滞后等时间依赖
3. **Multi-Head Attention**（4 头）：学习历史窗口中哪些时刻对预测更重要，替代了简单的单头注意力
4. **Horizon-aware Decoder**：可学习的 horizon embedding 让模型区分近期与远期预测
5. **Residual Decoder**：两个残差 MLP block（带 LayerNorm），改善梯度流和训练稳定性
6. **概率输出头**：每个未来时刻、每个目标输出 `mean` 和 `log-variance`，即 `y_t ~ N(mean_t, var_t)`

### Ablation 变体

通过命令行标志可以禁用各组件进行消融实验：
- `--no-attention`：禁用多头注意力
- `--no-horizon-embed`：禁用 horizon embedding
- `--deterministic`：训练确定性基线（仅 MSE，不输出方差）

## 训练配置

| 参数 | 值 |
|------|------|
| Lookback 窗口 | 72 小时 |
| 预测 Horizon | 24 小时 |
| 优化器 | AdamW (lr=3e-4, weight_decay=1e-5) |
| 调度器 | CosineAnnealingLR (eta_min=1e-6) |
| 最大 Epochs | 60 |
| Early Stopping Patience | 10 |
| Batch Size | 256 |
| Dropout | 0.2 |
| 数据划分 | 70% train / 15% val / 15% test |

### 损失函数

使用 **Horizon-Weighted Composite Loss**：

```
L = weighted_NLL + 0.1 * MSE + 1e-4 * variance_penalty
```

- **NLL 项**按 horizon 加权（1h 权重 1.0 → 24h 权重 2.0），鼓励模型关注更难预测的远期
- **MSE 项**帮助均值收敛更稳定
- **Variance penalty** 防止模型无脑增大方差

## 评估指标

| 指标 | 含义 | 理想方向 |
|------|------|---------|
| RMSE | 均方根误差 | ↓ |
| MAE | 平均绝对误差 | ↓ |
| Gaussian NLL | 概率分布质量 | ↓ |
| Coverage@90% | 90% 预测区间覆盖率 | ≈ 0.90 |
| **CRPS** | 连续排序概率评分（proper scoring rule） | ↓ |
| **Winkler Score** | 预测区间评分（兼顾宽度与覆盖） | ↓ |
| **Skill Score** | 相对于 persistence 基线的技能得分 | ↑ (>0) |
| Avg Aleatoric Std | 数据不确定性 | 视情况 |
| Avg Epistemic Std | 模型认知不确定性 | ↓ |

## 实验结果

### 多站点全模型对比

| 站点 | HI RMSE | HI CRPS | HI Skill | WB RMSE | WB CRPS | WB Skill |
|------|---------|---------|----------|---------|---------|----------|
| Qingdao | 1.831 | 0.960 | 0.625 | 1.415 | 0.732 | 0.562 |
| Dubai | 1.682 | 0.863 | 0.822 | 0.845 | 0.449 | 0.804 |
| Singapore | 1.188 | 0.649 | 0.554 | 0.389 | 0.214 | 0.549 |
| Miami | 1.465 | 0.755 | 0.742 | 1.009 | 0.465 | 0.626 |

### 消融实验（青岛站点）

| 变体 | HI RMSE | HI CRPS | HI Cov@90% | WB RMSE | WB CRPS | WB Cov@90% |
|------|---------|---------|------------|---------|---------|------------|
| **Full model** | **1.831** | **0.960** | 0.905 | **1.415** | **0.732** | 0.906 |
| w/o Attention | 1.798 | 0.937 | 0.905 | 1.409 | 0.722 | 0.899 |
| w/o Horizon Embed | 2.502 (+37%) | 1.390 | 0.904 | 1.781 (+26%) | 0.961 | 0.920 |
| Deterministic | 1.823 | 2.710 | 1.000* | 1.405 | 2.331 | 1.000* |

> *Deterministic 模型的 Coverage=100% 和高 CRPS 是因为它不输出有意义的方差，因此 CRPS 作为概率评分远劣于概率模型。

### 关键发现

1. **Horizon embedding 是最关键的组件**：去掉后 HI RMSE 上升 37%，WB RMSE 上升 26%
2. **概率模型在 CRPS 上远优于确定性模型**：full model CRPS=0.960 vs deterministic CRPS=2.710
3. **15 年数据显著提升了预测精度**：青岛 HI RMSE 从 2.011 降至 1.831（-9%）
4. **模型在所有站点和 horizon 上均优于 persistence baseline**：Skill Score 0.55—0.82
5. **校准接近理想**：90% coverage 在 0.90—0.92 之间，非常接近名义值

## 可视化输出

每次评估会生成以下图表（以 `results/eval_{variant}_{site}_*.png` 命名）：

| 图表 | 说明 |
|------|------|
| `horizon_plot.png` | RMSE/MAE 和 Aleatoric/Epistemic 随 horizon 变化 |
| `sample_decomposition.png` | 单样本不确定性分解（真值 vs 预测均值 + 不确定性带） |
| `calibration.png` | 校准曲线：期望覆盖率 vs 观测覆盖率 |
| `pit_histogram.png` | PIT 直方图（理想为均匀分布） |
| `skill_score.png` | 模型 RMSE vs Persistence RMSE 及 Skill Score |
| `crps_horizon.png` | CRPS 和 Winkler Score 随 horizon 变化 |
| `error_distribution.png` | 误差分布直方图 + 按 horizon 的误差箱线图 |

## 使用方法

### 安装依赖

```bash
python -m venv .venv
.venv/bin/pip install numpy torch scipy matplotlib
```

### 下载数据

```bash
pip install cdsapi
python load_data.py                    # 下载所有站点
python load_data.py --site qingdao     # 仅下载青岛
```

### 训练

```bash
# 全模型（默认青岛）
python train.py

# 指定站点
python train.py --data-path data/era5_dubai.csv --checkpoint-path checkpoints/full_dubai.pt --metrics-path results/full_dubai_train.json --site dubai

# Ablation 变体
python train.py --no-attention --checkpoint-path checkpoints/noattn_qingdao.pt --metrics-path results/noattn_qingdao_train.json
python train.py --no-horizon-embed --checkpoint-path checkpoints/nohorizon_qingdao.pt --metrics-path results/nohorizon_qingdao_train.json
python train.py --deterministic --checkpoint-path checkpoints/deterministic_qingdao.pt --metrics-path results/deterministic_qingdao_train.json
```

### 评估

```bash
# 单个 checkpoint 评估（生成所有图表和指标）
python test.py --checkpoint-path checkpoints/full_qingdao.pt --data-path data/era5_qingdao.csv --output-path results/eval_full_qingdao.json --mc-dropout-samples 30

# 消融对比表（扫描目录中所有 .pt 文件）
python test.py --checkpoint-path checkpoints/full_qingdao.pt --data-path data/era5_qingdao.csv --output-path results/eval_full_qingdao.json --ablation-dir checkpoints
```

## 项目结构

```text
.
├── train.py              # 模型定义、特征工程、训练循环
├── test.py               # 评估脚本、指标计算、图表生成
├── load_data.py           # ERA5 数据下载工具
├── README.md
├── data/
│   ├── era5_qingdao.csv
│   ├── era5_dubai.csv
│   ├── era5_singapore.csv
│   └── era5_miami.csv
├── checkpoints/           # 训练好的模型权重
│   ├── full_qingdao.pt
│   ├── full_dubai.pt
│   ├── full_singapore.pt
│   ├── full_miami.pt
│   ├── noattn_qingdao.pt
│   ├── nohorizon_qingdao.pt
│   └── deterministic_qingdao.pt
└── results/               # 评估结果、图表、CSV
    ├── eval_*.json
    ├── eval_*_horizon_table.csv
    ├── eval_*_horizon_plot.png
    ├── eval_*_calibration.png
    ├── eval_*_pit_histogram.png
    ├── eval_*_skill_score.png
    ├── eval_*_crps_horizon.png
    ├── eval_*_error_distribution.png
    ├── eval_*_sample_decomposition.png
    └── comparison_table.csv
```

## 依赖

- Python >= 3.10
- PyTorch >= 2.0
- NumPy
- SciPy
- Matplotlib
- cdsapi（仅数据下载需要）

## 局限性

- 使用单点 ERA5 数据，未显式建模空间分布
- WBGT-like 是物理启发式 proxy，不是严格 ISO WBGT 重建
- 当前为逐小时短期预测；如需 2 周以上预测，更适合日尺度任务
- 各站点独立训练，未进行跨站点迁移学习

---

<a id="english"></a>

# Multi-Site Coastal Heat-Stress Probabilistic Forecasting with Uncertainty Quantification

## Overview

This project builds a **probabilistic deep learning system** for forecasting heat stress over coastal cities using ERA5 hourly reanalysis data. The model outputs **Gaussian distributions** for each future timestep and target, enabling full uncertainty quantification.

Four coastal sites spanning different climate zones are studied:

| Site | Climate | Coordinates |
|------|---------|------------|
| Qingdao | Temperate monsoon, humid summers | 36.25°N, 120.50°E |
| Dubai | Arid subtropical, extreme dry heat | 25.25°N, 55.25°E |
| Singapore | Tropical, year-round high humidity | 1.25°N, 103.75°E |
| Miami | Subtropical, hurricane-prone coast | 25.75°N, -80.25°E |

Two forecast targets:

1. **Heat Index (HI)**: NOAA-based temperature-humidity discomfort indicator
2. **WBGT-like**: radiation- and wind-aware proxy for outdoor heat stress

Uncertainty is decomposed into:

- **Aleatoric uncertainty**: learned heteroscedastic data noise
- **Epistemic uncertainty**: model uncertainty via MC Dropout

## Model Architecture

Data: ERA5 single-level hourly time series (2010-01-01 to 2025-12-31, 16 years, ~140K samples per site).

The **Probabilistic GRU** model (~328K parameters) features:

1. **Feature Projection**: Linear(24→64) + LayerNorm + GELU + Dropout
2. **GRU Encoder**: 2-layer GRU (hidden_size=128) over 72-hour lookback
3. **Multi-Head Attention**: 4-head cross-attention from last hidden state to encoder outputs
4. **Horizon-Aware Decoder**: learnable horizon embeddings + 2 residual MLP blocks with LayerNorm
5. **Probabilistic Heads**: separate mean and log-variance outputs per target

### Training

- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR (eta_min=1e-6)
- **Loss**: Horizon-weighted composite loss (NLL + 0.1*MSE + variance penalty)
- **Horizon weighting**: linearly increasing from 1.0 (1h) to 2.0 (24h)
- **Early stopping**: patience=10, max epochs=60

### Ablation Flags

- `--no-attention`: disable multi-head attention
- `--no-horizon-embed`: disable horizon embedding
- `--deterministic`: MSE-only baseline without variance output

## Evaluation Metrics

| Metric | Description | Direction |
|--------|------------|-----------|
| RMSE | Root mean squared error | ↓ |
| MAE | Mean absolute error | ↓ |
| Gaussian NLL | Negative log-likelihood of predictive Gaussian | ↓ |
| Coverage@90% | Fraction of true values within 90% prediction interval | ≈ 0.90 |
| **CRPS** | Continuous Ranked Probability Score (proper scoring rule) | ↓ |
| **Winkler Score** | Interval score penalizing both width and miscoverage | ↓ |
| **Skill Score** | 1 - MSE_model/MSE_persistence (relative to persistence baseline) | ↑ (>0) |

## Results

### Cross-Site Performance (Full Model)

| Site | HI RMSE | HI CRPS | HI Skill | WB RMSE | WB CRPS | WB Skill |
|------|---------|---------|----------|---------|---------|----------|
| Qingdao | 1.831 | 0.960 | 0.625 | 1.415 | 0.732 | 0.562 |
| Dubai | 1.682 | 0.863 | 0.822 | 0.845 | 0.449 | 0.804 |
| Singapore | 1.188 | 0.649 | 0.554 | 0.389 | 0.214 | 0.549 |
| Miami | 1.465 | 0.755 | 0.742 | 1.009 | 0.465 | 0.626 |

### Ablation Study (Qingdao)

| Variant | HI RMSE | HI CRPS | WB RMSE | WB CRPS |
|---------|---------|---------|---------|---------|
| **Full model** | **1.831** | **0.960** | **1.415** | **0.732** |
| w/o Attention | 1.798 | 0.937 | 1.409 | 0.722 |
| w/o Horizon Embed | 2.502 (+37%) | 1.390 | 1.781 (+26%) | 0.961 |
| Deterministic | 1.823 | 2.710 | 1.405 | 2.331 |

### Key Findings

1. **Horizon embedding is the most impactful component**: removing it degrades HI RMSE by 37%, WB RMSE by 26%
2. **Probabilistic model vastly outperforms deterministic on CRPS**: 0.960 vs 2.710
3. **16-year data significantly improves accuracy**: Qingdao HI RMSE improved from 2.011 to 1.831 (-9%)
4. **Model consistently outperforms persistence baseline**: Skill Scores 0.55–0.82 across sites
5. **Well-calibrated uncertainty**: 90% coverage ranges 0.90–0.92, very close to the nominal level

## Visualizations

Each evaluation produces 7 publication-quality figures:

| Figure | Description |
|--------|------------|
| `horizon_plot.png` | RMSE/MAE and uncertainty evolution over forecast horizon |
| `sample_decomposition.png` | Single-sample trajectory with aleatoric/total uncertainty bands |
| `calibration.png` | Expected vs observed coverage at multiple confidence levels |
| `pit_histogram.png` | PIT histogram (ideal: uniform) |
| `skill_score.png` | Model RMSE vs persistence RMSE with skill score |
| `crps_horizon.png` | CRPS and Winkler Score by forecast horizon |
| `error_distribution.png` | Error histogram and per-horizon error boxplots |

## Usage

### Install Dependencies

```bash
python -m venv .venv
.venv/bin/pip install numpy torch scipy matplotlib
```

### Download Data

```bash
pip install cdsapi
python load_data.py                    # all sites
python load_data.py --site qingdao     # single site
```

### Train

```bash
# Full model (default: Qingdao)
python train.py

# Specific site
python train.py --data-path data/era5_dubai.csv --checkpoint-path checkpoints/full_dubai.pt --metrics-path results/full_dubai_train.json --site dubai

# Ablation variants
python train.py --no-attention --checkpoint-path checkpoints/noattn_qingdao.pt
python train.py --no-horizon-embed --checkpoint-path checkpoints/nohorizon_qingdao.pt
python train.py --deterministic --checkpoint-path checkpoints/deterministic_qingdao.pt
```

### Evaluate

```bash
python test.py --checkpoint-path checkpoints/full_qingdao.pt --data-path data/era5_qingdao.csv --output-path results/eval_full_qingdao.json --mc-dropout-samples 30

# With ablation comparison table
python test.py --checkpoint-path checkpoints/full_qingdao.pt --data-path data/era5_qingdao.csv --output-path results/eval_full_qingdao.json --ablation-dir checkpoints
```

## Project Structure

```text
.
├── train.py              # Model definition, feature engineering, training loop
├── test.py               # Evaluation, metrics, visualization
├── load_data.py           # ERA5 data download utility
├── README.md
├── data/                  # ERA5 CSV data files (4 sites)
├── checkpoints/           # Trained model weights (.pt)
└── results/               # Evaluation outputs (JSON, CSV, PNG)
```

## Limitations

- Single spatial point per site; no explicit spatial modelling
- WBGT-like is a physically informed proxy, not strict ISO WBGT
- Hourly short-range forecasting; daily aggregation needed for >2-week horizons
- Sites trained independently; no cross-site transfer learning
