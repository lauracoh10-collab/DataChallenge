# Return Sign Prediction Challenge

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-green.svg)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blueviolet.svg)](https://optuna.org/)

## 📋 Challenge Overview

This project tackles a **binary classification problem** in financial markets: predicting whether an allocation's next-day return will be positive or negative.

### 🎯 Objective

- **Task**: Binary classification (0 = negative return, 1 = positive return)
- **Metric**: Accuracy
- **Data Structure**: Panel time series (date × allocation)
- **Dataset Size**: 527,000 training observations / 31,000 test observations

### 💼 Business Context

Financial portfolio managers need to predict whether their allocations will generate positive or negative returns. This prediction helps in:
- **Risk management**: Identifying potentially loss-making positions
- **Portfolio rebalancing**: Adjusting weights based on expected sign
- **Strategy validation**: Testing momentum vs mean-reversion hypotheses

## 📊 Data Description

### Input Features

Each observation represents an allocation at a specific point in time with:

**Historical Returns** (`RET_1` to `RET_20`)
- 20 days of past daily returns
- `RET_1` = most recent day
- `RET_20` = oldest day

**Signed Volume** (`SIGNED_VOLUME_1` to `SIGNED_VOLUME_20`)
- 20 days of signed trading volume
- Positive = net buying pressure
- Negative = net selling pressure

**Static Features**
- `MEDIAN_DAILY_TURNOVER`: Liquidity metric
- `GROUP`: Allocation group/style (categorical)

### Target Variable

- **Original**: Continuous next-day return
- **Transformed**: Binary (1 if return > 0, else 0)
- **Balance**: ~50% positive class (balanced problem)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/return-prediction-challenge.git
cd return-prediction-challenge

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place your CSV files in the `data/` directory:
- `X_train.csv` - Training features
- `y_train.csv` - Training targets
- `X_test.csv` - Test features

### Run Pipeline

```bash
# Run complete training pipeline
python utils/pipeline.py

# Or use Jupyter notebook
jupyter notebook notebooks/return_prediction_analysis.ipynb
```

## 🔬 Methodology

### Feature Engineering (~80 Features)

The pipeline creates rich features across 7 categories:

#### 1. **Momentum & Mean Reversion**
```python
# Return aggregations over different windows
ret_sum_1d, ret_mean_3d, ret_sum_1w, ret_mean_1m

# Relative momentum
mom_1_vs_5   = recent return - 1-week average
mom_5_vs_20  = 1-week return - 1-month average

# Autocorrelation (mean reversion signal)
autocorr_lag1 = correlation(RET_t, RET_{t-1})

# Consecutive days in same direction
streak = count of consecutive up/down days
```

#### 2. **Volatility & Regime**
```python
# Standard deviations
vol_20, vol_5, vol_ratio = vol_5 / vol_20

# Z-score of latest return
zscore_ret1 = (RET_1 - mean(RET)) / std(RET)

# Risk-adjusted returns
sharpe_20 = mean(RET) / std(RET)

# Drawdown metrics
max_drawdown = minimum cumulative return
current_drawdown = current position vs peak

# Win rate
win_rate_20 = fraction of positive days
```

#### 3. **Distribution Shape**
```python
# Higher moments
skew_20, kurtosis_20

# Range statistics
ret_max_20, ret_min_20, ret_range_20

# Quantiles
ret_q75_20, ret_q25_20, ret_iqr_20
```

#### 4. **Signed Volume**
```python
# Volume aggregations
vol_sum_1d, vol_mean_3d, vol_sum_1w

# Volume standardization
vol_zscore_1 = (VOL_1 - mean) / std

# Volume trend
vol_trend = recent_vol - past_vol
```

#### 5. **Return-Volume Interaction**
```python
# Signal confirmation
ret_vol_agree_1 = sign(RET_1) == sign(VOL_1)

# Correlation
corr_ret_vol_20 = correlation(returns, volumes)
```

#### 6. **Turnover Effects**
```python
turnover_x_vol = TURNOVER × volatility
turnover_x_sharpe = TURNOVER × sharpe_ratio
```

#### 7. **Group Encoding**
```python
# One-hot encoding of allocation groups
grp_1, grp_2, ..., grp_N
```

### Model Architecture

#### **1. LightGBM (Primary Model)**

Gradient boosting decision tree optimized for:
- Fast training on large datasets
- Handling categorical features
- Feature interaction discovery

**Hyperparameter Optimization** (Optuna):
```python
Optimized parameters:
- learning_rate: [0.01, 0.1]
- num_leaves: [20, 100]
- max_depth: [3, 10]
- min_child_samples: [10, 100]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]

Trials: 50
Optimization metric: Validation accuracy
```

#### **2. Logistic Regression (Baseline)**

Linear model with L2 regularization:
- Provides interpretable coefficients
- Captures linear relationships
- Fast inference

**Preprocessing**:
- StandardScaler normalization
- Handle NaN values

#### **3. Ensemble (Final Model)**

Weighted average of probabilities:
```python
P_final = 0.75 × P_lgbm + 0.25 × P_logreg
Prediction = 1 if P_final > 0.5 else 0
```

**Rationale**: LightGBM captures non-linear patterns, LogReg provides stability

### Validation Strategy

**Temporal Split** (crucial for time series):
```python
# Training: First 85% of timestamps
# Validation: Last 15% of timestamps

Why temporal? To avoid look-ahead bias
- Models must predict future using only past
- Random split would leak future information
```

### Explainability (SHAP Analysis)

```python
import shap

# TreeExplainer for LightGBM
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_validation)

# Visualizations
shap.summary_plot(shap_values, X_validation)  # Global importance
shap.force_plot(...)  # Individual prediction explanation
```

**Top 20 Features** (typical ranking):
1. `ret_sum_1d` - Most recent return
2. `sign_ret_1` - Sign of latest return
3. `mom_1_vs_5` - Short-term momentum
4. `vol_20` - Historical volatility
5. `sharpe_20` - Risk-adjusted return
... (see SHAP output for full ranking)

## 📈 Results

### Validation Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| LightGBM | ~0.55-0.57 | Best single model |
| LogReg | ~0.52-0.54 | Baseline |
| **Ensemble** | **~0.56-0.58** | **Production model** |

### Key Insights

1. **Recent returns are most predictive**
   - `ret_sum_1d` consistently top feature
   - Short-term momentum exists

2. **Volatility regime matters**
   - High volatility → more predictable
   - Vol ratio captures regime shifts

3. **Group effects are strong**
   - Different allocation styles behave differently
   - Group dummies add significant value

4. **Mean reversion signals work**
   - Negative autocorrelation predicts reversals
   - Streak length useful for extreme moves

5. **Volume confirms direction**
   - Return-volume agreement improves accuracy
   - Volume surge often precedes continuation

## 🛠️ Project Structure

```
return-prediction-challenge/
├── notebooks/
│   └── return_prediction_analysis.ipynb  # Interactive analysis
├── utils/
│   └── pipeline.py                       # Complete pipeline script
├── data/
│   ├── X_train.csv                       # (not in repo - too large)
│   ├── y_train.csv
│   └── X_test.csv
├── models/
│   └── submission.csv                    # Predictions output
├── figures/
│   └── eda.png                           # Exploratory plots
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Usage Examples

### Basic Training

```python
from utils.pipeline import run_pipeline

# Run complete pipeline
lgbm_model, submission = run_pipeline()

# Output:
# - Trained models
# - submission.csv with predictions
# - Feature importance analysis
```

### Custom Training

```python
from utils.pipeline import *

# Load data
X_train, y_train, X_test = load_data()

# Feature engineering
X_train_feat = build_features(X_train)

# Temporal split
X_tr, y_tr, X_val, y_val = temporal_split(
    X_train_feat, y_train, X_train, val_ratio=0.15
)

# Train LightGBM with custom params
params = {
    'learning_rate': 0.05,
    'num_leaves': 50,
    'max_depth': 7
}
model = train_lgbm(X_tr, y_tr, X_val, y_val, params)

# SHAP analysis
shap_analysis(model, X_val)
```

### EDA

```python
from utils.pipeline import quick_eda

# Generate exploratory plots
quick_eda(X_train_raw, y_train)

# Outputs:
# - Return distribution histogram
# - Autocorrelation distribution
# - Turnover by group bar chart
```

## 📊 Visualization Gallery

### Feature Importance (SHAP)
![SHAP Summary](figures/shap_summary.png)

### Return Distribution
![Returns Histogram](figures/return_distribution.png)

### Model Performance
![Validation Accuracy](figures/val_accuracy.png)


