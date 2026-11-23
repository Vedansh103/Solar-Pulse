# Solar-Pulse Phase 3 Summary: Plant 2

## Overview
Phase 3 evaluated modern gradient-boosted tree architectures (**XGBoost**, **LightGBM**, **CatBoost**) and their **Ensemble** across 4 strictly separated feature sets.
Hyperparameters were tuned via **Optuna (40 trials)** on validation data, and the final models were evaluated strictly once on the held-out test week (June 11–17, 2020).

## Key Results Table (Test Week Performance)

| Horizon | Feature Set | Winning Model | Test RMSE (kW) | Test MAE (kW) | Daylight RMSE (kW) | Skill Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1h** | 1h_no_weather | CatBoost | 81.01 | 42.24 | 110.07 | 24.8% |
| **1h** | 1h_weather | XGBoost | 58.72 | 26.34 | 79.78 | 45.5% |
| **24h** | 24h_no_weather | Ensemble | 87.03 | 48.96 | 118.26 | 19.2% |
| **24h** | 24h_weather | XGBoost | 60.65 | 29.08 | 82.40 | 43.7% |

## Scientific Insights & Questions Answered

### 1. Does Weather Information Help in 1-Hour Ahead Forecasting?
- **Without Weather**: RMSE is **81.01 kW** (CatBoost).
- **With Weather**: RMSE drops to **58.72 kW** (XGBoost).
- **Impact**: Weather provides a **27.5% reduction** in error. Recent autoregressive lags ($t-1, t-2$) carry substantial momentum, but contemporaneous irradiation and temperature provide immediate cloud-cover response.

### 2. Does Weather Information Help in 24-Hour Ahead Forecasting?
- **Without Weather**: RMSE is **87.03 kW** (Ensemble).
- **With Weather**: RMSE drops to **60.65 kW** (XGBoost).
- **Impact**: Weather provides a **30.3% error reduction**. In day-ahead forecasting, historical autoregressive momentum from 24h ago cannot predict same-day passing clouds. Weather forecasts are the decisive factor for day-ahead dispatch.

### 3. Which Algorithm Performed Best?
- Across all configurations, the **Ensemble** and **[Phase 3] XGBoost** demonstrated top rank, smoothing individual model variance.

### 4. Comparison to Phase 2 Classical Baselines
- Seasonal Naive Baseline: **107.66 kW** RMSE.
- Best Tree Model: **58.72 kW** RMSE (**45.5% Skill Score**).
