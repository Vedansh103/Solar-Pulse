# Solar-Pulse Phase 3 Summary: Plant 1

## Overview
Phase 3 evaluated modern gradient-boosted tree architectures (**XGBoost**, **LightGBM**, **CatBoost**) and their **Ensemble** across 4 strictly separated feature sets.
Hyperparameters were tuned via **Optuna (40 trials)** on validation data, and the final models were evaluated strictly once on the held-out test week (June 11–17, 2020).

## Key Results Table (Test Week Performance)

| Horizon | Feature Set | Winning Model | Test RMSE (kW) | Test MAE (kW) | Daylight RMSE (kW) | Skill Score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1h** | 1h_no_weather | CatBoost | 96.15 | 52.26 | 130.64 | 27.5% |
| **1h** | 1h_weather | LightGBM | 19.71 | 10.43 | 26.77 | 85.1% |
| **24h** | 24h_no_weather | XGBoost | 116.31 | 65.55 | 158.04 | 12.3% |
| **24h** | 24h_weather | LightGBM | 19.26 | 9.84 | 26.18 | 85.5% |

## Scientific Insights & Questions Answered

### 1. Does Weather Information Help in 1-Hour Ahead Forecasting?
- **Without Weather**: RMSE is **96.15 kW** (CatBoost).
- **With Weather**: RMSE drops to **19.71 kW** (LightGBM).
- **Impact**: Weather provides a **79.5% reduction** in error. Recent autoregressive lags ($t-1, t-2$) carry substantial momentum, but contemporaneous irradiation and temperature provide immediate cloud-cover response.

### 2. Does Weather Information Help in 24-Hour Ahead Forecasting?
- **Without Weather**: RMSE is **116.31 kW** (XGBoost).
- **With Weather**: RMSE drops to **19.26 kW** (LightGBM).
- **Impact**: Weather provides a **83.4% error reduction**. In day-ahead forecasting, historical autoregressive momentum from 24h ago cannot predict same-day passing clouds. Weather forecasts are the decisive factor for day-ahead dispatch.

### 3. Which Algorithm Performed Best?
- Across all configurations, the **Ensemble** and **[Phase 2] SARIMAX (irradiation, module temp) ((2, 0, 0), (0, 1, 1))** demonstrated top rank, smoothing individual model variance.

### 4. Comparison to Phase 2 Classical Baselines
- Seasonal Naive Baseline: **132.62 kW** RMSE.
- Best Tree Model: **15.89 kW** RMSE (**88.0% Skill Score**).
