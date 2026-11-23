"""Milestone 2 - Step 3: Holt-Winters Exponential Smoothing.

Simple, explainable script for Mid-Semester Evaluation:
1. Implements Holt-Winters Triple Exponential Smoothing with a 24-hour daily seasonal period.
2. Captures three components:
   - Level (alpha): Base smoothed generation.
   - Trend (beta): Morning rise / afternoon fall rate (damped to prevent runaway).
   - Seasonality (gamma): 24-hour additive daily solar profile.
3. Evaluates 1-hour ahead (operational) and 24-hour ahead (day-ahead block) forecasts on the test week.
4. Computes RMSE, MAE, and Skill Score vs Seasonal Naive baseline.

Outputs:
  milestone 2/outputs/holt_winters_metrics.csv
  milestone 2/outputs/holt_winters_forecasts.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------
# 1. LOAD DATA & SPLIT
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "features" / "features_plant1.csv", parse_dates=["timestamp"], index_col="timestamp")

y = df["ac_power_kw"]
train_idx = df.index[df["split"] == "train"]
val_idx = df.index[df["split"] == "val"]
test_idx = df.index[df["split"] == "test"]

# Fit up to end of validation before testing
fit_idx = df.index[df["split"].isin(["train", "val"])]
fit_series = y.loc[fit_idx]

print(f"Fitting Holt-Winters on {len(fit_series)} hours (May 15 to June 10, 2020)...")

# ---------------------------------------------------------
# 2. FIT HOLT-WINTERS MODEL
# ---------------------------------------------------------
# Daily season period is 24 hours. Additive seasonality matches solar amplitude.
hw_model = ExponentialSmoothing(
    fit_series,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=24,
    initialization_method="estimated",
).fit(optimized=True)

print("Optimal Smoothing Parameters Found:")
print(f"  * Smoothing Level (Alpha):    {hw_model.params['smoothing_level']:.4f}")
print(f"  * Smoothing Trend (Beta):     {hw_model.params['smoothing_trend']:.4f}")
print(f"  * Smoothing Seasonal (Gamma): {hw_model.params['smoothing_seasonal']:.4f}")
print(f"  * Damping Factor (Phi):       {hw_model.params['damping_trend']:.4f}")

# ---------------------------------------------------------
# 3. FORECAST TEST WEEK (1h & 24h Horizons)
# ---------------------------------------------------------
# A. 1-Hour Ahead (Rolling forecast updating states as each actual hour arrives)
full_sim = ExponentialSmoothing(
    y.loc[:test_idx[-1]],
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=24,
    initialization_method="estimated",
).fit(
    smoothing_level=hw_model.params["smoothing_level"],
    smoothing_trend=hw_model.params["smoothing_trend"],
    smoothing_seasonal=hw_model.params["smoothing_seasonal"],
    damping_trend=hw_model.params["damping_trend"],
    optimized=False,
)
pred_1h = full_sim.fittedvalues.loc[test_idx].clip(lower=0.0)

# B. 24-Hour Ahead (Day-ahead block forecast produced at midnight for next 24h)
h24_pieces = []
days = pd.date_range(test_idx[0].floor("D"), test_idx[-1].floor("D"), freq="D")
for day in days:
    day_sub = test_idx[test_idx.floor("D") == day]
    hist_end = day - pd.Timedelta(hours=1)
    
    day_fit = ExponentialSmoothing(
        y.loc[:hist_end],
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated",
    ).fit(
        smoothing_level=hw_model.params["smoothing_level"],
        smoothing_trend=hw_model.params["smoothing_trend"],
        smoothing_seasonal=hw_model.params["smoothing_seasonal"],
        damping_trend=hw_model.params["damping_trend"],
        optimized=False,
    )
    block_pred = day_fit.forecast(24)
    h24_pieces.append(block_pred.loc[day_sub])

pred_24h = pd.concat(h24_pieces).clip(lower=0.0)

# ---------------------------------------------------------
# 4. ERROR METRICS
# ---------------------------------------------------------
baseline_rmse = 132.616  # Seasonal Naive 24h RMSE
y_test = y.loc[test_idx]
real_mask = (df.loc[test_idx, "ac_imputed"] == 0).values
day_mask = real_mask & (df.loc[test_idx, "is_daylight"] == 1).values

def score_predictions(actual, pred):
    err_all = actual.values[real_mask] - pred.values[real_mask]
    err_day = actual.values[day_mask] - pred.values[day_mask]
    r_all = np.sqrt(np.mean(err_all ** 2))
    m_all = np.mean(np.abs(err_all))
    r_day = np.sqrt(np.mean(err_day ** 2))
    m_day = np.mean(np.abs(err_day))
    skill = 1.0 - (r_all / baseline_rmse)
    return round(r_all, 3), round(m_all, 3), round(r_day, 3), round(m_day, 3), round(skill, 3)

r1_all, m1_all, r1_day, m1_day, skill_1h = score_predictions(y_test, pred_1h)
r24_all, m24_all, r24_day, m24_day, skill_24h = score_predictions(y_test, pred_24h)

metrics_df = pd.DataFrame([
    {"Model": "Holt-Winters", "Horizon": "1h", "RMSE_All (kW)": r1_all, "MAE_All (kW)": m1_all, "RMSE_Daylight (kW)": r1_day, "Skill_Score": skill_1h},
    {"Model": "Holt-Winters", "Horizon": "24h", "RMSE_All (kW)": r24_all, "MAE_All (kW)": m24_all, "RMSE_Daylight (kW)": r24_day, "Skill_Score": skill_24h},
])

print("\n" + "=" * 70)
print("HOLT-WINTERS TEST RESULTS:")
print("=" * 70)
print(metrics_df.to_string(index=False))

metrics_df.to_csv(OUTPUT_DIR / "holt_winters_metrics.csv", index=False)
print(f"\n[Saved] -> {OUTPUT_DIR / 'holt_winters_metrics.csv'}")

# ---------------------------------------------------------
# 5. FORECAST PLOT
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(test_idx, y_test.values, label="Actual AC Power", color="black", linewidth=2.0)
plt.plot(test_idx, pred_1h.values, label=f"Holt-Winters 1h (RMSE: {r1_all:.1f} kW)", color="darkorange", linestyle="--")
plt.plot(test_idx, pred_24h.values, label=f"Holt-Winters 24h (RMSE: {r24_all:.1f} kW)", color="crimson", alpha=0.7)

plt.title("Holt-Winters Forecasts vs Actual Solar Power (Test Week)", fontsize=13, fontweight="bold")
plt.xlabel("Timestamp", fontsize=11)
plt.ylabel("AC Power (kW)", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plot_path = OUTPUT_DIR / "holt_winters_forecasts.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[Saved] -> {plot_path}")

print("\nExplanation for your Professor:")
print("- Holt-Winters captures the smooth diurnal 24-hour bell curve.")
print(f"- It improves over Seasonal Naive by {skill_1h*100:.1f}% for 1h ahead (RMSE drops from 132.6 kW to {r1_all:.1f} kW).")
print("- However, without weather inputs, it cannot anticipate sudden cloudy weather.")
