"""Milestone 2 - Step 2: Baseline Benchmark Models.

Simple, explainable script for Mid-Semester Evaluation:
1. Implements reference baselines:
   - Persistence (t-1): Assumes power now will equal power 1 hour ago.
   - Seasonal Naive (t-24): Assumes power today at 12 PM equals power yesterday at 12 PM.
2. Evaluates test week performance (June 11 to June 17, 2020) using RMSE and MAE.
3. Explains why MAPE cannot be computed on solar datasets (night power is 0 kW).
4. Plots actual solar power vs baseline predictions.

Outputs:
  milestone 2/outputs/baseline_metrics.csv
  milestone 2/outputs/baseline_forecasts.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. LOAD DATA & SETUP
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "features" / "features_plant1.csv", parse_dates=["timestamp"], index_col="timestamp")

# We evaluate on the held-out test week
test_mask = df["split"] == "test"
y_all = df["ac_power_kw"]
y_test = y_all[test_mask]
is_daylight_test = df.loc[test_mask, "is_daylight"] == 1
real_actuals_mask = (df.loc[test_mask, "ac_imputed"] == 0).values

print(f"Test week has {len(y_test)} hours ({y_test.index[0]} to {y_test.index[-1]}).")

# ---------------------------------------------------------
# 2. GENERATE BASELINE FORECASTS
# ---------------------------------------------------------
# Baseline 1: Persistence (t-1) -> only valid for 1-hour ahead
pred_persistence_1h = y_all.shift(1).loc[test_mask].clip(lower=0.0)

# Baseline 2: Seasonal Naive (t-24) -> valid for both 1h and 24h day-ahead
pred_seasonal_naive = y_all.shift(24).loc[test_mask].clip(lower=0.0)

# ---------------------------------------------------------
# 3. EVALUATE ERROR METRICS (RMSE & MAE)
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred, mask, daylight_mask):
    """Calculates RMSE and MAE for both all hours and daylight hours."""
    # Filter only real (non-imputed) rows
    t = y_true.values[mask]
    p = y_pred.values[mask]
    day = daylight_mask.values[mask]

    err = t - p
    rmse_all = np.sqrt(np.mean(err ** 2))
    mae_all = np.mean(np.abs(err))

    err_day = t[day] - p[day]
    rmse_day = np.sqrt(np.mean(err_day ** 2))
    mae_day = np.mean(np.abs(err_day))

    return round(rmse_all, 3), round(mae_all, 3), round(rmse_day, 3), round(mae_day, 3)

models = [
    ("Persistence (t-1)", "1h", pred_persistence_1h),
    ("Seasonal Naive (t-24)", "1h", pred_seasonal_naive),
    ("Seasonal Naive (t-24)", "24h", pred_seasonal_naive),
]

metrics_records = []
for name, horizon, preds in models:
    r_all, m_all, r_day, m_day = calculate_metrics(y_test, preds, real_actuals_mask, is_daylight_test)
    metrics_records.append({
        "Model": name,
        "Horizon": horizon,
        "RMSE_All (kW)": r_all,
        "MAE_All (kW)": m_all,
        "RMSE_Daylight (kW)": r_day,
        "MAE_Daylight (kW)": m_day,
    })

metrics_df = pd.DataFrame(metrics_records)
print("\n" + "=" * 70)
print("BASELINE EVALUATION METRICS (TEST WEEK):")
print("=" * 70)
print(metrics_df.to_string(index=False))

# Save metrics CSV
metrics_df.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
print(f"\n[Saved] -> {OUTPUT_DIR / 'baseline_metrics.csv'}")

# ---------------------------------------------------------
# 4. PLOT FORECASTS
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test.values, label="Actual AC Power", color="black", linewidth=2.0)
plt.plot(y_test.index, pred_persistence_1h.values, label="Persistence (t-1)", color="gray", linestyle="--", alpha=0.7)
plt.plot(y_test.index, pred_seasonal_naive.values, label="Seasonal Naive (t-24)", color="blue", alpha=0.7)

plt.title("Baseline Forecasts vs Actual Solar Generation (Test Week)", fontsize=13, fontweight="bold")
plt.xlabel("Timestamp", fontsize=11)
plt.ylabel("AC Power (kW)", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plot_path = OUTPUT_DIR / "baseline_forecasts.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[Saved] -> {plot_path}")

print("\nExplanation for your Professor:")
print("- Persistence (t-1) has a 1-hour lag: it reacts late to sunrise and sunset.")
print("- Seasonal Naive (t-24) works well on sunny days, but fails if yesterday was cloudy and today is clear.")
print("- Important: MAPE cannot be computed because nighttime solar power is 0 kW (division by zero).")
print("  Therefore, we evaluate using RMSE and MAE across all hours and daylight hours.")
