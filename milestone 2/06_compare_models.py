"""Milestone 2 - Step 6: Multi-Model Comparison & Leaderboard.

Simple, explainable script for Mid-Semester Evaluation:
1. Gathers test metrics from all 5 classical models:
   - Persistence (t-1)
   - Seasonal Naive (t-24)
   - Holt-Winters Exponential Smoothing
   - SARIMA ((1,0,1) x (1,1,1)_24)
   - SARIMAX with Weather Inputs
2. Produces a unified leaderboard sorted by error (RMSE).
3. Plots side-by-side test forecasts comparing all models.
4. Generates an executive summary text for your Mid-Sem presentation slides.

Outputs:
  milestone 2/outputs/final_leaderboard.csv
  milestone 2/outputs/all_models_comparison.png
  milestone 2/outputs/milestone2_summary.txt
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Verify that previous step metric files exist
metric_files = {
    "Baselines": OUTPUT_DIR / "baseline_metrics.csv",
    "Holt-Winters": OUTPUT_DIR / "holt_winters_metrics.csv",
    "SARIMA": OUTPUT_DIR / "sarima_metrics.csv",
    "SARIMAX": OUTPUT_DIR / "sarimax_metrics.csv",
}

for name, path in metric_files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}! Run the earlier scripts (02, 03, 04, 05) first.")

# ---------------------------------------------------------
# 2. COMBINE INTO UNIFIED LEADERBOARD
# ---------------------------------------------------------
df_base = pd.read_csv(metric_files["Baselines"])
df_hw = pd.read_csv(metric_files["Holt-Winters"])
df_sarima = pd.read_csv(metric_files["SARIMA"])
df_sarimax = pd.read_csv(metric_files["SARIMAX"])

# Standardize columns
baseline_rmse = 132.616
df_base["Skill_Score"] = (1.0 - (df_base["RMSE_All (kW)"] / baseline_rmse)).round(3)

combined = pd.concat([df_base, df_hw, df_sarima, df_sarimax], ignore_index=True)
combined["Uses_Weather"] = combined["Model"].str.contains("Weather|SARIMAX")

# Sort by RMSE ascending (best model first)
leaderboard = combined.sort_values(by=["Horizon", "RMSE_All (kW)"], ascending=[True, True]).reset_index(drop=True)

print("=" * 85)
print("MILESTONE 2: COMPLETE MODEL LEADERBOARD (TEST WEEK BENCHMARK)")
print("=" * 85)
print(leaderboard.to_string(index=False))

leaderboard_path = OUTPUT_DIR / "final_leaderboard.csv"
leaderboard.to_csv(leaderboard_path, index=False)
print(f"\n[Saved] -> {leaderboard_path}")

# ---------------------------------------------------------
# 3. GENERATE UNIFIED COMPARISON PLOT
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
df_raw = pd.read_csv(ROOT / "data" / "features" / "features_plant1.csv", parse_dates=["timestamp"], index_col="timestamp")
test_df = df_raw[df_raw["split"] == "test"]

# Plot first 72 hours of test week so lines are crisp and distinct
sample_idx = test_df.index[:72]
y_sample = test_df.loc[sample_idx, "ac_power_kw"]

plt.figure(figsize=(15, 6))
plt.plot(sample_idx, y_sample, label="Actual AC Power", color="black", linewidth=2.5)

# Load existing test forecasts if available
forecasts_path = ROOT / "reports" / "phase2" / "plant1" / "forecasts_test.csv"
if forecasts_path.exists():
    f_df = pd.read_csv(forecasts_path, parse_dates=["timestamp"], index_col="timestamp").loc[sample_idx]
    
    # Plot top models
    col_map = {
        "Seasonal naive (t-24) | 1h": ("Seasonal Naive (t-24)", "blue", ":"),
        "Holt-Winters {'trend': 'add', 'damped_trend': True} | 1h": ("Holt-Winters 1h", "darkorange", "--"),
        "SARIMA ((1, 0, 1), (1, 1, 1)) | 1h": ("SARIMA 1h", "purple", "-."),
        "SARIMAX (irradiation, module temp) ((2, 0, 0), (0, 1, 1)) | 1h": ("SARIMAX (Weather) 1h", "green", "-"),
    }
    for col, (lbl, clr, ls) in col_map.items():
        if col in f_df.columns:
            plt.plot(sample_idx, f_df[col], label=lbl, color=clr, linestyle=ls, linewidth=1.8, alpha=0.85)

plt.title("Milestone 2 Model Comparison: Actual vs Forecasts (First 3 Days of Test Week)", fontsize=13, fontweight="bold")
plt.xlabel("Timestamp", fontsize=11)
plt.ylabel("AC Power (kW)", fontsize=11)
plt.legend(loc="upper right", framealpha=0.9)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plot_out = OUTPUT_DIR / "all_models_comparison.png"
plt.savefig(plot_out, dpi=150)
plt.close()
print(f"[Saved] -> {plot_out}")

# ---------------------------------------------------------
# 4. GENERATE SUMMARY FOR PRESENTATION SLIDES
# ---------------------------------------------------------
summary_text = f"""================================================================================
MILESTONE 2 (MID-SEM EVALUATION) SUMMARY & KEY TAKEAWAYS
================================================================================

1. PROBLEM & DOMAIN:
   - Forecasting hourly solar generation (Plant 1) over short-term (1h) and day-ahead (24h) horizons.
   - Evaluated on a 7-day held-out test week (168 hours: June 11 to 17, 2020).

2. KEY FINDINGS ACROSS MODELS (1-Hour Ahead):
   - Persistence (t-1):      RMSE = 143.05 kW  (Skill: -7.9% vs baseline)
   - Seasonal Naive (t-24):  RMSE = 132.62 kW  (Baseline benchmark)
   - Holt-Winters:           RMSE =  95.32 kW  (Skill: +28.1% improvement)
   - SARIMA:                 RMSE =  87.52 kW  (Skill: +34.0% improvement)
   - SARIMAX (Weather):      RMSE =  15.89 kW  (Skill: +88.0% improvement)

3. WHY WEATHER FEATURES ARE DECISIVE:
   - Pure statistical models (Holt-Winters and SARIMA) capture the smooth daily curve,
     reducing error by ~28% to 34% over Seasonal Naive.
   - However, they cannot foresee sudden clouds.
   - Adding exogenous weather inputs (SARIMAX with irradiation & module temperature)
     reduces RMSE down to 15.89 kW (+88% Skill Score), proving that physical weather
     measurements are the primary driver of solar generation.

4. METRIC CAVEAT ON MAPE:
   - Evaluator note: Standard MAPE is undefined / invalid on solar PV data because
     nighttime actual power is 0 kW (division by zero).
   - Therefore, RMSE and MAE across all hours and daylight hours are the standard metrics.
================================================================================
"""

with open(OUTPUT_DIR / "milestone2_summary.txt", "w") as f:
    f.write(summary_text)

print(f"[Saved] -> {OUTPUT_DIR / 'milestone2_summary.txt'}")
print("\n" + summary_text)
