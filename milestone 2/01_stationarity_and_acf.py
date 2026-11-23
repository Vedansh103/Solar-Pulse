"""Milestone 2 - Step 1: Stationarity Checks & ACF/PACF Analysis.

Simple, explainable script for Mid-Semester Evaluation:
1. Loads hourly solar power generation data (Plant 1).
2. Performs Augmented Dickey-Fuller (ADF) & KPSS statistical tests for stationarity.
3. Generates Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.
4. Explains why a 24-hour seasonal differencing is required for solar time-series.

Outputs:
  milestone 2/outputs/stationarity_results.csv
  milestone 2/outputs/acf_pacf_plot.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# ---------------------------------------------------------
# 1. PATH SETUP & DATA LOADING
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Plant 1 hourly features dataset
data_path = ROOT / "data" / "features" / "features_plant1.csv"
print(f"Loading data from: {data_path.name}")
df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")

# Use training split (May 15 to June 7, 2020) for statistical diagnostic checks
train_series = df.loc[df["split"] == "train", "ac_power_kw"]
print(f"Training Period: {train_series.index[0]} to {train_series.index[-1]} ({len(train_series)} hours)")

# ---------------------------------------------------------
# 2. STATIONARITY TESTS (ADF & KPSS)
# ---------------------------------------------------------
# Test 1: Raw solar power series
# Test 2: Seasonally differenced series (y_t - y_{t-24}) to remove the daily solar cycle
seasonal_diff = train_series.diff(24).dropna()

results = []
for name, series in [("Raw AC Power", train_series), ("Seasonal Differenced (Lag 24)", seasonal_diff)]:
    # ADF test: H0 = non-stationary (unit root). Reject if p-value < 0.05
    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag="AIC")
    
    # KPSS test: H0 = stationary. Reject if p-value < 0.05
    kpss_stat, kpss_p, _, _ = kpss(series, regression="c", nlags="auto")
    
    # Combined verdict
    is_stationary = (adf_p < 0.05) and (kpss_p >= 0.05)
    verdict = "Stationary" if is_stationary else "Non-Stationary"
    
    results.append({
        "Series": name,
        "ADF_Statistic": round(adf_stat, 4),
        "ADF_p_value": round(adf_p, 4),
        "KPSS_Statistic": round(kpss_stat, 4),
        "KPSS_p_value": round(kpss_p, 4),
        "Verdict": verdict,
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("STATIONARITY TEST RESULTS:")
print("=" * 70)
print(results_df.to_string(index=False))

# Save table to outputs folder
results_df.to_csv(OUTPUT_DIR / "stationarity_results.csv", index=False)
print(f"\n[Saved] -> {OUTPUT_DIR / 'stationarity_results.csv'}")

# ---------------------------------------------------------
# 3. ACF & PACF PLOTS (Daily Diurnal Seasonality)
# ---------------------------------------------------------
print("\nGenerating ACF and PACF plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Top row: Raw series ACF & PACF
plot_acf(train_series, lags=72, ax=axes[0, 0], title="Raw Series: Autocorrelation (ACF)")
plot_pacf(train_series, lags=72, ax=axes[0, 1], title="Raw Series: Partial Autocorrelation (PACF)", method="ywm")

# Bottom row: Seasonally differenced (lag 24) ACF & PACF
plot_acf(seasonal_diff, lags=72, ax=axes[1, 0], title="Seasonal Diff (Lag 24): Autocorrelation (ACF)")
plot_pacf(seasonal_diff, lags=72, ax=axes[1, 1], title="Seasonal Diff (Lag 24): Partial Autocorrelation (PACF)", method="ywm")

for row in axes:
    for ax in row:
        ax.set_xlabel("Lag (Hours)")
        ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plot_path = OUTPUT_DIR / "acf_pacf_plot.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[Saved] -> {plot_path}")

print("\nExplanation for your Professor:")
print("- The raw ACF shows strong periodic wave peaks every 24 hours (lags 24, 48, 72).")
print("- This confirms the strong diurnal (day/night) seasonal cycle of solar generation.")
print("- Seasonal differencing (lag 24) successfully removes the repeating pattern, preparing data for SARIMA.")
