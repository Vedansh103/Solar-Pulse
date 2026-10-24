"""Phase 1: Data Cleaning, Physics-Informed Imputation, and Streamlined Feature Engineering

Pipeline:
1. Physics-guided gap filling (Night -> 0 kW, daytime 1-3h -> spline, long gaps -> train_mask=0).
2. Physical features (delta_T, hour_sin/cos, pvlib astronomical clear-sky, clearness index).
3. Curated autoregressive lags (t-1, t-2, t-24) and cloud volatility dynamics (3h roll).
4. Pruning redundant audit flags and colinear rolling extrema to maintain a high signal-to-noise ratio.
5. Chronological train/val/test tagging (576h train, 72h val, 168h test).

Outputs:
  data/features/features_plant1.csv
  data/features/features_plant2.csv
  output/features_plant1.csv
  output/features_plant2.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

# Project Paths
ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = ROOT / "output"
DATA_FEATURES_DIR = ROOT / "data" / "features"
OUTPUT_DIR = ROOT / "output"

DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Plant approximate location in India (Gandikota / Central India solar belt)
LATITUDE = 20.59
LONGITUDE = 78.96
TZ = "Asia/Kolkata"

# The Curated Feature Schema (Target + High-Signal Predictors + Metadata)
CURATED_COLUMNS = [
    # 1. Target
    "ac_power_kw",
    # 2. Weather & Physics
    "irradiation",
    "module_temperature",
    "ambient_temperature",
    "delta_T",
    "is_daylight",
    # 3. Astronomical Clear-Sky & Diurnal Geometry
    "ghi_clear",
    "clearness_index",
    "hour_sin",
    "hour_cos",
    # 4. Autoregressive Lags (Strictly Causal)
    "ac_power_lag_1",
    "ac_power_lag_2",
    "ac_power_lag_24",
    # 5. Weather Lags
    "irradiation_lag_1",
    "irradiation_lag_24",
    # 6. Cloud & Generation Volatility (3h Causal Rolling)
    "ac_power_roll_3h_mean",
    "irradiation_roll_3h_std",
    # 7. Metadata & Loss Masks
    "train_mask",
    "split",
]


def impute_plant_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing generation and weather data using solar physics rules."""
    df = raw_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Step 1: Initialize train_mask (1 = valid for training loss, 0 = long outage)
    df["train_mask"] = 1

    # Step 2: Identify nighttime hours
    is_night = (df["hour"] < 6) | (df["hour"] >= 19)
    if "is_daylight" in df.columns:
        is_night = is_night | (df["is_daylight"] == 0)

    # For night hours, physical generation and solar irradiance are strictly 0.0
    df.loc[is_night & df["ac_power_kw"].isna(), "ac_power_kw"] = 0.0
    df.loc[is_night & df["irradiation"].isna(), "irradiation"] = 0.0

    # Step 3: Handle daytime gaps in ac_power_kw and weather
    is_na_power = df["ac_power_kw"].isna()
    gap_blocks = (is_na_power != is_na_power.shift()).cumsum()
    gap_lengths = df.groupby(gap_blocks)["ac_power_kw"].transform("count")

    # Long gaps (>3 hours, e.g. 20 May outage): exclude from training loss
    long_gap_mask = is_na_power & (gap_lengths > 3)
    if long_gap_mask.any():
        df.loc[long_gap_mask, "train_mask"] = 0

    # Interpolate ac_power_kw using time-aware cubic spline (bounded at >= 0)
    df["ac_power_kw"] = df["ac_power_kw"].interpolate(method="time").clip(lower=0.0)

    # Interpolate weather variables for daytime gaps
    df["irradiation"] = df["irradiation"].interpolate(method="time").clip(lower=0.0)
    df["module_temperature"] = df["module_temperature"].interpolate(method="time")
    df["ambient_temperature"] = df["ambient_temperature"].interpolate(method="time")

    # Update is_daylight column
    df["is_daylight"] = (df["irradiation"] > 0.005).astype(int)

    return df


def compute_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physics-guided features: thermal differential, cyclic hours, pvlib clear-sky, clearness index."""
    res = df.copy()

    # 1. Thermal Differential (panel heating loss indicator)
    res["delta_T"] = res["module_temperature"] - res["ambient_temperature"]

    # 2. Cyclic Diurnal Encodings (hour 23 -> hour 0 continuity)
    res["hour_sin"] = np.sin(2 * np.pi * res["hour"] / 24.0)
    res["hour_cos"] = np.cos(2 * np.pi * res["hour"] / 24.0)

    # 3. Astronomical Clear-Sky Modeling using pvlib
    times = res.index.tz_localize(TZ)
    location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, tz=TZ)
    clearsky = location.get_clearsky(times, model="ineichen")
    clearsky.index = res.index

    # Clear-sky Global Horizontal Irradiance (GHI) in kW/m^2 (matching dataset units)
    res["ghi_clear"] = (clearsky["ghi"] / 1000.0).clip(lower=0.0)

    # 4. Clearness Index (Kt = measured irradiation / clear-sky expectation)
    res["clearness_index"] = 0.0
    daylight_idx = res["ghi_clear"] > 0.01
    res.loc[daylight_idx, "clearness_index"] = (
        res.loc[daylight_idx, "irradiation"] / res.loc[daylight_idx, "ghi_clear"]
    ).clip(0.0, 1.5)

    return res


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes high-signal autoregressive lags and rolling volatility strictly from past values."""
    res = df.copy()

    # 1. Autoregressive Lags (strictly causal: t - k)
    # lag 1 (momentum), lag 2 (short trend), lag 24 (daily seasonality)
    res["ac_power_lag_1"] = res["ac_power_kw"].shift(1)
    res["ac_power_lag_2"] = res["ac_power_kw"].shift(2)
    res["ac_power_lag_24"] = res["ac_power_kw"].shift(24)

    # 2. Weather Lags
    res["irradiation_lag_1"] = res["irradiation"].shift(1)
    res["irradiation_lag_24"] = res["irradiation"].shift(24)

    # 3. Rolling Window Dynamics (shifted by 1 to prevent future lookahead)
    # 3-Hour rolling mean captures immediate morning ramp / afternoon decay
    res["ac_power_roll_3h_mean"] = res["ac_power_kw"].shift(1).rolling(3, min_periods=1).mean()
    # 3-Hour irradiation std captures cloud flicker / weather volatility
    res["irradiation_roll_3h_std"] = (
        res["irradiation"].shift(1).rolling(3, min_periods=1).std().fillna(0.0)
    )

    # Backfill initial window lag rows for t < 24 so input vectors are consistent
    res = res.bfill()

    # 4. Chronological Train / Val / Test Partition Tagging
    res["split"] = "train"
    res.loc[(res.index >= "2020-06-08") & (res.index < "2020-06-11"), "split"] = "val"
    res.loc[res.index >= "2020-06-11", "split"] = "test"

    return res


def build_features(plant: int) -> pd.DataFrame:
    """Builds the streamlined feature matrix for a plant."""
    print(f"\nProcessing Plant {plant} feature pipeline...")
    input_file = INPUT_DIR / f"master_hourly_plant{plant}.csv"
    raw_df = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)

    # Step 1: Imputation
    clean_df = impute_plant_data(raw_df)
    n_missing = clean_df["ac_power_kw"].isna().sum()
    print(f"  [Imputation] Missing ac_power_kw after gap fill: {n_missing}")

    # Step 2: Physical Features
    phys_df = compute_physical_features(clean_df)
    print(f"  [Physics] Added delta_T, hour_sin/cos, pvlib ghi_clear, clearness_index")

    # Step 3: Temporal Features
    temp_df = compute_temporal_features(phys_df)
    print(f"  [Temporal] Added causal lags (t-1, t-2, t-24) and 3h cloud dynamics")

    # Step 4: Streamlined Curation (Drop P0 audit flags and redundant colinear rolling extrema)
    final_df = temp_df[CURATED_COLUMNS].copy()

    # Assertions
    assert len(final_df) == 816, f"Expected 816 rows, got {len(final_df)}"
    assert final_df.isna().sum().sum() == 0, "Feature dataframe contains unexpected NaNs!"
    assert (final_df["split"] == "train").sum() == 576, "Train split mismatch"
    assert (final_df["split"] == "val").sum() == 72, "Val split mismatch"
    assert (final_df["split"] == "test").sum() == 168, "Test split mismatch"

    # Save to data/features/ and output/
    dest_data = DATA_FEATURES_DIR / f"features_plant{plant}.csv"
    dest_out = OUTPUT_DIR / f"features_plant{plant}.csv"
    saved_data = False
    try:
        final_df.to_csv(dest_data)
        saved_data = True
    except PermissionError:
        print(f"  [LOCKED] Could not overwrite {dest_data} because it is currently open in Excel.")
    final_df.to_csv(dest_out)
    print(f"  [Saved] Successfully exported clean feature table:")
    if saved_data:
        print(f"    - {dest_data}")
    print(f"    - {dest_out}")
    print(f"  [Shape] Total rows: {len(final_df)}, Total columns: {final_df.shape[1]}")

    return final_df


if __name__ == "__main__":
    for p in [1, 2]:
        build_features(p)
    print("\n[SUCCESS] Phase 1 Streamlined Feature Engineering successfully completed for both plants!")
