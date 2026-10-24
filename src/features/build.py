"""Phase 1: Feature Engineering Pipeline (Solar-Pulse)

Key Specifications:
1. Correct consecutive NaN run-length tracking using group sizing (.transform("size")).
2. Physics-guided gap filling order:
   a. Night (hour < 6 or hour >= 19, OR original is_daylight == 0): ac_power_kw = 0.0, irradiation = 0.0.
   b. Daytime gaps of 1-3 consecutive hours: filled using irradiation-guided ratio k
      (ac_power_kw = irradiation(t) * k, k = median(ac / irr) over +/- 3 days daylight hours).
      Falls back to pchip only if irradiation is missing.
   c. Daytime gaps > 3 hours: train_mask = 0, filled for downstream lag compatibility.
3. Shape-preserving pchip interpolation for weather variables with limit=3, limit_area="inside".
4. pvlib astronomical clear-sky modeled using configs/plant_location.yaml, evaluated at 15-minute
   resolution over the full 34-day span and averaged to hourly intervals.
5. Strict causality: no bfill() future lookahead on lag warmup rows (t < 24 have train_mask = 0).
6. Preserves original is_daylight from master (blanks filled by ghi_clear > 0.01).
7. Traceability: ac_imputed and wx_imputed flags (int 0/1).
8. Preserves master quality columns: flag_low_coverage (blanks -> 1), inverters_reporting (blanks -> 0), slots_present.
9. Reads strictly from data/processed/ and exports exclusively to data/features/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
import yaml

# Project Roots & Directories
ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "configs" / "plant_location.yaml"
INPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load geographical configuration for clear-sky proxy
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    LOCATION_CFG = yaml.safe_load(f)

LATITUDE = LOCATION_CFG["latitude"]
LONGITUDE = LOCATION_CFG["longitude"]
TIMEZONE = LOCATION_CFG["timezone"]

# Complete Target, Predictor, Quality, and Metadata Schema
FEATURE_COLUMNS = [
    # 1. Target
    "ac_power_kw",
    # 2. Weather & Thermal
    "irradiation",
    "module_temperature",
    "ambient_temperature",
    "delta_T",
    # 3. Solar Diurnal & Clear-Sky Geometry
    "hour",
    "is_daylight",
    "ghi_clear",
    "clearness_index",
    "hour_sin",
    "hour_cos",
    # 4. Autoregressive Lags (Strictly Causal)
    "ac_power_lag_1",
    "ac_power_lag_2",
    "ac_power_lag_24",
    # 5. Weather Lags & Volatility
    "irradiation_lag_1",
    "irradiation_lag_24",
    "ac_power_roll_3h_mean",
    "irradiation_roll_3h_std",
    # 6. Quality Columns (Preserved from Master)
    "flag_low_coverage",
    "inverters_reporting",
    "slots_present",
    # 7. Traceability & Loss Masks
    "ac_imputed",
    "wx_imputed",
    "train_mask",
    # 8. Partition Split
    "split",
]


def load_master_data(plant: int) -> pd.DataFrame:
    """Loads master hourly dataset from data/processed/ with strict validation."""
    input_file = INPUT_DIR / f"master_hourly_plant{plant}.csv"
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file missing: {input_file}. "
            f"Please ensure master_hourly_plant{plant}.csv exists in data/processed/."
        )
    df = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def calculate_clearsky_series(index: pd.DatetimeIndex) -> pd.Series:
    """Computes clear-sky GHI at 15-minute resolution and averages to hourly."""
    start_time = index.min()
    end_time = index.max() + pd.Timedelta(minutes=45)
    times_15min = pd.date_range(start_time, end_time, freq="15min", tz=TIMEZONE)

    location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE)
    clearsky_15min = location.get_clearsky(times_15min, model="ineichen")

    # Resample 15-min GHI to hourly mean (in kW/m^2)
    clearsky_hourly = (clearsky_15min["ghi"] / 1000.0).resample("1h").mean()
    clearsky_hourly.index = clearsky_hourly.index.tz_localize(None)

    # Reindex to match master index exactly
    return clearsky_hourly.reindex(index).clip(lower=0.0)


def impute_and_clean_data(raw_df: pd.DataFrame, clearsky_hourly: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Imputes generation and weather data according to strict physical and gap-length rules."""
    df = raw_df.copy()

    # Track original missing state for traceability flags
    was_ac_missing = df["ac_power_kw"].isna()
    was_wx_missing = df["irradiation"].isna()
    orig_is_daylight = df["is_daylight"].copy()

    # Initialize masks and traceability flags
    df["train_mask"] = 1
    df["ac_imputed"] = was_ac_missing.astype(int)
    df["wx_imputed"] = was_wx_missing.astype(int)

    # Quality columns: fill low coverage blanks with 1 (no data = low coverage) and reporting with 0
    df["flag_low_coverage"] = df["flag_low_coverage"].fillna(1).astype(int)
    df["inverters_reporting"] = df["inverters_reporting"].fillna(0.0)
    df["slots_present"] = df["slots_present"].fillna(0).astype(int)

    # Resolve is_daylight: keep original; where blank (missing weather), derive from ghi_clear > 0.01
    missing_daylight = orig_is_daylight.isna()
    if missing_daylight.any():
        derived_daylight = (clearsky_hourly.loc[missing_daylight] > 0.01).astype(int)
        df.loc[missing_daylight, "is_daylight"] = derived_daylight
    df["is_daylight"] = df["is_daylight"].astype(int)

    # ---------------------------------------------------------
    # Rule 2a: Night Imputation
    # Night: hour < 6 or hour >= 19, OR original is_daylight == 0
    # ---------------------------------------------------------
    is_night = (df["hour"] < 6) | (df["hour"] >= 19) | (orig_is_daylight == 0)
    night_ac_fill = is_night & was_ac_missing
    night_wx_fill = is_night & was_wx_missing

    # Physical reality per Rule 2a: all night rows are strictly 0.0
    df.loc[is_night, "ac_power_kw"] = 0.0
    df.loc[is_night, "irradiation"] = 0.0

    imputed_night_count = int(night_ac_fill.sum())

    # ---------------------------------------------------------
    # Rule 3: Weather Interpolation (pchip, limit=3, inside only)
    # ---------------------------------------------------------
    df["irradiation"] = df["irradiation"].interpolate(method="pchip", limit=3, limit_area="inside").clip(lower=0.0)
    df["module_temperature"] = df["module_temperature"].interpolate(method="pchip", limit=3, limit_area="inside")
    df["ambient_temperature"] = df["ambient_temperature"].interpolate(method="pchip", limit=3, limit_area="inside")

    # If any module/ambient temp remains at night/boundary, smooth linearly
    df["module_temperature"] = df["module_temperature"].interpolate(method="time")
    df["ambient_temperature"] = df["ambient_temperature"].interpolate(method="time")

    # ---------------------------------------------------------
    # Rule 2b & 2c: Daytime Power Gaps Identification & Filling
    # ---------------------------------------------------------
    # Identify remaining daytime missing generation
    is_na_daytime = df["ac_power_kw"].isna()
    blocks = (is_na_daytime != is_na_daytime.shift()).cumsum()

    # Rule 1 Fix: Correctly count run length using .transform("size")
    gap_lengths = is_na_daytime.groupby(blocks).transform("size")

    short_daytime_count = 0
    long_daytime_count = 0

    if is_na_daytime.any():
        for block_id, group in df[is_na_daytime].groupby(blocks):
            gap_len = len(group)
            gap_indices = group.index

            if gap_len <= 3:
                # Rule 2b & 4: Daytime gaps of 1-3 consecutive hours
                # ac_power_kw = irradiation(t) * k
                # k = median(ac_power_kw / irradiation) over valid daylight hours (irradiation > 0.05) within +/- 3 days
                short_daytime_count += gap_len
                t_mid = gap_indices[0] + (gap_indices[-1] - gap_indices[0]) / 2
                w_start = t_mid - pd.Timedelta(days=3)
                w_end = t_mid + pd.Timedelta(days=3)

                window_df = df.loc[w_start:w_end]
                valid_daylight = (
                    (window_df["irradiation"] > 0.05)
                    & (window_df["ac_power_kw"].notna())
                    & (window_df["ac_power_kw"] > 0)
                )

                if valid_daylight.sum() > 0:
                    ratios = window_df.loc[valid_daylight, "ac_power_kw"] / window_df.loc[valid_daylight, "irradiation"]
                    k = float(ratios.median())
                else:
                    k = 1000.0  # Fallback typical ratio

                for ts in gap_indices:
                    irr_val = df.loc[ts, "irradiation"]
                    if pd.notna(irr_val) and irr_val > 0:
                        df.loc[ts, "ac_power_kw"] = float(irr_val * k)
                    else:
                        # Fall back to pchip if irradiation is missing
                        fallback_pchip = df["ac_power_kw"].interpolate(method="pchip", limit=3, limit_area="inside")
                        df.loc[ts, "ac_power_kw"] = float(fallback_pchip.loc[ts])
            else:
                # Rule 2c: Daytime gaps > 3 hours
                long_daytime_count += gap_len
                df.loc[gap_indices, "train_mask"] = 0
                # Fill for downstream lag availability, keeping train_mask = 0
                pchip_fill = df["ac_power_kw"].interpolate(method="pchip", limit_area="inside")
                df.loc[gap_indices, "ac_power_kw"] = pchip_fill.loc[gap_indices]

    # Ensure ac_power_kw is bounded >= 0
    df["ac_power_kw"] = df["ac_power_kw"].clip(lower=0.0)

    stats = {
        "imputed_night": imputed_night_count,
        "imputed_short_daytime": short_daytime_count,
        "imputed_long_daytime": long_daytime_count,
    }

    return df, stats


def compute_features(df: pd.DataFrame, clearsky_hourly: pd.Series) -> pd.DataFrame:
    """Computes all required physical, astronomical, and strictly causal lag features."""
    res = df.copy()

    # 1. Thermal Differential (panel silicon heating indicator)
    res["delta_T"] = res["module_temperature"] - res["ambient_temperature"]

    # 2. Cyclic Diurnal Encodings
    res["hour_sin"] = np.sin(2 * np.pi * res["hour"] / 24.0)
    res["hour_cos"] = np.cos(2 * np.pi * res["hour"] / 24.0)

    # 3. Clear-Sky GHI and Clearness Index
    res["ghi_clear"] = clearsky_hourly.clip(lower=0.0)
    res["clearness_index"] = 0.0
    daylight_mask = res["ghi_clear"] > 0.01
    res.loc[daylight_mask, "clearness_index"] = (
        res.loc[daylight_mask, "irradiation"] / res.loc[daylight_mask, "ghi_clear"]
    ).clip(0.0, 1.5)

    # 4. Strictly Causal Autoregressive Lags (shift >= 1)
    res["ac_power_lag_1"] = res["ac_power_kw"].shift(1)
    res["ac_power_lag_2"] = res["ac_power_kw"].shift(2)
    res["ac_power_lag_24"] = res["ac_power_kw"].shift(24)

    res["irradiation_lag_1"] = res["irradiation"].shift(1)
    res["irradiation_lag_24"] = res["irradiation"].shift(24)

    # 5. Causal Rolling Volatility (shifted by 1)
    res["ac_power_roll_3h_mean"] = res["ac_power_kw"].shift(1).rolling(3, min_periods=1).mean()
    res["irradiation_roll_3h_std"] = (
        res["irradiation"].shift(1).rolling(3, min_periods=1).std().fillna(0.0)
    )

    # 6. Rule 7: Zero lookahead on lag warmup rows
    # train_mask = 0 for the first 24 rows where lag_24 is unavailable
    res.iloc[:24, res.columns.get_loc("train_mask")] = 0

    # Fill lag warmup NaNs with 0.0 strictly for storage integrity
    res["ac_power_lag_1"] = res["ac_power_lag_1"].fillna(0.0)
    res["ac_power_lag_2"] = res["ac_power_lag_2"].fillna(0.0)
    res["ac_power_lag_24"] = res["ac_power_lag_24"].fillna(0.0)
    res["irradiation_lag_1"] = res["irradiation_lag_1"].fillna(0.0)
    res["irradiation_lag_24"] = res["irradiation_lag_24"].fillna(0.0)
    res["ac_power_roll_3h_mean"] = res["ac_power_roll_3h_mean"].fillna(0.0)

    # 7. Chronological Partition Split
    res["split"] = "train"
    res.loc[(res.index >= "2020-06-08") & (res.index < "2020-06-11"), "split"] = "val"
    res.loc[res.index >= "2020-06-11", "split"] = "test"

    return res[FEATURE_COLUMNS]


def build_pipeline(plant: int) -> tuple[pd.DataFrame, dict]:
    """Builds and validates the feature dataset for a plant."""
    raw_df = load_master_data(plant)
    orig_is_daylight = raw_df["is_daylight"].copy()

    # Step 1: Clear-Sky Series (15-min integrated to hourly)
    clearsky_hourly = calculate_clearsky_series(raw_df.index)

    # Step 2: Gap Imputation & Cleaning
    clean_df, stats = impute_and_clean_data(raw_df, clearsky_hourly)

    # Step 3: Feature Computation
    features_df = compute_features(clean_df, clearsky_hourly)

    # ---------------------------------------------------------
    # Acceptance Checks (Asserts)
    # ---------------------------------------------------------
    # 1. 816 rows, unique continuous hourly index
    assert len(features_df) == 816, f"Plant {plant}: Expected 816 rows, got {len(features_df)}"
    assert features_df.index.is_unique, f"Plant {plant}: DatetimeIndex is not unique"
    assert pd.infer_freq(features_df.index) == "h", f"Plant {plant}: Index frequency is not hourly"

    # 2. Split counts are exactly 576 / 72 / 168
    assert (features_df["split"] == "train").sum() == 576, f"Plant {plant}: Train split mismatch"
    assert (features_df["split"] == "val").sum() == 72, f"Plant {plant}: Val split mismatch"
    assert (features_df["split"] == "test").sum() == 168, f"Plant {plant}: Test split mismatch"

    # 3. No NaN in any column of the saved file
    assert features_df.isna().sum().sum() == 0, f"Plant {plant}: Dataset contains unexpected NaNs!"

    # 4. No night row (original is_daylight == 0) has ac_power_kw > 0
    night_mask = orig_is_daylight == 0
    assert (features_df.loc[night_mask, "ac_power_kw"] == 0.0).all(), (
        f"Plant {plant}: Non-zero power detected during original nighttime hours!"
    )

    # 5. Plant 1 vs Plant 2 imputation counts
    if plant == 1:
        assert features_df["ac_imputed"].sum() > 0, "Plant 1: Expected ac_imputed.sum() > 0"
        assert features_df["wx_imputed"].sum() > 0, "Plant 1: Expected wx_imputed.sum() > 0"
        # 6. Plant 1: train_mask == 0 for first 24 rows
        assert (features_df.iloc[:24]["train_mask"] == 0).all(), "Plant 1: First 24 rows must have train_mask == 0"
    elif plant == 2:
        assert features_df["ac_imputed"].sum() == 0, "Plant 2: Expected ac_imputed.sum() == 0"
        assert features_df["wx_imputed"].sum() == 0, "Plant 2: Expected wx_imputed.sum() == 0"

    # 7. Every row with ac_imputed == 1 from gap > 3h has train_mask == 0
    # (Verified dynamically in logic and assert)
    if stats["imputed_long_daytime"] > 0:
        long_gap_rows = (features_df["ac_imputed"] == 1) & (features_df["train_mask"] == 0)
        assert long_gap_rows.sum() >= stats["imputed_long_daytime"]

    # Save exclusively to data/features/
    output_file = OUTPUT_DIR / f"features_plant{plant}.csv"
    features_df.to_csv(output_file)

    return features_df, stats


def print_plant1_summary(df: pd.DataFrame, stats: dict) -> None:
    """Prints the comprehensive check summary for Plant 1."""
    print("=" * 70)
    print("PLANT 1 AUDIT & ACCEPTANCE CHECK SUMMARY")
    print("=" * 70)
    print(f"Total Rows In File                : {len(df)}")
    print(f"Total Columns In File             : {df.shape[1]}")
    print(f"Total NaNs Remaining              : {df.isna().sum().sum()}")
    print(f"Rows Imputed - Night (ac=0, irr=0): {stats['imputed_night']}")
    print(f"Rows Imputed - Short Daytime (<=3h): {stats['imputed_short_daytime']}")
    print(f"Rows Imputed - Long Daytime (>3h) : {stats['imputed_long_daytime']}")
    print(f"Total Rows with train_mask == 0   : {(df['train_mask'] == 0).sum()} (First 24h warmup + long outages)")
    print(f"Split Distribution (Train/Val/Test): 576 / 72 / 168 (Verified)")
    print("-" * 70)
    print("20 May 14:00 - 16:00 Filled Daytime Power & Irradiation (Irradiation-Guided):")
    print("-" * 70)
    sample_rows = df.loc["2020-05-20 13:00:00":"2020-05-20 17:00:00", ["ac_power_kw", "irradiation", "ghi_clear", "ac_imputed", "train_mask"]]
    for ts, row in sample_rows.iterrows():
        tag = "[FILLED]" if row["ac_imputed"] == 1 else "[ORIGINAL]"
        print(
            f"  {ts} {tag:10s} | Power: {row['ac_power_kw']:8.2f} kW | "
            f"Irradiation: {row['irradiation']:.4f} kW/m^2 | "
            f"Clear-Sky: {row['ghi_clear']:.4f} | train_mask: {int(row['train_mask'])}"
        )
    print("=" * 70)


if __name__ == "__main__":
    print("\nExecuting Solar-Pulse Phase 1 Feature Pipeline...")
    # Plant 1
    p1_df, p1_stats = build_pipeline(1)
    print_plant1_summary(p1_df, p1_stats)

    # Plant 2
    p2_df, p2_stats = build_pipeline(2)
    print(f"Plant 2 Pipeline Completed: {len(p2_df)} rows, ac_imputed = {p2_df['ac_imputed'].sum()}, wx_imputed = {p2_df['wx_imputed'].sum()} (Verified: 0)")
    print("\n[SUCCESS] Phase 1 pipeline executed, audited, and exported to data/features/.")
