"""Pytest configuration and shared fixtures for Solar-Pulse."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure ROOT is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def plant1_df() -> pd.DataFrame:
    """Loads the real Plant 1 feature dataset."""
    path = ROOT / "data" / "features" / "features_plant1.csv"
    if not path.exists():
        pytest.skip(f"Features file not found: {path}")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Generates a synthetic 10-day hourly solar dataset (240 hours).
    
    Daily sine wave for daytime hours, strict zeros at night, with full feature schema.
    """
    idx = pd.date_range("2020-05-01 00:00:00", periods=240, freq="h", name="timestamp")
    hours = idx.hour.values

    # Daytime mask: hours 6 to 18 inclusive
    is_day = (hours >= 6) & (hours <= 18)
    day_fraction = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 12.0))
    power = np.where(is_day, day_fraction * 850.0, 0.0)
    irr = np.where(is_day, day_fraction * 0.95, 0.0)
    mod_temp = 22.0 + day_fraction * 28.0
    amb_temp = 20.0 + day_fraction * 12.0

    df = pd.DataFrame(
        {
            "ac_power_kw": power,
            "irradiation": irr,
            "module_temperature": mod_temp,
            "ambient_temperature": amb_temp,
            "delta_T": mod_temp - amb_temp,
            "hour": hours,
            "is_daylight": is_day.astype(int),
            "ghi_clear": np.where(is_day, day_fraction * 1.0, 0.0),
            "clearness_index": np.where(is_day, 0.95, 0.0),
            "hour_sin": np.sin(2 * np.pi * hours / 24.0),
            "hour_cos": np.cos(2 * np.pi * hours / 24.0),
            "flag_low_coverage": 0,
            "inverters_reporting": 22.0,
            "slots_present": 4,
            "ac_imputed": 0,
            "wx_imputed": 0,
            "train_mask": 1,
        },
        index=idx,
    )

    # Autoregressive lags & dynamics
    df["ac_power_lag_1"] = df["ac_power_kw"].shift(1).fillna(0.0)
    df["ac_power_lag_2"] = df["ac_power_kw"].shift(2).fillna(0.0)
    df["ac_power_lag_24"] = df["ac_power_kw"].shift(24).fillna(0.0)
    df["irradiation_lag_1"] = df["irradiation"].shift(1).fillna(0.0)
    df["irradiation_lag_24"] = df["irradiation"].shift(24).fillna(0.0)
    df["ac_power_roll_3h_mean"] = (
        df["ac_power_kw"].shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
    )
    df["irradiation_roll_3h_std"] = (
        df["irradiation"].shift(1).rolling(3, min_periods=1).std().fillna(0.0)
    )

    # First 24 rows have lag_24 unavailable -> train_mask = 0
    df.iloc[:24, df.columns.get_loc("train_mask")] = 0

    # Split markers: 6 days train (144h), 2 days val (48h), 2 days test (48h)
    df["split"] = "train"
    df.iloc[144:192, df.columns.get_loc("split")] = "val"
    df.iloc[192:, df.columns.get_loc("split")] = "test"

    return df
