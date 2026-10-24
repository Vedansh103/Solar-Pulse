"""Feature engineering per forecast horizon for tree models (Phase 3).

Builds causal feature matrices dynamically from data/features/features_plantX.csv.
Never edits the input file on disk.

Feature Sets:
1. '1h_no_weather': recent momentum lags and rolling dynamics (no contemporaneous weather).
2. '1h_weather': set 1 + measured weather at forecast hour (upper bound benchmark).
3. '24h_no_weather': strictly day-ahead information (lags >= 24h, no lag_1/2 or roll_3h).
4. '24h_weather': set 3 + measured weather at forecast hour (upper bound benchmark).
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# The 4 strictly defined feature sets
FEATURE_SETS: dict[str, list[str]] = {
    "1h_no_weather": [
        "hour_sin",
        "hour_cos",
        "ghi_clear",
        "ac_power_lag_1",
        "ac_power_lag_2",
        "ac_power_lag_24",
        "ac_power_lag_48",
        "ac_power_roll_3h_mean",
        "irradiation_lag_1",
        "irradiation_lag_24",
        "irradiation_roll_3h_std",
        "same_hour_mean_3d",
    ],
    "1h_weather": [
        "hour_sin",
        "hour_cos",
        "ghi_clear",
        "ac_power_lag_1",
        "ac_power_lag_2",
        "ac_power_lag_24",
        "ac_power_lag_48",
        "ac_power_roll_3h_mean",
        "irradiation_lag_1",
        "irradiation_lag_24",
        "irradiation_roll_3h_std",
        "same_hour_mean_3d",
        "irradiation",
        "module_temperature",
        "ambient_temperature",
        "delta_T",
        "clearness_index",
    ],
    "24h_no_weather": [
        "hour_sin",
        "hour_cos",
        "ghi_clear",
        "ac_power_lag_24",
        "ac_power_lag_48",
        "irradiation_lag_24",
        "irradiation_lag_48",
        "same_hour_mean_3d",
    ],
    "24h_weather": [
        "hour_sin",
        "hour_cos",
        "ghi_clear",
        "ac_power_lag_24",
        "ac_power_lag_48",
        "irradiation_lag_24",
        "irradiation_lag_48",
        "same_hour_mean_3d",
        "irradiation",
        "module_temperature",
        "ambient_temperature",
        "delta_T",
        "clearness_index",
    ],
}

BANNED_24H_GENERAL = {
    "ac_power_lag_1",
    "ac_power_lag_2",
    "ac_power_roll_3h_mean",
    "irradiation_lag_1",
    "irradiation_roll_3h_std",
}

BANNED_24H_NO_WEATHER = BANNED_24H_GENERAL | {
    "irradiation",
    "module_temperature",
    "ambient_temperature",
    "delta_T",
    "clearness_index",
}


def add_extended_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Computes extra causal lags from past values strictly without editing the source file."""
    res = df.copy()

    # ac_power_lag_48 (2 days ago at same hour)
    res["ac_power_lag_48"] = res["ac_power_kw"].shift(48)

    # irradiation_lag_48 (2 days ago sunlight)
    res["irradiation_lag_48"] = res["irradiation"].shift(48)

    # same_hour_mean_3d: mean of ac_power_kw at t-24, t-48, t-72
    lag_24 = res["ac_power_kw"].shift(24)
    lag_48 = res["ac_power_kw"].shift(48)
    lag_72 = res["ac_power_kw"].shift(72)
    res["same_hour_mean_3d"] = (lag_24 + lag_48 + lag_72) / 3.0

    return res


def get_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generates the feature matrix X and target y for the requested feature set.
    
    Returns: (X, y) with matching index.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Must be one of {list(FEATURE_SETS.keys())}")

    extended_df = add_extended_lags(df)
    columns = FEATURE_SETS[feature_set]
    X = extended_df[columns].copy()
    y = extended_df["ac_power_kw"].copy()

    return X, y


def get_train_val_test_data(
    df: pd.DataFrame,
    feature_set: str,
    include_val_in_train: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extracts strictly filtered train, val, and test partitions.
    
    Training rows filter:
    - split == 'train' (or split in ['train', 'val'] if include_val_in_train is True)
    - train_mask == 1
    - ac_imputed == 0
    - no NaN in chosen features (initial 72 rows naturally drop due to lag_72)
    
    Val and Test partitions:
    - real actuals only or evaluation index as passed.
    """
    X, y = get_feature_matrix(df, feature_set)

    # Base valid filter: no NaNs in features, valid train mask, real actuals
    usable_mask = (
        X.notna().all(axis=1)
        & (df["train_mask"] == 1)
        & (df["ac_imputed"] == 0)
    )

    if include_val_in_train:
        train_split_mask = df["split"].isin(["train", "val"])
    else:
        train_split_mask = df["split"] == "train"

    train_idx = df.index[train_split_mask & usable_mask]
    val_idx = df.index[df["split"] == "val"]
    test_idx = df.index[df["split"] == "test"]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
