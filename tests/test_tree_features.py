"""Unit tests for tree feature engineering in Solar-Pulse."""

import numpy as np
import pandas as pd
import pytest

from src.models.tree_features import (
    FEATURE_SETS,
    BANNED_24H_GENERAL,
    BANNED_24H_NO_WEATHER,
    add_extended_lags,
    get_feature_matrix,
    get_train_val_test_data,
)


def test_extended_lags_correctness(plant1_df):
    """Checks that extended lags (48h, 72h, same_hour_mean_3d) match manual calculations on random rows."""
    df_ext = add_extended_lags(plant1_df)

    # Pick 5 random rows after index 100 (where all lags are well-populated)
    np.random.seed(42)
    valid_indices = range(100, len(plant1_df))
    sample_indices = np.random.choice(valid_indices, size=5, replace=False)

    for idx in sample_indices:
        # Check lag 48
        expected_lag_48 = plant1_df["ac_power_kw"].iloc[idx - 48]
        actual_lag_48 = df_ext["ac_power_lag_48"].iloc[idx]
        assert np.isclose(actual_lag_48, expected_lag_48), f"Mismatch in lag 48 at idx {idx}"

        # Check same_hour_mean_3d: mean of lag 24, lag 48, lag 72
        lag_24 = plant1_df["ac_power_kw"].iloc[idx - 24]
        expected_lag_72 = plant1_df["ac_power_kw"].iloc[idx - 72]
        expected_same_hour_mean = (lag_24 + expected_lag_48 + expected_lag_72) / 3.0
        actual_same_hour_mean = df_ext["same_hour_mean_3d"].iloc[idx]
        assert np.isclose(actual_same_hour_mean, expected_same_hour_mean), f"Mismatch in same_hour_mean_3d at idx {idx}"


def test_banned_columns_not_in_24h_sets():
    """Asserts that 24h feature sets contain absolutely no banned columns."""
    for feature_set_name in ["24h_no_weather", "24h_weather"]:
        features = FEATURE_SETS[feature_set_name]
        for banned in BANNED_24H_GENERAL:
            assert banned not in features, f"Forbidden column '{banned}' found in {feature_set_name}"

    for banned in BANNED_24H_NO_WEATHER:
        assert banned not in FEATURE_SETS["24h_no_weather"], f"Weather column '{banned}' found in 24h_no_weather"


def test_leakage_perturbation(plant1_df):
    """Perturbation test: modify test week future values from noon onward;

    assert 24h features computed from midnight are completely invariant.
    """
    df_orig = add_extended_lags(plant1_df)

    df_mod = plant1_df.copy()
    test_mask = df_mod["split"] == "test"
    test_indices = np.where(test_mask)[0]
    first_test_idx = test_indices[0]

    # Modify values 12 hours into test week
    perturbed_idx = first_test_idx + 12
    df_mod.iloc[perturbed_idx:, df_mod.columns.get_loc("ac_power_kw")] = 9999.0

    df_ext_mod = add_extended_lags(df_mod)

    # Features for the first 12 hours of the test week in 24h sets must be 100% identical
    f_24h = FEATURE_SETS["24h_no_weather"]
    X_orig = df_orig[f_24h].iloc[first_test_idx:perturbed_idx]
    X_mod = df_ext_mod[f_24h].iloc[first_test_idx:perturbed_idx]

    pd.testing.assert_frame_equal(X_orig, X_mod)


def test_training_row_filtering(plant1_df):
    """Asserts training rows never include split == 'test', train_mask == 0, or ac_imputed == 1."""
    for fs_name in FEATURE_SETS:
        X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(plant1_df, fs_name)

        train_indices = X_train.index
        orig_train_rows = plant1_df.loc[train_indices]

        assert (orig_train_rows["split"] == "train").all(), f"Non-train rows in train set for {fs_name}"
        assert (orig_train_rows["train_mask"] == 1).all(), f"train_mask == 0 rows in train set for {fs_name}"
        assert (orig_train_rows["ac_imputed"] == 0).all(), f"ac_imputed == 1 rows in train set for {fs_name}"


def test_no_nans_in_returned_matrices(plant1_df):
    """Ensures no NaNs exist in train, val, or test feature matrices for any feature set."""
    for fs_name in FEATURE_SETS:
        X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(plant1_df, fs_name)
        assert not X_train.isna().any().any(), f"NaN found in X_train for {fs_name}"
        assert not X_val.isna().any().any(), f"NaN found in X_val for {fs_name}"
        assert not X_test.isna().any().any(), f"NaN found in X_test for {fs_name}"
        assert not y_train.isna().any(), f"NaN found in y_train for {fs_name}"
        assert not y_val.isna().any(), f"NaN found in y_val for {fs_name}"
        assert not y_test.isna().any(), f"NaN found in y_test for {fs_name}"
