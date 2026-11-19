"""Unit tests for src/evaluation/metrics.py."""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import metrics_table, score


def test_score_hand_computed():
    """Test RMSE and MAE calculations against hand-computed values.
    
    Errors: actual - pred = [3, -4]
    Mean squared error = (3^2 + (-4)^2) / 2 = 25 / 2 = 12.5 -> RMSE = sqrt(12.5) ~= 3.5355339
    MAE = (|3| + |-4|) / 2 = 7 / 2 = 3.5
    """
    actual = pd.Series([10.0, 6.0])
    pred = pd.Series([7.0, 10.0])  # errors: +3, -4
    mask = pd.Series([True, True])

    rmse, mae = score(actual, pred, mask)
    assert np.isclose(rmse, np.sqrt(12.5), atol=1e-4)
    assert np.isclose(mae, 3.5, atol=1e-4)


def test_ac_imputed_and_daylight_masking():
    """Verify that rows with ac_imputed == 1 are excluded, and daylight metrics only score is_daylight == 1."""
    idx = pd.date_range("2020-05-15 00:00:00", periods=4, freq="h")
    df = pd.DataFrame(
        {
            "ac_power_kw": [0.0, 100.0, 200.0, 300.0],
            "ac_imputed": [0, 0, 1, 0],       # row 2 is imputed -> should be ignored completely
            "is_daylight": [0, 1, 1, 1],      # row 0 is night
        },
        index=idx,
    )
    # Predictions
    pred = pd.Series([0.0, 120.0, 999.0, 280.0], index=idx)
    forecasts = {
        "TestModel": (pred, None),
        "Seasonal naive (t-24)": (pred, None),
    }

    t = metrics_table(df, idx, forecasts)
    model_row = t[(t["model"] == "TestModel") & (t["horizon"] == "1h")].iloc[0]

    # Real rows are idx[0], idx[1], idx[3] (ac_imputed == 0)
    # Errors on real rows: [0 - 0 = 0, 100 - 120 = -20, 300 - 280 = 20]
    # Expected rmse_all = sqrt((0^2 + 20^2 + 20^2)/3) = sqrt(800/3) ~= 16.330
    # Expected mae_all = (0 + 20 + 20)/3 = 40/3 ~= 13.333
    assert np.isclose(model_row["rmse_all"], np.sqrt(800 / 3.0), atol=1e-3)
    assert np.isclose(model_row["mae_all"], 40.0 / 3.0, atol=1e-3)

    # Daylight rows that are real: idx[1], idx[3] (is_daylight == 1 & ac_imputed == 0)
    # Errors on daylight real rows: [-20, 20]
    # Expected rmse_daylight = sqrt((20^2 + 20^2)/2) = 20.0
    # Expected mae_daylight = 20.0
    assert np.isclose(model_row["rmse_daylight"], 20.0, atol=1e-3)
    assert np.isclose(model_row["mae_daylight"], 20.0, atol=1e-3)


def test_negative_predictions_clipped():
    """Verify that negative predictions are clipped to 0.0 before scoring."""
    idx = pd.date_range("2020-05-15 00:00:00", periods=2, freq="h")
    df = pd.DataFrame(
        {
            "ac_power_kw": [0.0, 50.0],
            "ac_imputed": [0, 0],
            "is_daylight": [0, 1],
        },
        index=idx,
    )
    # Negative predictions
    pred_negative = pd.Series([-50.0, -20.0], index=idx)
    # Clipped predictions become [0.0, 0.0]
    # Errors against actual [0.0, 50.0]: [0.0, 50.0]
    # RMSE = sqrt((0 + 2500)/2) = sqrt(1250) ~= 35.355
    forecasts = {
        "NegModel": (pred_negative, None),
        "Seasonal naive (t-24)": (pred_negative, None),
    }

    t = metrics_table(df, idx, forecasts)
    row = t[(t["model"] == "NegModel") & (t["horizon"] == "1h")].iloc[0]
    assert np.isclose(row["rmse_all"], np.sqrt(1250.0), atol=1e-3)


def test_seasonal_naive_skill_against_itself():
    """Skill of seasonal naive against itself must be exactly 0.0."""
    idx = pd.date_range("2020-05-15 00:00:00", periods=24, freq="h")
    df = pd.DataFrame(
        {
            "ac_power_kw": np.random.uniform(0, 500, size=24),
            "ac_imputed": 0,
            "is_daylight": 1,
        },
        index=idx,
    )
    pred_sn = pd.Series(np.random.uniform(0, 500, size=24), index=idx)
    forecasts = {
        "Seasonal naive (t-24)": (pred_sn, pred_sn),
    }
    t = metrics_table(df, idx, forecasts)
    for _, r in t.iterrows():
        assert np.isclose(r["skill_vs_seasonal_naive"], 0.0, atol=1e-5)


def test_regression_phase2_seasonal_naive(plant1_df):
    """Regression check: recompute Phase 2 seasonal naive on Plant 1 test week.
    
    Must reproduce: RMSE == 132.616 (+-0.001) and MAE == 67.870 (+-0.001)
    matching reports/phase2/plant1/metrics_test.csv exactly.
    """
    test_idx = plant1_df[plant1_df["split"] == "test"].index
    # Seasonal naive forecast is ac_power_lag_24
    sn_forecast = plant1_df.loc[test_idx, "ac_power_lag_24"]

    forecasts = {
        "Seasonal naive (t-24)": (sn_forecast, sn_forecast),
    }
    t = metrics_table(plant1_df, test_idx, forecasts)
    row = t[(t["model"] == "Seasonal naive (t-24)") & (t["horizon"] == "1h")].iloc[0]

    assert np.isclose(row["rmse_all"], 132.616, atol=0.001), f"Expected 132.616, got {row['rmse_all']}"
    assert np.isclose(row["mae_all"], 67.870, atol=0.001), f"Expected 67.870, got {row['mae_all']}"

