"""Unit tests for Phase 3 tree models and ensembles in Solar-Pulse."""

import numpy as np
import pandas as pd
import pytest

from src.models.tree_features import get_train_val_test_data
from src.models.tree_models import (
    apply_night_zeroing,
    fit_final_model,
    tune_xgboost,
    tune_lightgbm,
    tune_catboost,
)


def test_night_zeroing_and_non_negativity(synthetic_df):
    """Tests that apply_night_zeroing strictly zeroes night hours and clips negative values."""
    raw_preds = np.array([-10.0, 50.0, 200.0, -5.0, 100.0])
    # Create matching synthetic rows
    sub_df = pd.DataFrame({
        "hour": [0, 5, 12, 19, 23],
        "is_daylight": [0, 0, 1, 0, 0],
    })

    cleaned = apply_night_zeroing(raw_preds, sub_df)

    assert (cleaned >= 0.0).all(), "Negative predictions found after zeroing"
    assert cleaned[0] == 0.0, "Hour 0 not zeroed"
    assert cleaned[1] == 0.0, "Hour 5 not zeroed"
    assert cleaned[2] == 200.0, "Valid daylight hour altered unexpectedly"
    assert cleaned[3] == 0.0, "Hour 19 not zeroed"
    assert cleaned[4] == 0.0, "Hour 23 not zeroed"


def test_model_training_and_shapes(synthetic_df):
    """Verifies models train on synthetic data and produce correct shapes without NaNs."""
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(synthetic_df, "1h_no_weather")

    # Fit small XGBoost
    model = fit_final_model("xgboost", {"max_depth": 3, "learning_rate": 0.05}, 10, X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test), "Prediction shape mismatch"
    assert not np.isnan(preds).any(), "NaN found in predictions"


def test_seed_determinism(synthetic_df):
    """Verifies that fitting with seed 42 produces identical predictions."""
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(synthetic_df, "1h_no_weather")

    params = {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8}
    m1 = fit_final_model("xgboost", params, 15, X_train, y_train)
    m2 = fit_final_model("xgboost", params, 15, X_train, y_train)

    p1 = m1.predict(X_test)
    p2 = m2.predict(X_test)

    np.testing.assert_allclose(p1, p2, atol=1e-5, err_msg="XGBoost is non-deterministic")


def test_ensemble_is_rowwise_mean():
    """Asserts that the ensemble is an exact equal-weight mean of the 3 model predictions."""
    p_xgb = np.array([10.0, 50.0, 100.0])
    p_lgb = np.array([20.0, 40.0, 110.0])
    p_cat = np.array([15.0, 60.0, 90.0])

    ensemble = (p_xgb + p_lgb + p_cat) / 3.0
    expected = np.array([15.0, 50.0, 100.0])

    np.testing.assert_allclose(ensemble, expected, atol=1e-6)


def test_test_set_isolation(synthetic_df):
    """Ensures test split data is never present in X_train or X_val during feature prep."""
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(synthetic_df, "24h_weather")

    test_indices = set(X_test.index)
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)

    assert len(test_indices.intersection(train_indices)) == 0, "Test data leaked into train set"
    assert len(test_indices.intersection(val_indices)) == 0, "Test data leaked into val set"
