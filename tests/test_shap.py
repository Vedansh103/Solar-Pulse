"""Unit tests for SHAP explainability and physical group attributions."""

import numpy as np
import pandas as pd
import pytest
import shap

from src.models.shap_analysis import FEATURE_GROUPS, compute_group_attribution
from src.models.tree_features import FEATURE_SETS, get_train_val_test_data
from src.models.tree_models import fit_final_model


def test_feature_groups_partition():
    """Verifies that all features across all 4 feature sets belong to exactly one physical group."""
    all_grouped_cols = []
    for g_name, cols in FEATURE_GROUPS.items():
        all_grouped_cols.extend(cols)

    # Check for no accidental duplicate assignment across groups
    assert len(all_grouped_cols) == len(set(all_grouped_cols)), "Duplicate columns found across FEATURE_GROUPS"

    # Check that every column in every feature set is present in FEATURE_GROUPS
    for fs_name, f_list in FEATURE_SETS.items():
        for col in f_list:
            assert col in all_grouped_cols, f"Column '{col}' in {fs_name} is missing from FEATURE_GROUPS"


def test_shap_additivity(synthetic_df):
    """Verifies that base_value + sum(shap_values) equals the model prediction (SHAP local accuracy)."""
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_data(synthetic_df, "1h_weather")

    model = fit_final_model("xgboost", {"max_depth": 3, "learning_rate": 0.05}, 15, X_train, y_train)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)

    raw_preds = model.predict(X_test)
    base_values = explanation.base_values
    shap_sums = explanation.values.sum(axis=1)

    reconstructed_preds = base_values + shap_sums
    np.testing.assert_allclose(reconstructed_preds, raw_preds, atol=1e-3, err_msg="SHAP additivity violated")


def test_group_attribution_sums_to_100():
    """Verifies that compute_group_attribution produces rows summing to 100%."""
    feature_names = ["irradiation", "module_temperature", "ac_power_lag_1", "hour_sin"]
    # Simulated SHAP values for 10 rows and 4 features
    shap_vals = np.array([
        [10.0, 5.0, 20.0, 2.0],
        [15.0, 7.0, 15.0, 3.0],
        [8.0, 4.0, 25.0, 1.0],
    ])

    group_df = compute_group_attribution(shap_vals, feature_names, horizon="1h")

    assert len(group_df) == 4
    total_pct = group_df["pct_importance"].sum()
    assert np.isclose(total_pct, 100.0, atol=0.1), f"Group percentages sum to {total_pct}, expected 100%"
