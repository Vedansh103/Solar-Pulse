"""Unit tests for Phase 3 output reports, leaderboards, and figures."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("plant", [1, 2])
def test_phase3_reports_exist_and_valid(plant: int):
    """Verifies that all Phase 3 report CSVs and markdown files exist and adhere to schema."""
    reports_dir = ROOT / "reports" / "phase3" / f"plant{plant}"
    if not (reports_dir / "metrics_test.csv").exists():
        pytest.skip(f"Phase 3 reports not yet generated for plant {plant}. Run pipeline first.")

    # 1. Check metrics_test.csv
    metrics_path = reports_dir / "metrics_test.csv"
    assert metrics_path.exists(), f"metrics_test.csv missing for plant {plant}"
    metrics_df = pd.read_csv(metrics_path)
    assert not metrics_df.empty
    expected_metrics_cols = {
        "model", "horizon", "feature_set", "uses_measured_weather",
        "rmse_all", "mae_all", "rmse_daylight", "mae_daylight", "skill_vs_seasonal_naive"
    }
    assert expected_metrics_cols.issubset(set(metrics_df.columns))
    assert (metrics_df["rmse_all"] >= 0).all()
    assert (metrics_df["mae_all"] >= 0).all()

    # 2. Check tuning_val.csv
    tuning_path = reports_dir / "tuning_val.csv"
    assert tuning_path.exists()
    tuning_df = pd.read_csv(tuning_path)
    assert not tuning_df.empty
    assert "val_rmse" in tuning_df.columns

    # 3. Check forecasts_test.csv
    forecasts_path = reports_dir / "forecasts_test.csv"
    assert forecasts_path.exists()
    forecasts_df = pd.read_csv(forecasts_path, index_col="timestamp")
    assert len(forecasts_df) == 168, f"Expected 168 test week rows, got {len(forecasts_df)}"
    assert "actual" in forecasts_df.columns
    assert (forecasts_df >= 0.0).all().all(), "Negative forecasts found"

    # 4. Check leaderboard.csv
    leaderboard_path = reports_dir / "leaderboard.csv"
    assert leaderboard_path.exists()
    lb_df = pd.read_csv(leaderboard_path)
    assert "uses_measured_weather" in lb_df.columns
    assert lb_df["rmse_all"].is_monotonic_increasing, "Leaderboard not sorted by RMSE ascending"

    # 5. Check shap_groups.csv
    shap_path = reports_dir / "shap_groups.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        for h in ["1h", "24h"]:
            h_df = shap_df[shap_df["horizon"] == h]
            if not h_df.empty:
                sum_pct = h_df["pct_importance"].sum()
                assert np.isclose(sum_pct, 100.0, atol=0.1), f"SHAP group percentages sum to {sum_pct} for {h}"

    # 6. Check summary.md
    summary_path = reports_dir / "summary.md"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 100


@pytest.mark.parametrize("plant", [1, 2])
def test_phase3_figures_exist(plant: int):
    """Verifies that all Phase 3 forecast and SHAP figures exist and are non-empty."""
    figures_dir = ROOT / "figures" / "phase3" / f"plant{plant}"
    if not (figures_dir / "test_forecasts_1h.png").exists():
        pytest.skip(f"Phase 3 figures not yet generated for plant {plant}. Run pipeline first.")

    expected_figures = [
        "test_forecasts_1h.png",
        "test_forecasts_24h.png",
        "shap_summary_1h.png",
        "shap_bar_1h.png",
        "shap_summary_24h.png",
        "shap_bar_24h.png",
    ]

    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Figure {fig_name} is missing for plant {plant}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is empty or corrupted"
