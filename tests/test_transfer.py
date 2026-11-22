"""Unit tests for Phase 6 Cross-Plant Transfer & Generalization."""

from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_outputs_exist():
    """Verify that reports and figures for Phase 6 exist."""
    metrics_path = ROOT / "reports/phase6/transfer_metrics.csv"
    report_path = ROOT / "reports/phase6/transfer_synthesis_report.md"
    bar_plot = ROOT / "figures/phase6/transfer_comparison_bar.png"
    overlay_plot = ROOT / "figures/phase6/cross_plant_forecast_overlay.png"

    assert metrics_path.exists(), "transfer_metrics.csv should exist"
    assert report_path.exists(), "transfer_synthesis_report.md should exist"
    assert bar_plot.exists(), "transfer_comparison_bar.png should exist"
    assert overlay_plot.exists(), "cross_plant_forecast_overlay.png should exist"


def test_transfer_metrics_validity():
    """Verify transfer metrics values and structure."""
    df = pd.read_csv(ROOT / "reports/phase6/transfer_metrics.csv")

    assert not df.empty
    expected_cols = ["model", "training_source", "testing_target", "transfer_mode", "rmse_all", "mae_all", "skill_score"]
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing from transfer metrics"

    # Verify that capacity-scaled transfer outperforms raw zero-shot transfer
    lgb_raw = df[(df["model"] == "LightGBM") & (df["transfer_mode"] == "Zero-Shot Direct Transfer")]
    lgb_scaled = df[(df["model"] == "LightGBM") & (df["transfer_mode"] == "Capacity-Scaled Transfer")]

    assert not lgb_raw.empty
    assert not lgb_scaled.empty

    rmse_raw = float(lgb_raw["rmse_all"].iloc[0])
    rmse_scaled = float(lgb_scaled["rmse_all"].iloc[0])

    assert rmse_scaled < rmse_raw, f"Capacity scaling ({rmse_scaled}) must improve RMSE over raw zero-shot ({rmse_raw})"
    assert float(lgb_scaled["skill_score"].iloc[0]) > 0.0, "Capacity scaled model must have positive skill score against baseline"

