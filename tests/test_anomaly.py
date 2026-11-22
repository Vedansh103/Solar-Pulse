"""Unit tests for Phase 5: Anomaly Detection Engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly.detect import (
    detect_inverter_anomalies,
    detect_stl_residual_anomalies,
    detect_isolation_forest,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_inverter_df():
    path = ROOT / "data/processed/inverter_hourly_plant1.csv"
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


@pytest.fixture(scope="module")
def sample_master_df():
    path = ROOT / "data/processed/master_hourly_plant1.csv"
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def test_inverter_peer_anomaly_shapes(sample_inverter_df):
    peer_ratio, alerts, summary = detect_inverter_anomalies(sample_inverter_df)

    assert peer_ratio.shape == sample_inverter_df.shape
    assert len(summary) == 22
    assert "inverter_id" in summary.columns
    assert "availability_health_score" in summary.columns
    assert not alerts.empty
    assert set(alerts["severity"].unique()).issubset({"CRITICAL", "HIGH", "MEDIUM"})


def test_stl_residual_detection(sample_master_df):
    res_df, stl_alerts = detect_stl_residual_anomalies(sample_master_df, z_threshold=3.0)

    assert "stl_trend" in res_df.columns
    assert "stl_seasonal" in res_df.columns
    assert "stl_resid" in res_df.columns
    assert "stl_z_score" in res_df.columns


def test_isolation_forest_detection(sample_master_df):
    df_day, iso_alerts = detect_isolation_forest(sample_master_df, contamination=0.03)

    assert "iso_score" in df_day.columns
    assert not iso_alerts.empty
    assert (iso_alerts["iso_score"] == -1).all()


def test_phase5_outputs_exist():
    for plant in (1, 2):
        rep = ROOT / f"reports/phase5/plant{plant}"
        fig = ROOT / f"figures/phase5/plant{plant}"
        data = ROOT / "data/anomalies"

        assert (rep / "anomaly_summary.md").exists()
        assert (rep / "inverter_health_ranking.csv").exists()
        assert (data / f"plant{plant}_inverter_alerts.csv").exists()
        assert (data / f"plant{plant}_stl_alerts.csv").exists()
        assert (fig / f"plant{plant}_inverter_fault_ranking.png").exists()
        assert (fig / f"plant{plant}_inverter_heatmap.png").exists()
        assert (fig / f"plant{plant}_anomaly_scatter.png").exists()

