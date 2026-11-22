"""Unit tests for Phase 7 Consolidated Leaderboard and Model Registry."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.evaluation.model_registry import load_registered_model, predict_operational


def test_leaderboard_outputs_exist():
    """Verify that reports and figures for Phase 7 exist."""
    master_csv = ROOT / "reports/phase7/master_leaderboard.csv"
    op_csv = ROOT / "reports/phase7/operational_leaderboard_1h.csv"
    da_csv = ROOT / "reports/phase7/dayahead_leaderboard_24h.csv"
    report_md = ROOT / "reports/phase7/leaderboard_synthesis_report.md"

    assert master_csv.exists(), "master_leaderboard.csv should exist"
    assert op_csv.exists(), "operational_leaderboard_1h.csv should exist"
    assert da_csv.exists(), "dayahead_leaderboard_24h.csv should exist"
    assert report_md.exists(), "leaderboard_synthesis_report.md should exist"

    fig1 = ROOT / "figures/phase7/leaderboard_1h_plant1_bar.png"
    fig2 = ROOT / "figures/phase7/leaderboard_24h_plant1_bar.png"
    fig3 = ROOT / "figures/phase7/weather_vs_noweather_impact.png"
    fig4 = ROOT / "figures/phase7/model_complexity_vs_accuracy.png"

    assert fig1.exists(), "leaderboard_1h_plant1_bar.png should exist"
    assert fig2.exists(), "leaderboard_24h_plant1_bar.png should exist"
    assert fig3.exists(), "weather_vs_noweather_impact.png should exist"
    assert fig4.exists(), "model_complexity_vs_accuracy.png should exist"


def test_leaderboard_schema_and_ranks():
    """Verify master leaderboard schema, row counts, and champion scores."""
    df = pd.read_csv(ROOT / "reports/phase7/master_leaderboard.csv")
    assert not df.empty
    expected_cols = ["plant", "phase", "model_family", "model_name", "horizon", "uses_weather", "rmse_all", "mae_all", "skill_score", "rank_in_group"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in master leaderboard"

    # Verify LightGBM with weather on Plant 1 has excellent scores
    p1_lgb = df[(df["plant"] == "Plant 1") & (df["model_name"] == "LightGBM (weather)") & (df["horizon"] == "1h")]
    assert not p1_lgb.empty
    rmse_lgb = float(p1_lgb["rmse_all"].iloc[0])
    skill_lgb = float(p1_lgb["skill_score"].iloc[0])

    assert rmse_lgb < 20.0, f"LightGBM 1h RMSE should be < 20 kW, got {rmse_lgb}"
    assert skill_lgb > 0.80, f"LightGBM 1h skill score should be > 0.80, got {skill_lgb}"


def test_model_registry_artifacts():
    """Verify serialized model artifacts and metadata card."""
    registry_dir = ROOT / "models/registry"
    card_path = registry_dir / "model_card.json"
    assert card_path.exists(), "model_card.json must exist in registry"

    with open(card_path, "r") as f:
        card = json.load(f)

    assert card["primary_champion_architecture"] == "LightGBM (Gradient Boosted Trees)"
    assert card["registry_status"] == "production_ready"

    # Verify model weights exist
    for plant_id in [1, 2]:
        for horizon in ["1h", "24h"]:
            model_file = registry_dir / f"lightgbm_{horizon}_plant{plant_id}.joblib"
            assert model_file.exists(), f"Serialized model {model_file} must exist"


def test_model_registry_inference():
    """Verify that registered model can be cleanly loaded and used for prediction."""
    model, meta = load_registered_model(plant=1, horizon="1h")
    assert model is not None
    assert "feature_names" in meta
    assert len(meta["feature_names"]) > 0

    # Load features and prepare test batch
    feat_df = pd.read_csv(ROOT / "data/features/features_plant1.csv", index_col="timestamp", parse_dates=True)
    from src.models.tree_features import add_extended_lags
    feat_df_extended = add_extended_lags(feat_df)
    test_rows = feat_df_extended[feat_df_extended["split"] == "test"].head(24)

    preds = predict_operational(model, test_rows, meta["feature_names"])
    assert len(preds) == 24
    assert np.all(preds >= 0.0), "All predictions must be non-negative"

    # Check night hours zeroing constraint
    for i, (_, row) in enumerate(test_rows.iterrows()):
        if row["hour"] < 6 or row["hour"] >= 19 or row["is_daylight"] == 0:
            assert preds[i] == 0.0, f"Hour {row['hour']} should be 0.0 at night"
