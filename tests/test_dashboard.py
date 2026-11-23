"""Unit tests for Phase 10 & 11: Interactive Operations Dashboard."""

import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_file_syntax():
    """Verify that app/dashboard.py exists and parses cleanly without Python syntax errors."""
    dash_file = ROOT / "app/dashboard.py"
    assert dash_file.exists(), "app/dashboard.py must exist"

    code = dash_file.read_text(encoding="utf-8")
    parsed = ast.parse(code)
    assert parsed is not None, "dashboard.py must parse cleanly"


def test_dashboard_data_loading():
    """Verify that data loaders for dashboard retrieve valid data."""
    from app.dashboard import load_plant_data

    # Test Plant 1 data load
    df_feat1, df_h1, df_a1 = load_plant_data(1)
    assert not df_feat1.empty
    assert len(df_feat1) == 816
    assert not df_h1.empty
    assert len(df_h1) == 22  # 22 inverters

    # Test Plant 2 data load
    df_feat2, df_h2, df_a2 = load_plant_data(2)
    assert not df_feat2.empty
    assert len(df_feat2) == 816
    assert not df_h2.empty
    assert len(df_h2) == 22  # 22 inverters
    assert "Quc1TzYxW2pYoWX" in df_h2["inverter_id"].values


def test_dashboard_model_inference():
    """Verify that model loading inside dashboard generates valid predictions."""
    from app.dashboard import get_model, load_plant_data
    from src.evaluation.model_registry import predict_operational

    model_1h, meta_1h = get_model(1, "1h")
    df_feat, _, _ = load_plant_data(1)
    test_rows = df_feat[df_feat["split"] == "test"].head(10)

    preds = predict_operational(model_1h, test_rows, meta_1h["feature_names"])
    assert len(preds) == 10
    assert (preds >= 0.0).all()


def test_ai_forecast_explanation():
    """Verify that AI forecast explanation generates structured, grounded plain-language points."""
    from src.llm.assistant import SolarOperationalAssistant

    assistant = SolarOperationalAssistant(use_remote_llm=False)
    test_context = {
        "plant": 1,
        "horizon": "1h",
        "weather": "🌧️ Rainy / Stormy",
        "input_irrad_kw_m2": 0.565,
        "effective_irrad_kw_m2": 0.085,
        "optical_transmittance_pct": 15.0,
        "ambient_temp_c": 28.5,
        "module_temp_c": 45.0,
        "thermal_rise_delta_t_c": 16.5,
        "thermal_derating_pct": 8.0,
        "clearness_index": 0.059,
        "predicted_power_kw": 223.9,
        "capacity_utilization_pct": 18.7,
        "plant_uptime_pct": 99.8,
    }

    explanation = assistant._fallback_forecast_explanation(test_context)
    assert "- ☀️ **Sunlight & Sky:**" in explanation
    assert "- 🌡️ **Temperature Impact:**" in explanation
    assert "- ⚡ **Equipment & Grid Status:**" in explanation
    assert "223.9 kW" in explanation
    assert "18.7%" in explanation


