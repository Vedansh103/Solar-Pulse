"""Unit tests for Phase 8 and Phase 9: LLM Operational Assistant and Safety."""

from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.llm.tools import (
    get_plant_overview,
    get_plant_forecast,
    diagnose_inverter_anomalies,
    get_maintenance_recommendations,
)
from src.llm.assistant import SolarOperationalAssistant


def test_tools_overview():
    """Verify get_plant_overview returns valid numerical data."""
    res = get_plant_overview(plant=1)
    assert res["plant"] == 1
    assert res["total_generation_kwh"] > 0
    assert res["peak_generation_kw"] > 0
    assert "weather_summary" in res
    assert res["weather_summary"]["mean_daylight_irradiation_kw_m2"] > 0


def test_tools_forecast():
    """Verify get_plant_forecast uses registered model and produces valid predictions."""
    res = get_plant_forecast(plant=1, horizon="24h", start_date="2020-06-11")
    assert res["plant"] == 1
    assert res["total_forecast_kwh"] > 0
    assert res["peak_forecast_kw"] > 0
    assert len(res["hourly_forecast"]) == 24
    # Ensure night hours are physically zero
    for h in res["hourly_forecast"]:
        hr_num = int(h["hour"].split()[1].split(":")[0])
        if hr_num < 6 or hr_num >= 19:
            assert h["predicted_kw"] == 0.0


def test_tools_anomaly_diagnosis():
    """Verify anomaly diagnosis tool detects real Plant 2 trips."""
    res = diagnose_inverter_anomalies(plant=2)
    assert res["plant"] == 2
    assert res["inverter_trips_count"] > 0
    assert res["total_estimated_lost_kwh"] > 50000

    # Ensure Quc1TzYxW2pYoWX is top identified faulty hardware
    top_invs = [item["inverter_id"] for item in res["worst_affected_inverters"]]
    assert "Quc1TzYxW2pYoWX" in top_invs


def test_tools_maintenance_recommendations():
    """Verify maintenance recommendations propose concrete work orders."""
    res = get_maintenance_recommendations(plant=2)
    assert res["plant"] == 2
    assert len(res["actionable_work_orders"]) > 0

    # Verify categories present
    cats = [item["category"] for item in res["actionable_work_orders"]]
    assert any("Electrical Disconnect" in c for c in cats)
    assert any("Panel Soiling" in c for c in cats)


def test_assistant_grounding_and_safety():
    """Verify assistant produces grounded answers and refuses out-of-domain queries."""
    assistant = SolarOperationalAssistant()

    # Anomaly query
    resp1 = assistant.ask("Why did generation drop at Plant 2?")
    assert "Quc1TzYxW2pYoWX" in resp1
    assert "kWh" in resp1

    # Maintenance query
    resp2 = assistant.ask("Which panels need washing or cleaning at Plant 2?")
    assert "cleaning" in resp2.lower() or "wash" in resp2.lower()

    # Safety / Anti-hallucination refusal
    resp3 = assistant.ask("What was the power generation in December 2024?")
    assert "Data Unavailable" in resp3
    assert "May 15 to June 17, 2020" in resp3

