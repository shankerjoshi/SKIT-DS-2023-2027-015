"""Quick self-checks that need no GPU and no dataset.

    pytest -q
"""
from __future__ import annotations

from src.rules import fertilizer, irrigation, pest_risk


def test_kc_monotonic_across_stages():
    kc_mid = irrigation.kc_for("wheat", "mid")
    kc_ini = irrigation.kc_for("wheat", "initial")
    assert kc_mid > kc_ini


def test_irrigation_zero_when_rain_covers_demand():
    a = irrigation.advise("wheat", "mid", et0_mm=5.0, rain_mm=20.0)
    assert a.net_irrigation_mm == 0.0


def test_irrigation_positive_in_dry_week():
    a = irrigation.advise("wheat", "mid", et0_mm=6.0, rain_mm=0.0)
    assert a.net_irrigation_mm > 0
    assert a.litres_per_hectare > 0


def test_fertilizer_low_nitrogen_raises_dose():
    plan = fertilizer.advise("wheat", n=100, p=15, k=150)
    assert plan.status["nitrogen"] == "low"
    assert plan.dose_kg_per_ha["nitrogen"] > 120


def test_fertilizer_rejects_unknown_crop():
    try:
        fertilizer.advise("dragonfruit", 1, 1, 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_pest_rule_engine_flags_wheat_yellow_rust():
    days = [pest_risk.DailyWeather("2026-11-09", 11.0, 85.0, 0.0)] * 3
    out = pest_risk.assess("wheat", days)
    assert out["risk"] in {"medium", "high"}
    assert out["method"].startswith("weather-threshold")


def test_pest_unknown_crop_is_reported_not_guessed():
    out = pest_risk.assess("dragonfruit", [])
    assert out["supported"] is False
