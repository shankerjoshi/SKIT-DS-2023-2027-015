"""Pest outbreak risk from weather thresholds. Deliberately a rule engine, not a model.

There is no public labelled pest-outbreak dataset for Rajasthan, so a trained classifier would be
fiction. These thresholds come from ICAR/state extension advisories and are stated as heuristics
in the report, with the citation for each crop.
"""
from __future__ import annotations

from dataclasses import dataclass

# crop -> list of (name, condition description, lambda over daily aggregates)
THRESHOLDS = {
    "mustard": [
        ("aphid", "max temp 12-20 C, humidity > 70%, dry spell", lambda t, rh, rain: 12 <= t <= 20 and rh > 70 and rain < 2),
        ("alternaria blight", "18-25 C with RH > 80% for 2+ days", lambda t, rh, rain: 18 <= t <= 25 and rh > 80),
    ],
    "wheat": [
        ("yellow rust", "8-15 C, RH > 80%, dew", lambda t, rh, rain: 8 <= t <= 15 and rh > 80),
        ("aphid", "15-25 C, dry, low wind", lambda t, rh, rain: 15 <= t <= 25 and rain < 1),
    ],
    "bajra": [
        ("downy mildew", "25-30 C with RH > 85%", lambda t, rh, rain: 25 <= t <= 30 and rh > 85),
    ],
    "rice": [
        ("brown planthopper", "25-30 C, RH > 80%, cloudy", lambda t, rh, rain: 25 <= t <= 30 and rh > 80),
    ],
}


@dataclass
class DailyWeather:
    date: str
    t_mean: float
    rh_mean: float
    rain_mm: float


def assess(crop: str, days: list[DailyWeather], window: int = 3) -> dict:
    """Score the last `window` days against the crop's pest rules."""
    crop = crop.strip().lower()
    rules = THRESHOLDS.get(crop)
    if rules is None:
        return {"crop": crop, "supported": False, "risk": "unknown",
                "reason": f"no pest rule table for {crop!r}; supported: {sorted(THRESHOLDS)}"}

    recent = days[-window:]
    if not recent:
        return {"crop": crop, "supported": True, "risk": "unknown", "reason": "no weather data"}

    t = sum(d.t_mean for d in recent) / len(recent)
    rh = sum(d.rh_mean for d in recent) / len(recent)
    rain = sum(d.rain_mm for d in recent)

    hits = [{"pest": name, "condition": desc} for name, desc, test in rules if test(t, rh, rain)]
    score = len(hits) / len(rules)
    level = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"
    return {
        "crop": crop,
        "supported": True,
        "risk": level,
        "score": round(score, 2),
        "matched": hits,
        "window_days": len(recent),
        "aggregates": {"t_mean": round(t, 1), "rh_mean": round(rh, 1), "rain_mm": round(rain, 1)},
        "method": "weather-threshold rule engine (not a trained model)",
    }


def from_open_meteo(payload: dict) -> list[DailyWeather]:
    """Convert an Open-Meteo daily response into DailyWeather rows."""
    d = payload.get("daily", {})
    keys = d.get("time", [])
    out = []
    for i, day in enumerate(keys):
        out.append(DailyWeather(
            date=day,
            t_mean=float(d.get("temperature_2m_mean", [0] * len(keys))[i]),
            rh_mean=float(d.get("relative_humidity_2m_mean", [0] * len(keys))[i]),
            rain_mm=float(d.get("precipitation_sum", [0] * len(keys))[i]),
        ))
    return out
