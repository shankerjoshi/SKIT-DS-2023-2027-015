"""Irrigation advisory: FAO-56 reference evapotranspiration x crop coefficient.

Open-Meteo already returns `et0_fao_evapotranspiration` (Penman-Monteith), so we do not
recompute it — we apply Kc per growth stage and compare with effective rainfall.

Net irrigation (mm) = max(0, ETc - effective_rain), where ETc = Kc x ET0.
"""
from __future__ import annotations

from dataclasses import dataclass

# FAO-56 single crop coefficients by stage: (initial, mid-season, late)
KC = {
    "wheat": (0.4, 1.15, 0.45),
    "mustard": (0.35, 1.1, 0.35),
    "bajra": (0.3, 1.0, 0.4),
    "guar": (0.35, 1.05, 0.5),
    "moth bean": (0.3, 0.95, 0.4),
    "rice": (1.05, 1.2, 0.9),
}

STAGES = ("initial", "development", "mid", "late")
STAGE_FRACTION = {"initial": 0.15, "development": 0.35, "mid": 0.75, "late": 0.95}


@dataclass
class IrrigationAdvice:
    crop: str
    stage: str
    kc: float
    et0_mm: float
    etc_mm: float
    effective_rain_mm: float
    net_irrigation_mm: float
    litres_per_hectare: float
    note: str


def kc_for(crop: str, stage: str) -> float:
    crop = crop.strip().lower()
    if crop not in KC:
        raise ValueError(f"no Kc table for {crop!r}; supported: {sorted(KC)}")
    ini, mid, late = KC[crop]
    if stage == "initial":
        return ini
    if stage == "development":
        return (ini + mid) / 2
    if stage == "mid":
        return mid
    if stage == "late":
        return late
    raise ValueError(f"stage must be one of {STAGES}")


def advise(crop: str, stage: str, et0_mm: float, rain_mm: float, rain_effectiveness: float = 0.8) -> IrrigationAdvice:
    """rain_effectiveness: fraction of rainfall that actually reaches the root zone."""
    kc = kc_for(crop, stage)
    etc = kc * et0_mm
    eff_rain = rain_mm * rain_effectiveness
    net = max(0.0, etc - eff_rain)
    return IrrigationAdvice(
        crop=crop.strip().lower(),
        stage=stage,
        kc=round(kc, 2),
        et0_mm=round(et0_mm, 2),
        etc_mm=round(etc, 2),
        effective_rain_mm=round(eff_rain, 2),
        net_irrigation_mm=round(net, 1),
        litres_per_hectare=round(net * 10_000, 0),   # 1 mm over 1 ha = 10,000 L
        note="apply in the early morning; skip if rain > 5 mm is forecast within 24 h",
    )


def stage_from_progress(days_since_sowing: int, season_length_days: int = 120) -> str:
    p = min(max(days_since_sowing / max(season_length_days, 1), 0.0), 1.0)
    for stage, threshold in STAGE_FRACTION.items():
        if p <= threshold:
            return stage
    return "late"
