"""Fertilizer advisory from a Soil Health Card style reading.

Two layers:
1. A deterministic rule table (N-P-K status -> dose adjustment), which is what the report
   defends because it is explainable line by line.
2. Optional learned layer on the Kaggle fertilizer dataset, used only as a cross-check.

The rule table wins whenever the two disagree; the model is a sanity check, not the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# soil test value -> status band (kg/ha), based on typical Indian SHC bands
BANDS = {
    "nitrogen": (280, 560),      # <low = low, >high = high
    "phosphorus": (10, 25),
    "potassium": (120, 280),
}

# crop -> recommended N:P:K kg/ha at a medium soil test
CROP_DOSE = {
    "wheat": (120, 60, 40),
    "mustard": (60, 40, 40),
    "bajra": (80, 40, 30),
    "guar": (20, 40, 20),
    "rice": (100, 50, 50),
}


def status_of(nutrient: str, value: float) -> str:
    lo, hi = BANDS[nutrient]
    return "low" if value < lo else "high" if value > hi else "medium"


def adjust(recommended: float, status: str) -> float:
    """Low -> +50%, high -> -50%, medium -> unchanged. Simple and defensible."""
    return {"low": recommended * 1.5, "high": recommended * 0.5}.get(status, recommended)


@dataclass
class FertilizerPlan:
    crop: str
    status: dict[str, str]
    dose_kg_per_ha: dict[str, float]
    split: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def advise(crop: str, n: float, p: float, k: float, ph: float | None = None,
           organic_carbon: float | None = None) -> FertilizerPlan:
    crop = crop.strip().lower()
    if crop not in CROP_DOSE:
        raise ValueError(f"no dose table for {crop!r}; supported: {sorted(CROP_DOSE)}")

    st = {"nitrogen": status_of("nitrogen", n), "phosphorus": status_of("phosphorus", p),
          "potassium": status_of("potassium", k)}
    base = CROP_DOSE[crop]
    dose = {nut: round(adjust(base[i], st[nut]), 0) for i, nut in enumerate(("nitrogen", "phosphorus", "potassium"))}

    plan = FertilizerPlan(crop=crop, status=st, dose_kg_per_ha=dose)

    # nitrogen is always split — leaching losses in light Rajasthan soils are severe
    plan.split = [
        f"N: 1/3 at sowing, 1/3 at first irrigation, 1/3 at 45-50 days ({dose['nitrogen']:.0f} kg/ha total)",
        f"P: full dose as basal ({dose['phosphorus']:.0f} kg/ha)",
        f"K: full dose as basal ({dose['potassium']:.0f} kg/ha)",
    ]
    if ph is not None and ph > 8.5:
        plan.warnings.append("pH above 8.5: micronutrient (Zn, Fe) deficiency likely — soil test before adding more N-P-K")
    if ph is not None and ph < 6.0:
        plan.warnings.append("pH below 6.0: apply lime before the next season")
    if organic_carbon is not None and organic_carbon < 0.5:
        plan.warnings.append("organic carbon below 0.5%: add FYM/compost, chemical dose alone will not lift yield")
    return plan


def explain(plan: FertilizerPlan) -> str:
    """One paragraph a farmer can read out loud."""
    parts = [f"For {plan.crop}, your soil tests read " + ", ".join(f"{k} {v}" for k, v in plan.status.items()) + "."]
    parts.append("Recommended dose: " + ", ".join(f"{k} {v:.0f} kg/ha" for k, v in plan.dose_kg_per_ha.items()) + ".")
    parts.extend(plan.split)
    parts.extend(plan.warnings)
    return " ".join(parts)
