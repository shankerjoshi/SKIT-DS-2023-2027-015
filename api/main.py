"""FastAPI service for the frozen contract in docs/CONTRACT.md.

Design rule: if a model file is missing, return HTTP 503 with a clear message. Never invent a
prediction — a demo that lies is worse than a demo that is not ready.

    uvicorn api.main:app --reload
    curl -F "file=@leaf.jpg" localhost:8000/api/v1/diagnose
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (AdvisoryRequest, AdvisoryResponse, CropRequest, CropResponse,
                         DiagnosisResponse, LoginRequest, PestRiskResponse, Prediction, TokenResponse)
from src.rules import fertilizer as fert_rules
from src.rules import irrigation as irr_rules
from src.rules import pest_risk

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
VERSION = os.getenv("MODEL_VERSION", "dev")
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

app = FastAPI(title="Smart Crop Disease Intelligence System", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {"session": None, "classes": None, "transform": None}


def _load_disease_model():
    """Lazy-load the ONNX classifier once."""
    if _state["session"] is not None:
        return _state["session"]
    path = MODEL_DIR / "model_int8_static.onnx"
    if not path.exists():
        path = MODEL_DIR / "model_fp32.onnx"
    if not path.exists():
        raise HTTPException(503, detail={"error": {"code": "MODEL_NOT_LOADED",
                                                   "message": f"no ONNX model in {MODEL_DIR}/ — run src/quantize.py"}})
    import onnxruntime as ort
    _state["session"] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    cls_file = MODEL_DIR / "classes.json"
    _state["classes"] = json.loads(cls_file.read_text()) if cls_file.exists() else []
    from src.data.dataset import build_transforms
    _state["transform"] = build_transforms(False, 224)
    return _state["session"]


def _treatments() -> dict:
    p = Path("data/treatments.json")
    return json.loads(p.read_text()) if p.exists() else {}


def _weather(lat: float, lon: float) -> dict:
    params = {"latitude": lat, "longitude": lon,
              "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
                       "et0_fao_evapotranspiration,temperature_2m_max",
              "forecast_days": 7, "timezone": "Asia/Kolkata"}
    try:
        r = requests.get(OPEN_METEO, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(502, detail={"error": {"code": "WEATHER_UNAVAILABLE", "message": str(e)}})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": VERSION,
            "model_loaded": (MODEL_DIR / "model_int8_static.onnx").exists() or (MODEL_DIR / "model_fp32.onnx").exists()}


@app.post("/api/v1/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...)) -> DiagnosisResponse:
    sess = _load_disease_model()
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(413, detail={"error": {"code": "FILE_TOO_LARGE", "message": "max 5 MB"}})
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, detail={"error": {"code": "BAD_IMAGE", "message": "cannot decode image"}})

    x = _state["transform"](img).unsqueeze(0).numpy().astype(np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    classes = _state["classes"] or [f"class_{i}" for i in range(len(p))]
    order = np.argsort(p)[::-1][:3]
    label = classes[int(order[0])]
    meta = _treatments().get(label, {})
    return DiagnosisResponse(
        disease=label,
        display_name=label.replace("___", " ").replace("_", " "),
        confidence=float(p[order[0]]),
        top3=[Prediction(label=classes[int(i)], p=float(p[i])) for i in order],
        cause=meta.get("cause", ""),
        treatment=meta.get("treatment", []),
        prevention=meta.get("prevention", []),
        model_version=VERSION,
    )


@app.post("/api/v1/recommend-crop", response_model=CropResponse)
def recommend_crop(req: CropRequest) -> CropResponse:
    """Rule-scored fallback until the trained recommender is exported to models/crop_model.onnx."""
    scores = {"rice": req.rainfall / 250, "wheat": (200 - abs(req.temperature - 20)) / 200,
              "mustard": (200 - abs(req.temperature - 18)) / 200, "bajra": (200 - abs(req.temperature - 30)) / 200}
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return CropResponse(recommended=top[0][0],
                        top3=[Prediction(label=k, p=float(min(max(v, 0), 1))) for k, v in top],
                        explanation={"note": "heuristic placeholder — replace with models/crop_model.onnx"},
                        model_version=VERSION)


@app.post("/api/v1/advisory", response_model=AdvisoryResponse)
def advisory(req: AdvisoryRequest) -> AdvisoryResponse:
    wx = _weather(req.lat, req.lon)
    daily = wx.get("daily", {})
    et0 = float(daily.get("et0_fao_evapotranspiration", [5.0])[0])
    rain = float(daily.get("precipitation_sum", [0.0])[0])
    irr = irr_rules.advise(req.crop, req.stage, et0, rain)

    fert = None
    if req.soil is not None:
        try:
            plan = fert_rules.advise(req.crop, req.soil.N, req.soil.P, req.soil.K, req.soil.ph, req.soil.organic_carbon)
            fert = {"status": plan.status, "dose_kg_per_ha": plan.dose_kg_per_ha,
                    "split": plan.split, "warnings": plan.warnings, "summary": fert_rules.explain(plan)}
        except ValueError as e:
            fert = {"error": str(e)}

    alerts = []
    tmax = daily.get("temperature_2m_max", [])
    if tmax and max(tmax) > 42:
        alerts.append("heat wave: irrigate in the evening and mulch to cut evaporation")
    if rain > 20:
        alerts.append("heavy rain expected: hold off irrigation and fertilizer for 48 h")

    return AdvisoryResponse(irrigation=irr.__dict__, fertilizer=fert, weather_alerts=alerts, model_version=VERSION)


@app.get("/api/v1/pest-risk", response_model=PestRiskResponse)
def pest(crop: str = Query(...), lat: float = Query(...), lon: float = Query(...)) -> PestRiskResponse:
    wx = _weather(lat, lon)
    days = pest_risk.from_open_meteo(wx)
    out = pest_risk.assess(crop, days)
    return PestRiskResponse(crop=crop, risk=out["risk"], matched=out.get("matched", []),
                            method=out.get("method", "weather-threshold rule engine"), model_version=VERSION)


@app.get("/api/v1/dashboard/{user_id}")
def dashboard(user_id: str) -> dict:
    """Wire to the database in Track B; empty shell until then, never fabricated numbers."""
    return {"user_id": user_id, "crop_health_trend": [], "recent_diagnoses": [],
            "irrigation_log": [], "weather_summary": {}, "note": "database not connected yet"}


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    raise HTTPException(501, detail={"error": {"code": "AUTH_NOT_IMPLEMENTED",
                                               "message": "Track B owns auth; contract is fixed in docs/CONTRACT.md"}})
