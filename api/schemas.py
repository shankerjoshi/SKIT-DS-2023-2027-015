"""Pydantic models mirroring docs/CONTRACT.md. Keep in step with the frontend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    p: float


class DiagnosisResponse(BaseModel):
    disease: str
    display_name: str
    confidence: float = Field(ge=0, le=1)
    top3: list[Prediction]
    cause: str = ""
    treatment: list[str] = []
    prevention: list[str] = []
    model_version: str


class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class CropResponse(BaseModel):
    recommended: str
    top3: list[Prediction]
    explanation: dict[str, str] = {}
    model_version: str


class SoilReading(BaseModel):
    N: float
    P: float
    K: float
    ph: float | None = None
    organic_carbon: float | None = None


class AdvisoryRequest(BaseModel):
    lat: float
    lon: float
    crop: str
    stage: Literal["initial", "development", "mid", "late"] = "mid"
    soil: SoilReading | None = None


class AdvisoryResponse(BaseModel):
    irrigation: dict
    fertilizer: dict | None = None
    weather_alerts: list[str] = []
    model_version: str


class PestRiskResponse(BaseModel):
    crop: str
    risk: str
    matched: list[dict] = []
    method: str = "weather-threshold rule engine"
    model_version: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
