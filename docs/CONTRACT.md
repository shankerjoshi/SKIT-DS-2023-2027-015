# API contract — frozen 9 November 2026

Both tracks build against this. Changing a field name means telling the other track the same day.
All responses are JSON, snake_case, and carry `model_version`.

Base path: `/api/v1`

## POST /diagnose

Multipart upload.

| field | type | notes |
| --- | --- | --- |
| file | image | jpg/png, max 5 MB |
| crop_hint | string, optional | e.g. `wheat` |

```json
{
  "disease": "Tomato___Late_blight",
  "display_name": "Tomato late blight",
  "confidence": 0.91,
  "top3": [{"label": "Tomato___Late_blight", "p": 0.91},
           {"label": "Tomato___Early_blight", "p": 0.06},
           {"label": "Tomato___healthy", "p": 0.02}],
  "cause": "Phytophthora infestans",
  "treatment": ["Remove and destroy infected leaves", "Apply mancozeb 2.5 g/L ..."],
  "prevention": ["Avoid overhead irrigation", "Rotate with non-solanaceous crop"],
  "model_version": "mobilenetv3l-int8-v1"
}
```

## POST /recommend-crop

```json
{"N": 90, "P": 42, "K": 43, "temperature": 26.5, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0}
```

Response: `{"recommended": "rice", "top3": [...], "explanation": {"N": "high", "rainfall": "adequate"}, "model_version": "..."}`

## POST /advisory

```json
{"lat": 26.91, "lon": 75.79, "crop": "wheat", "stage": "mid", "soil": {"N": 250, "P": 15, "K": 150, "ph": 8.1}}
```

Response: `{"irrigation": {...}, "fertilizer": {...}, "weather_alerts": [...], "model_version": "..."}`

## GET /pest-risk?crop=wheat&lat=26.91&lon=75.79

Response: `{"crop": "wheat", "risk": "medium", "matched": [...], "method": "weather-threshold rule engine"}`

## GET /dashboard/{user_id}

Response: `{"crop_health_trend": [...], "recent_diagnoses": [...], "irrigation_log": [...], "weather_summary": {...}}`

## POST /auth/login

`{"email": "...", "password": "..."}` -> `{"access_token": "...", "token_type": "bearer"}`

## Error shape

```json
{"error": {"code": "MODEL_NOT_LOADED", "message": "model file missing; run src/quantize.py"}}
```

HTTP 503 when the model is not on disk. Never return a fake prediction.
