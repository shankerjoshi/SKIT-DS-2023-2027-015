# AGENTS.md — read this before you write any code

Project: **Smart Crop Disease Intelligence System**
Repo: `shankerjoshi/SKIT-DS-2023-2027-015` · SKIT Jaipur, CSE (Data Science), DS-B · Project ID SKIT/DS/2023-2027/015
Owner: Shanker Joshi (AI/ML). Deadline for the working demo: **15 November 2026**.

## What this project is

A software-only farmer advisory platform for Rajasthan. Five objectives:

1. Crop disease detection from a leaf photo (trained model)
2. Pest outbreak risk (weather rule engine — **not** a trained model)
3. Crop recommendation from soil and climate values (trained model)
4. Irrigation and fertilizer advisory (FAO-56 ET0 + rule tables + weather API)
5. Analytical dashboard over the farmer's own history (SQL aggregation)

## Hard rules

- **No field data exists.** We have no Rajasthan field photos. Never state or imply field accuracy for bajra, mustard or guar. Always report two numbers separately: lab accuracy on PlantVillage, field-proxy accuracy on PlantDoc.
- **Pest risk stays a rule engine.** No labelled outbreak dataset exists. Do not invent one or train a model on fabricated labels.
- **Nothing is trained in this repo's CI.** The sandbox has no GPU. Training runs on Kaggle notebooks (T4, free tier).
- **Never commit datasets or weights.** `data/`, `*.pt`, `*.onnx`, `*.tflite` are gitignored. Publish weights as GitHub Releases or a Hugging Face repo.
- **Honest metrics only.** Every number in `results/` must come from a script in this repo that can be re-run. If a number is copied from a paper, label it as such.

## The API contract — frozen, do not change without telling Track B

See `docs/CONTRACT.md`. Six endpoints, snake_case JSON, all responses include `model_version`.

## Layout

```
data/download.py        Kaggle + Open-Meteo fetch and verification
src/data/dataset.py     dataset indexing, transforms, PlantVillage<->PlantDoc label mapping
src/models/disease.py   model factory (MobileNetV3-Large / EfficientNet-B0)
src/train.py            fp16 AMP training, class weights, early stopping
src/evaluate.py         macro-F1, per-class F1, confusion matrix, ECE calibration
src/xai.py              Grad-CAM, occlusion sensitivity, SHAP helpers
src/quantize.py         ONNX export + fp32/fp16/int8/int4 ladder + benchmark
src/rules/              pest risk, irrigation (FAO-56), fertilizer tables
api/                    FastAPI service
notebooks/              Kaggle-ready notebooks
```

## Working agreements

- Python 3.11+, type hints on public functions, `black -l 100` style.
- Every script must run with `--help` and take paths from arguments, never hard-coded absolutes.
- If something cannot be done, say so in the docstring and open an issue. Do not stub a function that returns fake metrics.
- One logical change per commit. Message format: `feat(quant): add int4 weight-only path`.

## What is deliberately out of scope until 15 Nov

Own photo collection · mustard/bajra/guar classes · LSTM yield prediction · NDVI · research paper · Kubernetes.
