# Smart Crop Disease Intelligence System

Software-only crop advisory platform for Rajasthan. Final year project, SKIT Jaipur —
CSE (Data Science), DS-B, project SKIT/DS/2023-2027/015.

## Status

| Module | State |
| --- | --- |
| Data acquisition | scripted, verified sources |
| Disease classifier | training pipeline ready, runs on Kaggle T4 |
| Crop recommender | pipeline ready (RandomForest / XGBoost + SHAP) |
| Pest risk | rule engine |
| Irrigation / fertilizer | FAO-56 + rule tables, Open-Meteo |
| Quantization | ONNX fp32 → fp16 → int8 → int4 ladder + benchmark |
| API | FastAPI skeleton against the frozen contract |

## Quick start

```bash
pip install -r requirements.txt

# 1. Data (needs a Kaggle API token in ~/.kaggle/kaggle.json)
python data/download.py --what all --dest data/raw

# 2. Train (GPU strongly recommended; on Kaggle use notebooks/02)
python -m src.train --config configs/train_disease.yaml

# 3. Evaluate + explain
python -m src.evaluate --ckpt runs/best.pt --out results/

# 4. Quantize and benchmark
python -m src.quantize --ckpt runs/best.pt --calib data/raw/plantvillage/valid --out results/

# 5. Serve
uvicorn api.main:app --reload
```

## Data sources

All verified on 9 November 2026.

| Dataset | Content | Link |
| --- | --- | --- |
| PlantVillage (augmented) | 87.9k leaf images, 38 classes, pre-split | kaggle.com/datasets/vipoooool/new-plant-diseases-dataset |
| PlantVillage (original) | 54,305 raw images, 38 class folders | github.com/spMohanty/PlantVillage-Dataset |
| PlantDoc | 2,598 field photos, 13 species, 27 classes | kaggle.com/datasets/nirmalsankalana/plantdoc-dataset |
| Crop recommendation | 2,200 rows, 22 crops, 7 features | kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| Open-Meteo | free weather + ET0, no API key | api.open-meteo.com/v1/forecast |

## Honest reporting

PlantVillage is laboratory imagery on plain backgrounds. PlantDoc is real field imagery. Accuracy on
the first is not accuracy on the second, and neither is accuracy on a Rajasthan farmer's phone.
Both numbers are reported separately in every results table.

## License

Academic project, SKIT Jaipur. Datasets retain their own licences (PlantDoc: CC BY 4.0;
crop recommendation: Apache 2.0).
