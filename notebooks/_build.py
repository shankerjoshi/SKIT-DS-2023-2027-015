"""Generate the Kaggle-ready notebooks. Run once: python notebooks/_build.py"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src: str):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").split("\n")]}


def write(name: str, cells: list[dict]):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / name).write_text(json.dumps(nb, indent=1))
    print("wrote", name, f"({len(cells)} cells)")


SETUP = code("""
# Kaggle setup — run this first in every notebook.
# Settings -> Accelerator: GPU T4 x2 | Internet: ON
import os, subprocess, sys
REPO = "/kaggle/working/SKIT-DS-2023-2027-015"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "https://github.com/shankerjoshi/SKIT-DS-2023-2027-015.git", REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxconverter-common", "shap"], check=True)
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
""")

# ---------------------------------------------------------------- 01 EDA
write("01_eda_plantvillage.ipynb", [
    md("# 01 · Exploratory data analysis — PlantVillage + PlantDoc",
       "",
       "What this notebook answers, and the numbers that go into the report:",
       "1. How imbalanced are the 38 classes?",
       "2. What are the image sizes and are they consistent?",
       "3. Are there exact duplicate images between train and valid (leakage)?",
       "4. How different does PlantDoc look from PlantVillage? (this is the honest-framing slide)"),
    SETUP,
    code("""
from data.download import kaggle_download
from pathlib import Path
kaggle_download("vipoooool/new-plant-diseases-dataset", Path("data/raw/plantvillage"))
kaggle_download("nirmalsankalana/plantdoc-dataset", Path("data/raw/plantdoc"))
kaggle_download("atharvaingle/crop-recommendation-dataset", Path("data/raw/crop"))
"""),
    code("""
import pandas as pd, numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

PV = Path("data/raw/plantvillage/New Plant Diseases Dataset(Augmented)")
train_dir, valid_dir = PV / "train", PV / "valid"
rows = []
for split, root in (("train", train_dir), ("valid", valid_dir)):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            rows.append({"split": split, "class": p.parent.name, "path": str(p), "size_bytes": p.stat().st_size})
df = pd.DataFrame(rows)
print(df.split.value_counts())
print("classes:", df["class"].nunique(), "| total images:", len(df))
"""),
    code("""
import matplotlib.pyplot as plt
counts = df[df.split == "train"]["class"].value_counts()
print("most common:", counts.head(3).to_dict())
print("rarest:", counts.tail(3).to_dict())
print("imbalance ratio:", round(counts.max() / counts.min(), 2))

fig, ax = plt.subplots(figsize=(10, 10))
counts.sort_values().plot.barh(ax=ax)
ax.set_xlabel("training images"); ax.set_title("PlantVillage class distribution")
plt.tight_layout(); plt.savefig("results_eda_class_balance.png", dpi=110); plt.show()
"""),
    code("""
# image size + duplicate check (md5 catches the augmented copies Kaggle ships)
import hashlib
sizes = Counter()
hashes = Counter()
for p in df[df.split == "train"]["path"].sample(2000, random_state=0):
    with Image.open(p) as im:
        sizes[im.size] += 1
    hashes[hashlib.md5(open(p, "rb").read()).hexdigest()] += 1
print("distinct sizes:", sizes.most_common(5))
print("duplicate rate in the 2000 sampled:", round(1 - len(hashes) / 2000, 4))
"""),
    code("""
# side-by-side: lab image vs field image. Put this figure in the report.
import random
pv_files = df[df.split == "train"].sample(6, random_state=1)["path"].tolist()
pd_files = [str(p) for p in Path("data/raw/plantdoc").rglob("*") if p.suffix.lower() in {".jpg", ".png"}][:6]

fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for ax, f in zip(axes[0], pv_files):
    ax.imshow(Image.open(f)); ax.set_title("lab", fontsize=8); ax.axis("off")
for ax, f in zip(axes[1], pd_files):
    ax.imshow(Image.open(f)); ax.set_title("field", fontsize=8); ax.axis("off")
plt.suptitle("PlantVillage (top) vs PlantDoc (bottom) — why two accuracy numbers are reported")
plt.tight_layout(); plt.savefig("results_eda_lab_vs_field.png", dpi=110); plt.show()
"""),
    code("""
# crop recommendation table
crop_csv = next(Path("data/raw/crop").rglob("*.csv"))
c = pd.read_csv(crop_csv)
print(c.shape); print(c["label"].nunique(), "crops"); print(c.describe().round(2))
print(c["label"].value_counts().describe())
"""),
    md("### Findings to carry into the report",
       "- Class imbalance ratio and the rarest classes (these drive the class-weight choice in training).",
       "- Whether duplicate images exist between train and valid — if yes, say so and use the shipped split anyway, noting the caveat.",
       "- The lab-vs-field figure justifies reporting two accuracy numbers."),
])

# ---------------------------------------------------------------- 02 TRAIN
write("02_train_disease.ipynb", [
    md("# 02 · Train the disease classifier (fp16 mixed precision)",
       "",
       "Kaggle T4 gives ~2.5 h for 12 epochs on the full 70k-image training set.",
       "Two-stage schedule: 1 epoch head-only, then unfreeze everything at a lower LR.",
       "The number to beat: **val macro-F1**, not accuracy — the classes are imbalanced."),
    SETUP,
    md("### Find the CPU sweet spot first",
       "Each DataLoader worker is a separate process decoding JPEGs. Too few starve the GPU; more than "
       "the vCPU count oversubscribes and everything slows down. Measure it once, then use the winner."),
    code("""
import os, time
from torch.utils.data import DataLoader
from src.data.dataset import LeafDataset, build_transforms, index_folder

TRAIN_DIR = "data/fast/plantvillage/train"   # from data/prepare_fast.py; falls back to the raw dir
items, classes = index_folder(TRAIN_DIR)
ds = LeafDataset(items[:4000], build_transforms(True, 224), classes)
print("vCPUs available:", os.cpu_count())

for w in [0, 1, 2, 4]:
    kw = {"prefetch_factor": 4} if w else {}
    dl = DataLoader(ds, batch_size=64, num_workers=w, pin_memory=True, persistent_workers=w > 0, **kw)
    it = iter(dl); next(it)                       # warm-up batch
    t0 = time.perf_counter(); n = 0
    for i, _ in enumerate(it):
        n += 64
        if i == 19:
            break
    print(f"workers={w}: {n / (time.perf_counter() - t0):6.1f} images/s")
"""),
    md("Use the best worker count below, and try `--batch-size 128` once — if it does not raise "
       "CUDA out-of-memory, it is free throughput."),
    code("""
!python -m src.train --config configs/train_disease.yaml --batch-size 128
"""),
    code("""
import json, pandas as pd, matplotlib.pyplot as plt
h = pd.DataFrame(json.load(open("runs/history.json")))
display(h)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
ax[0].set_title("loss"); ax[0].legend()
ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
ax[1].plot(h.epoch, h.val_acc, label="val accuracy"); ax[1].legend(); ax[1].set_title("metrics")
plt.tight_layout(); plt.savefig("results_training_curve.png", dpi=110); plt.show()
print("best macro-F1:", h.val_macro_f1.max())
"""),
    md("### Before you close this notebook",
       "- Save `runs/best.pt` to the Kaggle output (or push it to a GitHub Release) — it is gitignored on purpose.",
       "- Record the epoch count, batch size, LR and the best macro-F1 in the report table."),
])

# ---------------------------------------------------------------- 03 EVAL + XAI
write("03_evaluate_xai.ipynb", [
    md("# 03 · Evaluate and explain",
       "",
       "Two evaluations, always reported separately:",
       "- **Lab** — PlantVillage valid split.",
       "- **Field-proxy** — PlantDoc test split, remapped to the PlantVillage label space.",
       "",
       "Then Grad-CAM, because the report needs to show the model looks at the leaf, not the background."),
    SETUP,
    code("""
!python -m src.evaluate --ckpt runs/best.pt --data "data/raw/plantvillage/New Plant Diseases Dataset(Augmented)/valid" --tag plantvillage --out results
!python -m src.evaluate --ckpt runs/best.pt --data data/raw/plantdoc/test --field --tag plantdoc --out results
"""),
    code("""
import json, numpy as np, matplotlib.pyplot as plt
m_pv = json.load(open("results/metrics_plantvillage.json"))
m_pd = json.load(open("results/metrics_plantdoc.json"))
print(f"{'dataset':14s} {'acc':>7s} {'macroF1':>8s} {'top3':>7s} {'ECE':>7s}")
for m in (m_pv, m_pd):
    print(f"{m['dataset']:14s} {m['accuracy']:7.4f} {m['macro_f1']:8.4f} {m['top3_accuracy']:7.4f} {m['ece_15bin']:7.4f}")
"""),
    code("""
cm = np.load("results/cm_plantvillage.npy")
fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(cm, cmap="Blues")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("PlantVillage confusion matrix")
plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig("results_cm.png", dpi=110); plt.show()

per_class = sorted(m_pv["per_class"].items(), key=lambda kv: kv[1]["f1-score"])[:8]
print("weakest classes:"); [print(f"  {k:45s} f1={v['f1-score']:.3f}") for k, v in per_class]
"""),
    code("""
!python -m src.xai --ckpt runs/best.pt --data data/raw/plantdoc/test --n 12 --out results/xai
from IPython.display import Image as I
I("results/xai/gradcam_panel.jpg")
"""),
    md("### Reading the Grad-CAM panel",
       "If the heat sits on the background or the image border, the model learned the dataset, not the disease. "
       "That is the single most useful thing this panel can tell you, and it belongs in the report either way."),
])

# ---------------------------------------------------------------- 04 QUANT
write("04_quantize_benchmark.ipynb", [
    md("# 04 · Quantization ladder and benchmark",
       "",
       "fp32 → fp16 → int8 (dynamic) → int8 (static) → int4 weight-only, measured on the same images.",
       "",
       "**fp4 is not measured here.** It needs Blackwell-class hardware and NVIDIA ModelOpt; there is no "
       "ONNX Runtime path that runs on Kaggle. It goes in the report as future work with that reason — "
       "a made-up fp4 row is worse than an honest gap."),
    SETUP,
    code("""
!python -m src.quantize --ckpt runs/best.pt --calib "data/raw/plantvillage/New Plant Diseases Dataset(Augmented)/valid" --out results/quant
"""),
    code("""
import pandas as pd
from IPython.display import Markdown
r = pd.read_json("results/quant/quantization_report.json")
r["size_reduction_x"] = (r.size_mb.max() / r.size_mb).round(2)
display(r[["precision", "size_mb", "size_reduction_x", "latency_ms_per_image", "top1_accuracy"]])
print(open("results/quant/quantization_report.md").read())
"""),
    md("### What to write in the report",
       "One table, one sentence per rung: what it costs in accuracy and what it saves in size. "
       "The deployment pick is almost always int8 static — fp16 buys little on CPU, int4 costs too much accuracy."),
])

# ---------------------------------------------------------------- 05 CROP
write("05_crop_recommender.ipynb", [
    md("# 05 · Crop recommender with SHAP",
       "",
       "2,200 rows, 22 crops, 7 features. This is the module where explainability is easiest to show "
       "and most convincing in a viva: SHAP tells the farmer *why* a crop was recommended."),
    SETUP,
    code("""
from data.download import kaggle_download
from pathlib import Path
kaggle_download("atharvaingle/crop-recommendation-dataset", Path("data/raw/crop"))
"""),
    code("""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

csv = next(Path("data/raw/crop").rglob("*.csv"))
df = pd.read_csv(csv)
X = df.drop(columns=["label"]); y = df["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1).fit(Xtr, ytr)
pred = rf.predict(Xte)
print("macro-F1:", round(f1_score(yte, pred, average="macro"), 4))
print("accuracy:", round((pred == yte).mean(), 4))
"""),
    code("""
import shap
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Xte.iloc[:200])
shap.summary_plot(sv, Xte.iloc[:200], show=True, max_display=10)
"""),
    code("""
# one farmer's recommendation, explained
row = Xte.iloc[[0]]
proba = rf.predict_proba(row)[0]
top = np.argsort(proba)[::-1][:3]
print("input:", row.to_dict("records")[0])
for i in top:
    print(f"  {rf.classes_[i]:12s} p={proba[i]:.3f}")
sv1 = explainer.shap_values(row)[0]
print("\\nwhy:", {f: round(float(v), 3) for f, v in zip(X.columns, sv1)})
"""),
    md("### Caveat to state in the report",
       "The 2,200 rows are synthetic/augmented Indian averages, not Rajasthan farm records. "
       "The model learns plausible agronomy, not local truth — say that plainly and treat the output as advisory."),
])

print("all notebooks written")
