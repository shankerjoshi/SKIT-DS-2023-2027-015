"""Evaluate a checkpoint: accuracy, macro-F1, per-class F1, confusion matrix, calibration.

Reports lab numbers (PlantVillage) and field-proxy numbers (PlantDoc) separately, always.

    python -m src.evaluate --ckpt runs/best.pt --data data/raw/plantvillage/valid --out results/
    python -m src.evaluate --ckpt runs/best.pt --data data/raw/plantdoc/test --field --out results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import LeafDataset, build_transforms, index_folder, map_plantdoc_to_plantvillage
from src.models.disease import build_model
from src.train import run_epoch


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        P.append(torch.softmax(logits.float(), 1).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--field", action="store_true", help="mark this run as field-proxy (PlantDoc)")
    ap.add_argument("--tag", default=None, help="filename tag, e.g. plantvillage or plantdoc")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu")
    classes = ck["classes"]
    model = build_model(ck["arch"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    items, data_classes = index_folder(args.data)
    if args.field:
        # PlantDoc labels differ from PlantVillage: remap so the comparison is meaningful.
        mapping = map_plantdoc_to_plantvillage(classes, data_classes)
        unmapped = [k for k, v in mapping.items() if v is None]
        items = [(p, classes.index(mapping[data_classes[y]])) for p, y in items
                 if mapping.get(data_classes[y]) is not None]
        if unmapped:
            print(f"WARNING: {len(unmapped)} PlantDoc classes have no PlantVillage counterpart and were dropped:")
            for u in unmapped:
                print("   -", u)
        print(f"scored on {len(items)} of the field images")

    ds = LeafDataset(items, build_transforms(False, args.size), classes)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    probs, labels = collect_probs(model, dl, device)
    preds = probs.argmax(1)

    tag = args.tag or ("plantdoc" if args.field else "plantvillage")
    report = classification_report(labels, preds, target_names=[classes[i] for i in sorted(set(labels))],
                                   labels=sorted(set(labels)), output_dict=True, zero_division=0)
    metrics = {
        "dataset": tag,
        "kind": "field-proxy" if args.field else "lab",
        "n_images": int(len(labels)),
        "n_classes_scored": int(len(set(labels.tolist()))),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "ece_15bin": expected_calibration_error(probs, labels),
        "top3_accuracy": float(np.mean([labels[i] in np.argsort(probs[i])[-3:] for i in range(len(labels))])),
        "checkpoint_val_macro_f1": ck.get("val_macro_f1"),
        "per_class": {k: v for k, v in report.items() if isinstance(v, dict)},
    }
    (args.out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    np.save(args.out / f"cm_{tag}.npy", confusion_matrix(labels, preds))
    np.save(args.out / f"probs_{tag}.npy", probs)

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
    print(f"\nwrote metrics_{tag}.json, cm_{tag}.npy, probs_{tag}.npy to {args.out}")
    print("Next: python -m src.xai --ckpt ... --data ... to produce Grad-CAM overlays.")


if __name__ == "__main__":
    main()
