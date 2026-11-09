"""Explainability: Grad-CAM for the vision model, SHAP for the tabular models.

Grad-CAM is implemented directly with forward/backward hooks so there is no extra dependency.
Every saved overlay is meant to go into the report — a model a farmer cannot trust is a model
they will not use.

    python -m src.xai --ckpt runs/best.pt --data data/raw/plantdoc/test --n 12 --out results/xai
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, index_folder
from src.models.disease import build_model, target_layer_for_gradcam


class GradCAM:
    """Hook-based Grad-CAM. One instance per model; call .generate(x) per batch."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._h1 = layer.register_forward_hook(self._fwd)
        self._h2 = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, idx

    def close(self):
        self._h1.remove()
        self._h2.remove()


def overlay(path: str, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend the heatmap on the original image (no matplotlib dependency for the blend)."""
    base = Image.open(path).convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heat = np.zeros((*cam.shape, 3), dtype=np.uint8)
    heat[..., 0] = (cam * 255).astype(np.uint8)              # red = high attention
    heat[..., 1] = ((1 - cam) * 120).astype(np.uint8)
    heat_img = Image.fromarray(heat)
    return Image.blend(base, heat_img, alpha)


def save_panel(rows: list[tuple[Image.Image, str]], out_path: Path, cols: int = 4) -> None:
    """Tile overlays into one contact sheet for the report."""
    if not rows:
        return
    w, h = rows[0][0].size
    n = len(rows)
    r = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, r * h), "white")
    for i, (im, _) in enumerate(rows):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path, quality=90)


def occlusion_sensitivity(model, x: torch.Tensor, patch: int = 32, stride: int = 32) -> np.ndarray:
    """Delete-one-patch map. Slow but assumption-free; use on ~10 images for the report."""
    model.eval()
    with torch.no_grad():
        base = torch.softmax(model(x), 1).max().item()
    _, _, H, W = x.shape
    heat = np.zeros((H // stride, W // stride))
    with torch.no_grad():
        for i, y in enumerate(range(0, H - patch + 1, stride)):
            for j, xx in enumerate(range(0, W - patch + 1, stride)):
                xo = x.clone()
                xo[..., y:y + patch, xx:xx + patch] = 0
                heat[i, j] = base - torch.softmax(model(xo), 1).max().item()
    return heat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/xai"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--occlusion", action="store_true", help="also run occlusion sensitivity on the first image")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    classes = ck["classes"]
    model = build_model(ck["arch"], len(classes), pretrained=False).to(device).eval()
    model.load_state_dict(ck["model"])

    cam_engine = GradCAM(model, target_layer_for_gradcam(model, ck["arch"]))
    tf = build_transforms(False, args.size)
    items, _ = index_folder(args.data)

    rows, log = [], []
    for path, label in items[: args.n]:
        with Image.open(path) as im:
            im = im.convert("RGB")
        x = tf(im).unsqueeze(0).to(device)
        cam, idx = cam_engine.generate(x)
        rows.append((overlay(path, cam), classes[idx]))
        log.append({"file": Path(path).name, "true": classes[label], "pred": classes[idx],
                    "correct": bool(idx == label)})

    save_panel(rows, args.out / "gradcam_panel.jpg")
    if args.occlusion and items:
        with Image.open(items[0][0]) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        occ = occlusion_sensitivity(model, x)
        np.save(args.out / "occlusion.npy", occ)
        print(f"occlusion map saved (mean sensitivity {occ.mean():.4f})")

    cam_engine.close()
    (args.out / "gradcam_log.json").write_text(json.dumps(log, indent=2))
    acc = np.mean([r["correct"] for r in log]) if log else 0.0
    print(f"wrote {args.out}/gradcam_panel.jpg  (spot-check accuracy on these {len(log)}: {acc:.2f})")


if __name__ == "__main__":
    main()
