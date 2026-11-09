"""Dataset indexing, transforms and the PlantVillage -> PlantDoc label bridge."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def index_folder(root: str | Path) -> tuple[list[tuple[str, int]], list[str]]:
    """Walk a class-per-folder tree. Returns (path, label) pairs and the class names."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    items: list[tuple[str, int]] = []
    for idx, cls in enumerate(classes):
        for p in (root / cls).rglob("*"):
            if p.suffix in IMG_EXT:
                items.append((str(p), idx))
    if not items:
        raise RuntimeError(f"no images under {root} — wrong folder level?")
    return items, classes


def split_items(items: Sequence[tuple[str, int]], val_frac: float = 0.15, seed: int = 42
                ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Stratified-ish split: shuffle inside each class so rare classes appear in both halves."""
    by_class: dict[int, list[tuple[str, int]]] = {}
    for it in items:
        by_class.setdefault(it[1], []).append(it)
    rng = random.Random(seed)
    train, val = [], []
    for _, group in sorted(by_class.items()):
        rng.shuffle(group)
        cut = max(1, int(len(group) * val_frac))
        val.extend(group[:cut])
        train.extend(group[cut:])
    return train, val


def build_transforms(train: bool, size: int = 224, strong: bool = True):
    if train:
        aug = [
            transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
        ]
        if strong:
            aug += [
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.06),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            ]
        aug += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(aug)
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LeafDataset(Dataset):
    def __init__(self, items: Iterable[tuple[str, int]], transform=None, class_names: Sequence[str] | None = None):
        self.items = list(items)
        self.transform = transform or build_transforms(False)
        self.class_names = list(class_names) if class_names else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        return self.transform(im), label


def class_weights(items: Sequence[tuple[str, int]], n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights — PlantVillage is badly imbalanced (healthy vs. rust etc.)."""
    counts = torch.zeros(n_classes)
    for _, y in items:
        counts[y] += 1
    counts = counts.clamp(min=1)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


# ---------------------------------------------------------------- label bridge
def normalise(name: str) -> str:
    """'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot' -> 'corn cercospora leaf spot'."""
    s = name.replace("___", "|").replace("_", " ").lower()
    s = s.replace("(maize)", "corn").replace("(including sour)", "")
    for junk in ("|", "  "):
        s = s.replace(junk, " ")
    return " ".join(s.split())


def map_plantdoc_to_plantvillage(pv_classes: Sequence[str], pd_classes: Sequence[str]) -> dict[str, str | None]:
    """Best-effort mapping for cross-dataset evaluation.

    PlantDoc has 27 classes over 13 species; PlantVillage has 38. Species-level mapping is
    reliable, disease-level is not — so callers should report both 'exact' and 'species-only'
    agreement. Returns PlantDoc class -> PlantVillage class (or None when unmappable).
    """
    pv_norm = {normalise(c): c for c in pv_classes}
    species_map: dict[str, str | None] = {}
    for pd_cls in pd_classes:
        n = normalise(pd_cls)
        if n in pv_norm:
            species_map[pd_cls] = pv_norm[n]
            continue
        species = n.split()[0]
        candidates = [pv for k, pv in pv_norm.items() if k.split()[0] == species]
        species_map[pd_cls] = candidates[0] if len(candidates) == 1 else None
    return species_map
