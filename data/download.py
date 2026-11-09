"""Download and verify every dataset this project uses.

Kaggle datasets need an API token at ~/.kaggle/kaggle.json (Kaggle -> Account -> Create New Token).

    python data/download.py --what all --dest data/raw
    python data/download.py --what weather --dest data/raw --lat 26.91 --lon 75.79
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

KAGGLE = {
    # key: (kaggle slug, expected top-level folder, description)
    "plantvillage": (
        "vipoooool/new-plant-diseases-dataset",
        "New Plant Diseases Dataset(Augmented)",
        "87.9k images, 38 classes, pre-split train/valid",
    ),
    "plantdoc": (
        "nirmalsankalana/plantdoc-dataset",
        None,
        "2,598 field images, 27 classes, train/test folders",
    ),
    "crop": (
        "atharvaingle/crop-recommendation-dataset",
        None,
        "2,200 rows, 22 crops",
    ),
}

GITHUB_PLANTVILLAGE = "https://github.com/spMohanty/PlantVillage-Dataset.git"

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def kaggle_download(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def verify_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sum(1 for p in root.rglob("*") if p.suffix in exts)


def weather_snapshot(dest: Path, lat: float, lon: float, days: int = 7) -> Path:
    """Pull a forecast + ET0 series. Open-Meteo is free and needs no key."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_max",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration",
        ]),
        "forecast_days": days,
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=30)
    r.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"weather_{lat:.2f}_{lon:.2f}.json"
    out.write_text(json.dumps(r.json(), indent=2))
    print(f"  saved {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all", choices=["all", "plantvillage", "plantdoc", "crop", "weather", "verify"])
    ap.add_argument("--dest", type=Path, default=Path("data/raw"))
    ap.add_argument("--lat", type=float, default=26.9124, help="Jaipur by default")
    ap.add_argument("--lon", type=float, default=75.7873)
    ap.add_argument("--github-plantvillage", action="store_true",
                    help="fetch the original 54,305-image repo instead of the Kaggle mirror")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    todo = ["plantvillage", "plantdoc", "crop", "weather"] if args.what == "all" else [args.what]

    for name in todo:
        print(f"[{name}]")
        if name == "weather":
            weather_snapshot(args.dest / "weather", args.lat, args.lon)
        elif name == "verify":
            continue
        elif name == "plantvillage" and args.github_plantvillage:
            target = args.dest / "plantvillage_github"
            if target.exists():
                print("  already present")
            else:
                subprocess.run(["git", "clone", "--depth", "1", GITHUB_PLANTVILLAGE, str(target)], check=True)
        else:
            slug, _, desc = KAGGLE[name]
            print(f"  {desc}")
            kaggle_download(slug, args.dest / name)

    print("\n[verify]")
    for name in ("plantvillage", "plantdoc"):
        p = args.dest / name
        if p.exists():
            n = verify_images(p)
            flag = "OK" if n > 1000 else "SUSPICIOUS - check the download"
            print(f"  {name:14s} {n:>7d} images  {flag}")
    crop = list((args.dest / "crop").rglob("*.csv")) if (args.dest / "crop").exists() else []
    for c in crop:
        print(f"  {c.name}: {sum(1 for _ in c.open()) - 1} rows")

    print("\nDone. Datasets are gitignored; never commit them.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"elapsed {time.time() - t0:.1f}s")
