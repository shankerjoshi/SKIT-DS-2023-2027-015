"""Pre-resize the dataset once so training epochs stop paying for JPEG decode.

On Kaggle the GPU is almost never the bottleneck — decoding 70,000 full-size JPEGs every epoch is.
Resizing each image to (size + 32) on the short side and re-saving at quality 92 typically makes an
epoch 1.5-3x faster and lets the OS page cache hold much more of the set.

    python data/prepare_fast.py --src "data/raw/plantvillage/New Plant Diseases Dataset(Augmented)/train" \
                                --dst data/fast/plantvillage/train --size 224 --workers 4

Then train with --data data/fast/plantvillage/train. Nothing else changes.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def _one(job: tuple[str, str, int, int]) -> tuple[bool, str]:
    src, dst, size, quality = job
    out = Path(dst)
    if out.exists():
        return True, "skipped"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
            im.save(out, "JPEG", quality=quality, optimize=False)
        return True, "ok"
    except Exception as e:  # noqa: BLE001 - a corrupt file should not kill a 70k run
        return False, f"{src}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224, help="target short side; keep 32 px above the crop size")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src_files = [p for p in args.src.rglob("*") if p.suffix in IMG_EXT]
    if not src_files:
        raise SystemExit(f"no images under {args.src} — wrong folder level?")
    print(f"{len(src_files)} images: {args.src} -> {args.dst} (short side {args.size}px, q{args.quality})")

    jobs = [
        (str(p), str(args.dst / p.relative_to(args.src).with_suffix(".jpg")), args.size, args.quality)
        for p in src_files
    ]

    t0 = time.time()
    ok = bad = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (success, msg) in enumerate(pool.map(_one, jobs, chunksize=64), 1):
            ok += success
            if not success:
                bad += 1
                if bad <= 5:
                    print("  FAILED", msg)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}  {i / (time.time() - t0):.0f} img/s")

    dt = time.time() - t0
    size_gb = sum(f.stat().st_size for f in args.dst.rglob("*.jpg")) / 1e9
    print(f"\ndone in {dt:.1f}s ({ok} written, {bad} failed) — cache is {size_gb:.2f} GB")
    if bad:
        print("failed files are listed above (first 5); they are skipped, not fatal")


if __name__ == "__main__":
    main()
