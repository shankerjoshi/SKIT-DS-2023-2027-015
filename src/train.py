"""fp16 mixed-precision training loop with class weights and early stopping.

    python -m src.train --config configs/train_disease.yaml
    python -m src.train --data data/raw/plantvillage --epochs 3 --arch mobilenet_v3_large
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LeafDataset, build_transforms, class_weights, index_folder, split_items
from src.models.disease import build_model, count_params, unfreeze_all


def _make_scaler(amp: bool):
    """GradScaler with the modern API, falling back for torch < 2.3."""
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=True):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, labels_all = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if training else "eval"):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = criterion(out, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            loss_sum += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds_all.append(pred.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "preds": preds,
        "labels": labels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("data/raw/plantvillage/New Plant Diseases Dataset(Augmented)/train"))
    ap.add_argument("--val", type=Path, default=None, help="defaults to <data>/../valid")
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup-epochs", type=int, default=1, help="train head only for this many epochs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (one per vCPU, capped at 8)")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast (CPU debugging)")
    ap.add_argument("--no-channels-last", action="store_true", help="disable NHWC memory format")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (torch 2.x)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.config:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        for k, v in cfg.items():
            if hasattr(args, k) and k != "config":
                setattr(args, k, type(getattr(args, k))(v) if not isinstance(v, Path) else Path(v))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cores = os.cpu_count() or 1
    if args.workers <= 0:
        args.workers = min(cores, 8)
    # One DataLoader worker per vCPU; the main process then stays out of OpenMP's way so the
    # worker processes own the cores they were given.
    if args.workers > 0:
        torch.set_num_threads(1)
    amp = (device.type == "cuda") and not args.no_amp
    print(f"cpu cores={cores} dataloader_workers={args.workers} torch_threads={torch.get_num_threads()}")
    if device.type == "cuda":
        # autotune the conv algorithms for our fixed 224x224 shape
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} amp_fp16={amp} arch={args.arch} channels_last={not args.no_channels_last}")

    items, classes = index_folder(args.data)
    train_items, val_items = split_items(items, 0.15, args.seed)
    if args.val and Path(args.val).is_dir():
        val_items, _ = index_folder(args.val)
    print(f"train={len(train_items)} val={len(val_items)} classes={len(classes)}")

    train_ds = LeafDataset(train_items, build_transforms(True, args.size), classes)
    val_ds = LeafDataset(val_items, build_transforms(False, args.size), classes)
    dl_extra = dict(pin_memory=True, persistent_workers=args.workers > 0)
    if args.workers > 0:
        dl_extra["prefetch_factor"] = 4  # keep the GPU fed; Kaggle CPU is the usual bottleneck
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True, **dl_extra)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, **dl_extra)

    model = build_model(args.arch, len(classes), pretrained=True,
                        freeze_backbone=args.warmup_epochs > 0).to(device)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled (first epoch will be slower while it warms up)")
        except Exception as e:  # noqa: BLE001 - older torch or Windows
            print(f"torch.compile unavailable, continuing eagerly: {e}")
    print(f"trainable params: {count_params(model):,}")

    w = class_weights(train_items, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = _make_scaler(amp)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "classes.json").write_text(json.dumps(classes, indent=2))

    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            print("unfreezing backbone")
            unfreeze_all(model)
            optimizer = AdamW(model.parameters(), lr=args.lr / 3, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs - epoch, 1))
            scaler = _make_scaler(amp)

        t0 = time.time()
        tr = run_epoch(model, train_dl, criterion, device, optimizer, scaler, amp)
        va = run_epoch(model, val_dl, criterion, device, amp=amp)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": round(tr["loss"], 4), "train_acc": round(tr["acc"], 4),
               "val_loss": round(va["loss"], 4), "val_acc": round(va["acc"], 4),
               "val_macro_f1": round(va["macro_f1"], 4), "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row))

        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save({"model": model.state_dict(), "arch": args.arch, "classes": classes,
                        "val_macro_f1": best_f1, "epoch": epoch}, args.out / "best.pt")
            print(f"  saved best.pt (macro_f1={best_f1:.4f})")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
