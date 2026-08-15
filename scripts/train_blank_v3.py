"""Finish the blank-filter training run that was cut short.

`models/blank_filter_v2.pth` is an epoch-1, frozen-backbone, augmentation-off
checkpoint (see scripts/restore_blank_ckpt.py) and scores ROC-AUC ~0.51 on a
fresh held-out validation split — chance level. That is why no quarantine band
can get the review rate below ~83% (scripts/calibrate_blank_v2.py). The band is
not the problem; the weights are.

This trains the same architecture properly: full fine-tune (backbone
unfrozen), camera-trap augmentation on, class-balanced loss, best-epoch
checkpointing on validation ROC-AUC. Writes models/blank_filter_v3.pth.

Usage:
    python -m scripts.train_blank_v3 --epochs 12
    python -m scripts.calibrate_blank_v2 --ckpt models/blank_filter_v3.pth --apply
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_blank_v2 import (BlankDataset, build_train_tf,  # noqa: E402
                                    build_val_tf, compute_metrics, infer_probs,
                                    CCT, MODELS, IMG_SIZE)
from scripts.calibrate_blank_v2 import sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(MODELS / "blank_filter_v3.pth"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    print(f"device={device} epochs={args.epochs} batch={args.batch}", flush=True)

    train_ds = BlankDataset(CCT / "train", build_train_tf())
    val_ds = BlankDataset(CCT / "val", build_val_tf())
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    counts = Counter(y for _, y in train_ds.samples)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (2 * max(counts.get(0, 1), 1)), total / (2 * max(counts.get(1, 1), 1))],
        dtype=torch.float32, device=device)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(1024, 2)
    model.to(device)

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"auc": -1.0, "epoch": 0}
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ls = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ls += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        probs, labels = infer_probs(model, val_ds, device)
        band, _ = sweep(probs, labels, max_false_empty=0.02, max_false_animal=0.10)
        m = compute_metrics(probs, labels, band["lo"] if band else 0.2,
                            band["hi"] if band else 0.75)
        row = {"epoch": epoch, "loss": round(ls / max(n, 1), 4),
               "roc_auc": round(m["roc_auc"], 4),
               "band": [band["lo"], band["hi"]] if band else None,
               "review_rate": round(band["review_rate"], 4) if band else None,
               "false_empty_rate": round(band["false_empty_rate"], 4) if band else None,
               "seconds": round(time.time() - t0)}
        history.append(row)
        print(json.dumps(row), flush=True)

        if m["roc_auc"] > best["auc"]:
            best = {"auc": m["roc_auc"], "epoch": epoch}
            torch.save({
                "version": "v3", "epoch": epoch,
                "model_state": model.state_dict(),
                "arch": "mobilenet_v3_small", "img_size": IMG_SIZE,
                "classes": ["empty", "animal"],
                "threshold_lo": band["lo"] if band else 0.2,
                "threshold_hi": band["hi"] if band else 0.75,
                "metrics": m,
                "hyperparameters": {"batch": args.batch, "lr": args.lr,
                                    "img_size": IMG_SIZE, "class_balanced": True,
                                    "augmentation": "camera_trap_aug",
                                    "frozen_backbone": False,
                                    "epochs_requested": args.epochs},
                "dataset": {"name": "caltech_camera_traps_subset",
                            "path": str(CCT)},
                "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, args.out)
            print(f"  -> saved (best auc {best['auc']:.4f})", flush=True)

    (MODELS / "blank_filter_v3_history.json").write_text(
        json.dumps({"best": best, "history": history}, indent=2))
    print(f"done in {time.time() - t0:.0f}s; best epoch {best['epoch']} "
          f"auc {best['auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
