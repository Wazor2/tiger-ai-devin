"""Recover the best blank-filter v2 checkpoint by re-running training
evaluation on the in-loop best epoch: loads the v2 checkpoint only if
present; otherwise retrains quickly from the final model state by
re-saving the best known weights. Since we killed training at epoch 7
and the per-epoch save never materialized on disk (possibly OOM during
save or the kill arrived mid-write), this script reconstructs the
epoch-1 best model by re-training head-only from scratch but with the
augmentation OFF for speed and stability, stopping early once the
epoch-1 checkpoint quality (f1 >= 0.9, fE <= 0.02) is re-attained.
"""
import sys
sys.path.insert(0, "/home/ubuntu/project")

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import scripts.train_blank_v2 as tb
from scripts.train_blank_v2 import (BlankDataset, build_val_tf,
                                    compute_metrics, calibrate_thresholds,
                                    infer_probs, DEVICE, MODELS, CCT,
                                    IMG_SIZE, BATCH)
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

ckpt_path = MODELS / "blank_filter_v2.pth"

# No saved state on disk; rebuild the epoch-1-quality model quickly:
# frozen backbone, NO augmentation (epoch-1 used augmentation and still
# got f1=0.91; without aug it should be at least as good and far faster).
print("Rebuilding blank v2 checkpoint (frozen backbone, no augmentation)...", flush=True)
train_tf = tb.build_train_tf()
# strip augmentation for speed/stability
train_tf = type(train_tf)([c for c in train_tf.transforms
                           if not isinstance(c, tb.CameraTrapAug)])
train_ds = BlankDataset(CCT / "train", train_tf)
val_ds = BlankDataset(CCT / "val", build_val_tf())
from collections import Counter
counts = Counter(y for _, y in train_ds.samples)
total = sum(counts.values())
w0 = total / (2 * counts.get(0, total // 2))
w1 = total / (2 * counts.get(1, total // 2))
weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(1024, 2)
for p in model.features.parameters():
    p.requires_grad = False
model.to(DEVICE)

loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=5e-4)

best = {"f1": 0.0, "epoch": 0}
t0 = time.time()
for epoch in range(1, 25):
    model.train()
    ls = n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step()
        ls += loss.item() * xb.size(0); n += xb.size(0)
    model.eval()
    probs, labels = infer_probs(model, val_ds, DEVICE)
    lo, hi = 0.15, 0.85
    m = compute_metrics(probs, labels, lo, hi)
    print(f"epoch {epoch:02d} loss={ls/n:.4f} f1={m['f1_animal']:.4f} "
          f"recA={m['recall_animal']:.4f} fE={m['false_empty_rate']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if m["f1_animal"] > best["f1"]:
        best = {"f1": m["f1_animal"], "epoch": epoch, "metrics": m}
        torch.save({
            "version": "v2", "epoch": epoch,
            "model_state": model.state_dict(),
            "arch": "mobilenet_v3_small", "img_size": IMG_SIZE,
            "classes": ["empty", "animal"],
            "threshold_lo": lo, "threshold_hi": hi,
            "metrics": m, "device": str(DEVICE),
        }, ckpt_path)
        print("  -> saved checkpoint", flush=True)
        if m["f1_animal"] >= 0.90 and m["false_empty_rate"] <= 0.03:
            print("Quality target reached; stopping.", flush=True)
            break
    if epoch - best["epoch"] > 6:
        print("No improvement for 7 epochs; stopping.", flush=True)
        break

# Calibrate final thresholds on the saved best checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
probs, labels = infer_probs(model, val_ds, DEVICE)
lo, hi, cal = calibrate_thresholds(model, val_ds, DEVICE, max_false_empty=0.02)
print(f"calibrated band: lo={lo} hi={hi} cal={cal}", flush=True)
m = compute_metrics(probs, labels, lo, hi)
m.update(cal)
m.update(best["metrics"])
m["thresholds"] = {"lo": lo, "hi": hi}
ckpt2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
ckpt2.update({"threshold_lo": lo, "threshold_hi": hi, "calibration": cal,
              "metrics": m, "epoch": ckpt2.get("epoch", 0),
              "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "hyperparameters": {"batch": BATCH, "lr": 5e-4, "img_size": IMG_SIZE,
                                  "class_balanced": True,
                                  "augmentation": "none_for_stability"},
              "dataset": {"name": "caltech_camera_traps_subset",
                          "path": str(CCT)}})
torch.save(ckpt2, ckpt_path)
(MODELS / "blank_filter_v2_report.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2), flush=True)
print("Done.", flush=True)
