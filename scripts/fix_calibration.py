"""Properly calibrate the blank v2 checkpoint thresholds.

Problem: calibrate_thresholds returned lo=0.171, hi=0.975 because it only
searches the tail of the probability distribution. With the frozen model,
most probabilities cluster away from 1.0, so hi near 0.975 auto-accepts
nothing. We instead sweep (lo, hi) over a grid, keeping bands where
false_empty <= 0.02 and false_animal <= 0.10, and pick the band that
minimizes review rate.
"""
import sys
sys.path.insert(0, "/home/ubuntu/project")

import json
import time
from pathlib import Path

import numpy as np
import torch

import scripts.train_blank_v2 as tb
from scripts.train_blank_v2 import (BlankDataset, build_val_tf,
                                    compute_metrics, infer_probs, DEVICE,
                                    MODELS)

ckpt_path = MODELS / "blank_filter_v2.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = tb.build_model() if hasattr(tb, "build_model") else None
if model is None:
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(1024, 2)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)

val_ds = BlankDataset(tb.CCT / "val", build_val_tf())
probs, labels = infer_probs(model, val_ds, DEVICE)

print(f"prob distribution: p5={np.percentile(probs,5):.3f} "
      f"p50={np.percentile(probs,50):.3f} p95={np.percentile(probs,95):.3f}", flush=True)

# grid search over candidate bands
animal_mask = labels == 1
empty_mask = labels == 0
best = None
for hi in np.arange(0.40, 0.901, 0.05):
    for lo in np.arange(0.05, hi - 0.05, 0.05):
        fE = float((probs[animal_mask] < lo).mean())
        fA = float((probs[empty_mask] > hi).mean())
        review = float(((probs >= lo) & (probs <= hi)).mean())
        if fE <= 0.02 and fA <= 0.10 and (best is None or review < best["review"]):
            m = compute_metrics(probs, labels, lo, hi)
            best = {"lo": lo, "hi": hi, "false_empty": fE,
                    "false_animal": fA, "review": review, "metrics": m}

if best is None:
    # fallback: use lo=0, hi=1 (nothing auto-blanked, max safety)
    best = {"lo": 0.0, "hi": 1.0, "false_empty": 0.0, "false_animal": 0.0,
            "review": 1.0, "metrics": compute_metrics(probs, labels, 0.0, 1.0)}

lo, hi = best["lo"], best["hi"]
m = best["metrics"]
m.update({"false_empty_rate": best["false_empty"],
          "false_animal_rate": best["false_animal"],
          "review_rate": best["review"],
          "thresholds": {"lo": lo, "hi": hi}})

ckpt2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
ckpt2.update({"threshold_lo": lo, "threshold_hi": hi,
              "calibration": {"method": "grid_search",
                              "max_false_empty": 0.02,
                              "max_false_animal": 0.10},
              "metrics": m,
              "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "hyperparameters": {"batch": 256, "lr": 5e-4, "img_size": 224,
                                  "class_balanced": True, "augmentation": "none"},
              "dataset": {"name": "caltech_camera_traps_subset",
                          "path": str(tb.CCT)}})
torch.save(ckpt2, ckpt_path)
(MODELS / "blank_filter_v2_report.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2), flush=True)
print("Done.", flush=True)
