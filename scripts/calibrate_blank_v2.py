"""Sweep the blank-filter quarantine band on a held-out validation set.

Fix-spec item 2: the shipped `models/blank_filter_v2.pth` carries a
[0.20, 0.75] band whose review rate is 84%, which is unusable operationally.
This script re-derives the band from the model's actual validation score
distribution instead of trusting the stored numbers, and reports the whole
safety/review frontier so the chosen operating point is auditable.

Policy: p_animal >= hi -> animal (auto-accept)
        p_animal <= lo -> empty  (auto-archive, the only unsafe direction)
        otherwise      -> review (quarantine)

Constraint: false_empty_rate = P(decide empty | truly animal) <= --max-false-empty.
Objective:  minimise review_rate.

Usage:
    python -m scripts.calibrate_blank_v2                      # report only
    python -m scripts.calibrate_blank_v2 --apply              # write band into ckpt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_blank_v2 import (BlankDataset, build_val_tf,  # noqa: E402
                                    compute_metrics, infer_probs, CCT, MODELS)

DEVICE = torch.device("cpu")


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = mobilenet_v3_small()
    model.classifier[3] = nn.Linear(1024, 2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def sweep(probs, labels, max_false_empty=0.02, max_false_animal=0.10):
    """Return (best_band, frontier) over a fine (lo, hi) grid."""
    animal, empty = labels == 1, labels == 0
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    best, frontier = None, []
    for hi in grid:
        for lo in grid[grid < hi]:
            f_empty = float((probs[animal] <= lo).mean())
            if f_empty > max_false_empty:
                continue
            f_animal = float((probs[empty] >= hi).mean())
            review = float(((probs > lo) & (probs < hi)).mean())
            cand = {"lo": float(lo), "hi": float(hi),
                    "false_empty_rate": f_empty,
                    "false_animal_rate": f_animal,
                    "review_rate": review}
            if f_animal <= max_false_animal and (best is None or
                                                 review < best["review_rate"]):
                best = cand
            frontier.append(cand)
    return best, frontier


def review_floor_by_false_animal(frontier):
    """Lowest achievable review rate at each false-animal budget."""
    out = {}
    for budget in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00):
        ok = [c for c in frontier if c["false_animal_rate"] <= budget]
        if ok:
            b = min(ok, key=lambda c: c["review_rate"])
            out[f"false_animal<={budget:.2f}"] = {
                "review_rate": round(b["review_rate"], 4),
                "lo": b["lo"], "hi": b["hi"],
                "false_empty_rate": round(b["false_empty_rate"], 4),
                "false_animal_rate": round(b["false_animal_rate"], 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS / "blank_filter_v2.pth"))
    ap.add_argument("--val-dir", default=str(CCT / "val"))
    ap.add_argument("--max-false-empty", type=float, default=0.02)
    ap.add_argument("--max-false-animal", type=float, default=0.10)
    ap.add_argument("--apply", action="store_true",
                    help="write the chosen band into the checkpoint")
    ap.add_argument("--out", default=str(MODELS / "blank_filter_v2_calibration.json"))
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise SystemExit(f"validation set missing: {val_dir}\n"
                         "run: python -m scripts.fetch_cct_subset")

    model, ckpt = load_model(ckpt_path)
    val_ds = BlankDataset(val_dir, build_val_tf())
    probs, labels = infer_probs(model, val_ds, DEVICE)
    print(f"val n={len(labels)} animal={int((labels == 1).sum())} "
          f"empty={int((labels == 0).sum())}", flush=True)

    animal, empty = labels == 1, labels == 0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    dist = {
        "roc_auc": round(auc, 4),
        "all": {f"p{q}": round(float(np.percentile(probs, q)), 4)
                for q in (1, 5, 25, 50, 75, 95, 99)},
        "animal": {f"p{q}": round(float(np.percentile(probs[animal], q)), 4)
                   for q in (1, 5, 25, 50, 75, 95, 99)},
        "empty": {f"p{q}": round(float(np.percentile(probs[empty], q)), 4)
                  for q in (1, 5, 25, 50, 75, 95, 99)},
    }
    print("score distribution:", json.dumps(dist, indent=2), flush=True)

    best, frontier = sweep(probs, labels, args.max_false_empty,
                           args.max_false_animal)
    stored_lo = ckpt.get("threshold_lo")
    stored_hi = ckpt.get("threshold_hi")
    stored = None
    if stored_lo is not None and stored_hi is not None:
        stored = {"lo": stored_lo, "hi": stored_hi,
                  "false_empty_rate": round(float((probs[animal] <= stored_lo).mean()), 4),
                  "false_animal_rate": round(float((probs[empty] >= stored_hi).mean()), 4),
                  "review_rate": round(float(((probs > stored_lo) &
                                              (probs < stored_hi)).mean()), 4)}

    report = {
        "checkpoint": str(ckpt_path.name),
        "checkpoint_epoch": ckpt.get("epoch"),
        "validation_set": str(val_dir),
        "n_validation": int(len(labels)),
        "score_distribution": dist,
        "stored_band_on_this_val_set": stored,
        "constraints": {"max_false_empty": args.max_false_empty,
                        "max_false_animal": args.max_false_animal},
        "best_band": None,
        "review_floor_by_false_animal_budget": review_floor_by_false_animal(frontier),
    }

    if best is None:
        report["conclusion"] = (
            "no (lo, hi) band satisfies both constraints on this validation "
            "set; the score distribution is not separable enough")
    else:
        m = compute_metrics(probs, labels, best["lo"], best["hi"])
        report["best_band"] = {**best, "metrics": m}
        report["conclusion"] = (
            f"band [{best['lo']:.2f}, {best['hi']:.2f}] gives review "
            f"{best['review_rate']:.1%} at false-empty "
            f"{best['false_empty_rate']:.2%}")
        if args.apply:
            ckpt["threshold_lo"] = best["lo"]
            ckpt["threshold_hi"] = best["hi"]
            ckpt["calibration"] = {
                "method": "grid_sweep_min_review_under_false_empty_cap",
                "max_false_empty": args.max_false_empty,
                "max_false_animal": args.max_false_animal,
                "validation_set": str(val_dir),
                "n_validation": int(len(labels)),
                "false_empty_rate": best["false_empty_rate"],
                "false_animal_rate": best["false_animal_rate"],
                "review_rate": best["review_rate"],
            }
            torch.save(ckpt, ckpt_path)
            print(f"applied band to {ckpt_path}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "score_distribution"}, indent=2), flush=True)
    print(f"\nreport written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
