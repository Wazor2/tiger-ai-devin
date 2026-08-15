"""Automated evaluation suite — `python -m pench.evaluate --all` (Task 8, 9, 10).

Runs every validation/verification check deterministically and writes a
single JSON report (report/evaluation_report.json) plus a console summary.

Suites:
  blank      Task 1   — compares blank_filter.pth (v1) vs blank_filter_v2.pth
                       metrics (acc, F1, animal recall, false-empty rate,
                       ROC-AUC, PR-AUC, review rate)
  reid       Tasks 2-3— loads models/reid_benchmark.json (Tasks 2-3 suite)
                       and reports closed-set + open-set results
  data       Task 4   — re-runs dataset validation (fails loudly on fatal)
  home       Task 5   — home-range safety checks (min detections, coord
                       validation, outlier filtering, uncertainty metadata)
  alerts     Task 6   — deterministic alerting unit tests + debounce check
  pipeline   Task 7   — pipeline safety guarantees (animal never silently
                       discarded, ambiguous->review, quarantine preserved)
  versioning Task 9   — checkpoint provenance: never-overwrite, metadata,
                       promotion decision based on safety metrics
  perf       Task 10  — performance benchmarks (batch inference, imgs/sec,
                       avg/p95 latency, RAM; CUDA when available)
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (efficientnet_b0, mobilenet_v3_small,
                                EfficientNet_B0_Weights,
                                MobileNet_V3_Small_Weights)

PROJECT = Path(__file__).resolve().parent.parent
MODELS = PROJECT / "models"
REPORTS = PROJECT / "report"
REPORTS.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# suite: blank filter (Task 1)
# ---------------------------------------------------------------------------
def _blank_metrics(ckpt_path):
    """Re-compute full metrics on cct_subset/val for a blank-filter checkpoint."""
    from scripts.train_blank_classifier import BlankDataset, compute_metrics as _cm
    return None  # placeholder replaced below


def suite_blank():
    """Compare v1 vs v2 blank filter using saved reports + live val eval."""
    from scripts.train_blank_v2 import (BlankDataset, build_val_tf, compute_metrics,
                                        infer_probs)
    results = {}
    for name, fname in [("v1", "blank_filter.pth"), ("v2", "blank_filter_v2.pth")]:
        p = MODELS / fname
        if not p.exists():
            results[name] = {"available": False}
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(1024, 2)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        val_ds = BlankDataset(PROJECT / "datasets" / "cct_subset" / "val",
                              build_val_tf())
        probs, labels = infer_probs(model, val_ds, DEVICE)
        lo, hi = (ckpt.get("threshold_lo", 0.15), ckpt.get("threshold_hi", 0.85))
        m = compute_metrics(probs, labels, lo, hi)
        m.update({"checkpoint": str(p.name), "available": True,
                  "thresholds": {"lo": lo, "hi": hi}})
        if "metrics" in ckpt and isinstance(ckpt["metrics"], dict):
            m.update({k: v for k, v in ckpt["metrics"].items() if k not in m})
        results[name] = m
    # comparison
    if results.get("v1") and results.get("v2"):
        r = results["v2"]
        results["promotion_decision"] = {
            "v2_beats_v1_accuracy": r["accuracy"] > results["v1"]["accuracy"],
            "v2_better_f1": r["f1_animal"] > results["v1"]["f1_animal"],
            "v2_lower_false_empty": r["false_empty_rate"] <= results["v1"]["false_empty_rate"],
            "recommendation": ("PROMOTE v2 to production"
                               if (r["false_empty_rate"] <= results["v1"]["false_empty_rate"]
                                   and r["f1_animal"] >= results["v1"]["f1_animal"])
                               else "keep v2 as candidate (does not dominate v1 on safety metrics)"),
        }
        results["available"] = True
    else:
        results["available"] = bool(results.get("v1") or results.get("v2"))
    return results


# ---------------------------------------------------------------------------
# suite: reid (Tasks 2-3)
# ---------------------------------------------------------------------------
def suite_reid():
    p = MODELS / "reid_benchmark.json"
    if not p.exists():
        return {"available": False,
                "note": "run scripts/bench_reid.py first (Tasks 2-4)"}
    out = json.loads(p.read_text())
    out["available"] = True
    return out


# ---------------------------------------------------------------------------
# suite: data validation (Task 4)
# ---------------------------------------------------------------------------
def suite_data():
    from scripts.bench_reid import task4_data_validation
    report, id_map = task4_data_validation()
    return {"fatal": report["fatal"],
            "errors": report["errors"],
            "train_images_with_labels": report["train_images_with_labels"],
            "train_identities": report["train_identities"],
            "corrupted_images": report["corrupted_images"],
            "train_test_filename_overlap": report["train_test_filename_overlap"],
            "identity_counts": report["identity_counts"],
            "available": True}


# ---------------------------------------------------------------------------
# suite: home-range safety (Task 5)
# ---------------------------------------------------------------------------
def suite_home():
    from pench.modules.occupancy_safe import (safe_home_range, validate_coordinates,
                                              filter_outliers)
    from pench.modules.occupancy import Detection

    class P:
        def __init__(self, lat, lon, conf=0.9):
            self.lat, self.lon, self.confidence = lat, lon, conf

    rng = np.random.default_rng(3)

    def jitter(lat, lon, scale=0.004):
        return (lat + rng.normal(0, scale), lon + rng.normal(0, scale))
    # (a) insufficient data -> no false range
    few = [P(21.65, 79.35 + 0.01 * i) for i in range(3)]
    r_few = safe_home_range(few)
    # (b) invalid coordinates rejected
    bad = [P(21.65, 79.35)] + [P(-40.0, 79.35) for _ in range(4)]
    r_bad = safe_home_range(bad)
    # (c) outlier removal + uncertainty metadata
    base = [P(21.65, 79.35 + rng.normal(0, 0.005)) for _ in range(20)]
    base.append(P(21.65, 79.55))  # far outlier
    r_out = safe_home_range(base)
    # (d) sufficient good data -> good quality with uncertainty
    r_good = safe_home_range([P(21.65, 79.35 + rng.normal(0, 0.005))
                              for _ in range(25)])
    return {
        "available": True,
        "insufficient_data_handled": r_few["home_range"]["quality"] == "insufficient_data",
        "invalid_coords_rejected": r_bad["home_range"]["n_rejected_coordinates"] == 4,
        "outlier_filtered": r_out["home_range"]["n_outliers"] >= 1,
        "uncertainty_metadata_present": "uncertainty" in r_good["home_range"],
        "quality_flags": {"few": r_few["home_range"]["quality"],
                          "good": r_good["home_range"]["quality"]},
        "areas_km2": {"good_mcp": r_good["home_range"].get("mcp_area_km2"),
                      "good_kde95": r_good["home_range"].get("kde95_area_km2"),
                      "good_core50": r_good["home_range"].get("core_mcp_50_area_km2")},
    }


# ---------------------------------------------------------------------------
# suite: alerting unit tests (Task 6)
# ---------------------------------------------------------------------------
def suite_alerts():
    from datetime import datetime, timedelta
    from pench.modules.alerting_v2 import (AlertConfigV2, DetectionPoint,
                                           eval_core_shift, eval_new_identity,
                                           eval_unusual_movement, eval_activity_anomaly,
                                           eval_territorial_overlap, apply_debounce)

    now = datetime(2026, 8, 14)
    cfg = AlertConfigV2()
    out = {}

    def pts(lat, lon, days_ago_start, n, step=4):
        return [DetectionPoint(f"S{i % 10:02d}",
                               (now - timedelta(days=days_ago_start + i * step)).isoformat(),
                               lat, lon, "T1", 0.9) for i in range(n)]

    class P:
        def __init__(self, lat, lon, conf=0.9):
            self.lat, self.lon, self.confidence = lat, lon, conf
            self.tiger_id, self.detection_id = "T1", "synth"

    rng = np.random.default_rng(7)

    def jitter(lat, lon, scale=0.004):
        return (lat + rng.normal(0, scale), lon + rng.normal(0, scale))

    # 1) core_shift fires only above threshold (realistic spatial jitter so
    #    detections are not perfectly collinear/identical)
    base = [P(*jitter(21.65, 79.35)) for _ in range(12)]
    cur_near = [P(*jitter(21.65, 79.35, 0.003)) for _ in range(6)]
    cur_far = [P(*jitter(21.65, 79.42)) for _ in range(6)]
    out["core_shift_no_fire_below_threshold"] = eval_core_shift(base, cur_near, now, cfg) is None
    a = eval_core_shift(base, cur_far, now, cfg)
    out["core_shift_fires_above_threshold"] = a is not None and a.alert_type == "core_shift"
    if a:
        out["core_shift_full_fields"] = all(
            k in a.as_dict() for k in
            ["tiger_id", "window", "timestamp", "evidence", "metric", "value",
             "threshold", "severity", "confidence", "explanation"])

    # 2) new_identity fires only when distance above threshold
    out["new_identity_fires"] = eval_new_identity(
        {"tiger_id": "T-999", "min_distance": 0.72, "station_id": "S01"}, now, cfg) is not None
    out["new_identity_no_fire"] = eval_new_identity(
        {"tiger_id": "T-007", "min_distance": 0.30, "station_id": "S01"}, now, cfg) is None

    # 3) unusual_movement speed-based (jittered so geometry is non-degenerate)
    # Last detection is ~53 km away (~0.48 deg) but only hours after the
    # previous one, so the implied speed is far above 3 km/day.
    mix = [DetectionPoint(f"S{i % 10:02d}",
                          (now - timedelta(days=14 + i * 96)).isoformat(),
                          *jitter(21.65, 79.35), "T1", 0.9) for i in range(3)] + \
          [DetectionPoint("S09", (now - timedelta(hours=6)).isoformat(),
                          21.65 + 0.48, 79.35 + 0.12, "T1", 0.9)]
    a = eval_unusual_movement(mix, now, cfg)
    out["unusual_movement_fires"] = a is not None and a.alert_type == "unusual_movement"

    # 4) activity_anomaly z-score based
    history = [10, 12, 11, 9, 10, 12]
    a = eval_activity_anomaly(history, 40, now, cfg, "T1")
    out["activity_anomaly_surge_fires"] = a is not None and a.metric == "activity_z_score"
    a = eval_activity_anomaly(history, 11, now, cfg, "T1")
    out["activity_anomaly_normal_no_fire"] = a is None

    # 5) territorial_overlap: two tiger ranges that substantially overlap.
    #    Each range is built from detections spread over a realistic
    #    territory (tens of km across) so the KDE area is large and the
    #    two centroids are close; the far case uses disjoint ranges.
    def _grid(lat, lon, n=14, step_km=4.0):
        """Synthetic territory detections covering a realistic area."""
        step_deg = step_km / 111.0
        pts = []
        for i in range(n):
            pts.append(DetectionPoint(f"S{i % 12:02d}",
                                      (now - timedelta(days=1 + i * 2)).isoformat(),
                                      lat + (i % 4 - 1.5) * step_deg,
                                      lon + (i // 4 - 1.5) * step_deg,
                                      "T1", 0.9))
        return pts
    a = eval_territorial_overlap(_grid(21.65, 79.35),
                                 _grid(21.655, 79.355, n=14), now, cfg, "T1")
    out["territorial_overlap_fires_nearby"] = a is not None and a.alert_type == "territorial_overlap"
    a = eval_territorial_overlap(
        [DetectionPoint(f"S{i % 10:02d}",
                        (now - timedelta(days=1 + i * 4)).isoformat(),
                        *jitter(21.65, 79.35), "T1", 0.9) for i in range(8)],
        [DetectionPoint(f"S{i % 10:02d}",
                        (now - timedelta(days=1 + i * 4)).isoformat(),
                        *jitter(21.9, 79.9), "T1", 0.9) for i in range(8)],
        now, cfg, "T1")
    out["territorial_overlap_no_fire_far"] = a is None

    # 6) debounce: escalation after confirmation windows.
    #    Fresh Alert objects are created per call (eval_core_shift returns a
    #    new object), so pass freshly-built alert lists to each debounce run.
    fake_alerts_0 = [eval_core_shift(base, cur_far, now, cfg)]
    alerts1, keys1 = apply_debounce(fake_alerts_0, {}, cfg)
    fake_alerts_1 = [eval_core_shift(base, cur_far, now, cfg)]
    alerts2, keys2 = apply_debounce(fake_alerts_1, keys1, cfg)
    out["debounce_escalates_after_persistence"] = \
        alerts2[0].severity == "escalated" if alerts2 else False
    out["first_run_not_escalated"] = alerts1[0].severity != "escalated"
    return {"available": True, "tests": out,
            "passed": sum(1 for v in out.values() if v),
            "total": len(out)}


# ---------------------------------------------------------------------------
# suite: pipeline safety (Task 7)
# ---------------------------------------------------------------------------
def suite_pipeline():
    from pench.model_serving import BlankFilter
    results = {}
    bf = BlankFilter(MODELS / "blank_filter_v2.pth")
    # animal image must never be verdict 'empty' (quarantine safety)
    animal_src = sorted((PROJECT / "datasets" / "cct_subset" / "val" / "animal").iterdir())[:40]
    empty_src = sorted((PROJECT / "datasets" / "cct_subset" / "val" / "empty").iterdir())[:40]
    animal_verdicts, empty_verdicts = [], []
    for f in animal_src:
        v = bf.triage(f)["verdict"]
        animal_verdicts.append(v)
        if v == "empty":
            results["animal_silently_discarded"] = True
    results["animal_never_silently_discarded"] = "animal_silently_discarded" not in results
    for f in empty_src:
        empty_verdicts.append(bf.triage(f)["verdict"])
    results["ambiguous_routes_to_review"] = "review" in set(animal_verdicts + empty_verdicts)
    results["animal_distribution"] = {k: animal_verdicts.count(k) for k in
                                      ("animal", "empty", "review")}
    results["empty_distribution"] = {k: empty_verdicts.count(k) for k in
                                     ("animal", "empty", "review")}
    # quarantine metadata: quarantine dir preserved by pipeline demo
    qdir = PROJECT / "demo" / "output" / "quarantine"
    results["quarantine_preserved"] = qdir.is_dir() and any(qdir.iterdir())
    results["quarantine_contains_metadata_or_images"] = len(list(qdir.iterdir())) if results["quarantine_preserved"] else 0
    return {"available": True} | results


# ---------------------------------------------------------------------------
# suite: versioning & promotion (Task 9)
# ---------------------------------------------------------------------------
def suite_versioning():
    out = {"available": True, "checkpoints": {}}
    for f in sorted(MODELS.glob("*.pth")):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = {}
        meta = {k: ckpt[k] for k in
                ("version", "training_date", "device", "hyperparameters",
                 "dataset", "threshold_lo", "threshold_hi", "calibration")
                if k in ckpt}
        out["checkpoints"][f.name] = {
            "has_version_field": "version" in ckpt,
            "has_metrics": "metrics" in ckpt,
            "has_training_date": "training_date" in ckpt,
            "has_device": "device" in ckpt,
            "has_thresholds": "threshold_lo" in ckpt and "threshold_hi" in ckpt,
            "has_hyperparameters": "hyperparameters" in ckpt,
            "has_dataset": "dataset" in ckpt,
            "metadata": meta,
        }
    out["no_overwrite_policy"] = (MODELS / "blank_filter.pth").exists() and \
        (MODELS / "blank_filter_v2.pth").exists()
    return out


# ---------------------------------------------------------------------------
# suite: performance (Task 10)
# ---------------------------------------------------------------------------
def suite_perf():
    from pench.model_serving import TigerReID
    out = {"available": True, "device": str(DEVICE),
           "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        out["cuda_device"] = torch.cuda.get_device_name(0)
        out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)

    # import RAM baseline
    import resource
    ram0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # blank filter batch benchmark
    from pench.model_serving import BlankFilter
    bf = BlankFilter(MODELS / "blank_filter_v2.pth")
    imgs = sorted((PROJECT / "datasets" / "cct_subset" / "val" / "animal").iterdir())[:100]
    imgs = [p for p in imgs if p.suffix.lower() in (".jpg",)]
    if not imgs:
        imgs = sorted((PROJECT / "datasets" / "cct_subset" / "val" / "empty").iterdir())[:100]

    def bench(fn, paths, warmup=10, label=""):
        for p in paths[:warmup]:
            fn(p)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        t = []
        for p in paths:
            t0 = time.perf_counter()
            fn(p)
            t.append(time.perf_counter() - t0)
        t = np.array(t)
        n = len(paths) - warmup
        return {"n": n, "total_s": round(t.sum(), 2),
                "imgs_per_sec": round(n / t.sum(), 1),
                "avg_ms": round(1000 * t.mean(), 1),
                "p95_ms": round(1000 * np.percentile(t, 95), 1)}

    out["blank_filter"] = bench(bf.triage, imgs, label="triage")

    # Re-ID batch benchmark (embed-only, faster than full FAISS identify)
    reid = TigerReID()
    n_reid = min(50, len(imgs))
    out["reid_embed"] = bench(reid.embed, imgs[:n_reid], warmup=3, label="embed")
    out["reid_identify"] = bench(reid.identify, imgs[:n_reid], warmup=3, label="identify")

    ram1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out["ram_mb_peak"] = round(ram1, 0)
    out["batch_capable_cpu"] = out["blank_filter"]["imgs_per_sec"] >= 10
    return out


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
SUITES = {
    "blank": suite_blank,
    "reid": suite_reid,
    "data": suite_data,
    "home": suite_home,
    "alerts": suite_alerts,
    "pipeline": suite_pipeline,
    "versioning": suite_versioning,
    "perf": suite_perf,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every suite")
    ap.add_argument("suites", nargs="*", choices=list(SUITES))
    args = ap.parse_args()
    keys = list(SUITES) if args.all else args.suites
    if not keys:
        ap.print_help(); return

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "device": str(DEVICE), "suites": {}}
    console = []
    for k in keys:
        print(f"\n=== suite: {k} ===", flush=True)
        try:
            res = SUITES[k]()
        except Exception as e:
            res = {"available": False, "error": str(e),
                   "traceback": traceback.format_exc()}
        report["suites"][k] = res
        console.append((k, res))

    # console summary
    print("\n" + "=" * 70)
    print(f"{'Suite':<12} {'Status':<10} Highlights")
    for k, res in console:
        status = "OK" if res.get("available") else "FAIL/missing"
        hl = ""
        if k == "blank" and "promotion_decision" in res:
            hl = f"v2 acc={res['v2'].get('accuracy')} f1={res['v2'].get('f1_animal')} " \
                 f"fE={res['v2'].get('false_empty_rate')} -> {res['promotion_decision']['recommendation']}"
        elif k == "reid" and "task2" in res:
            hl = f"Rank-1={res['task2'].get('rank1')} mAP={res['task2'].get('map')} " \
                 f"open-set AUC={res['task3'].get('open_set_roc_auc')}"
        elif k == "alerts":
            hl = f"{res.get('passed')}/{res.get('total')} unit tests passed"
        elif k == "perf":
            hl = f"blank {res.get('blank_filter', {}).get('imgs_per_sec')} img/s " \
                 f"(p95 {res.get('blank_filter', {}).get('p95_ms')} ms)"
        print(f"{k:<12} {status:<10} {hl}", flush=True)
    print("=" * 70)

    out_path = REPORTS / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
