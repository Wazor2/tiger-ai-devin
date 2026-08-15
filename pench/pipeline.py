"""Pench Tiger Reserve — Camera-Trap Triage & Movement Intelligence Pipeline.

End-to-end runner that chains the four modules over an ingestion folder:

  Module 1  BlankFilter.triage           ->  animal | empty | review
  Module 2  TigerReID.identify           ->  known_identity | new_identity | human_review
  Module 3  occupancy_safe.safe_home_range   ->  validated MCP + 95% KDE, overlap
  Module 4  alerting_v2.evaluate_all     ->  five deterministic alert types + debounce

Modules 3 and 4 are the safety-hardened implementations: they validate
coordinates, refuse to report a home range from too few detections, survive
degenerate geometry, and debounce alerts across runs using persisted state in
`<output>/alert_state.json`.

Usage:
    python3 -m pench.pipeline --ingest ./ingest_demo --output ./run_output
"""
import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pench.model_serving import BlankFilter, TigerReID
from pench.modules.occupancy_safe import safe_home_range, overlap_report_safe
from pench.modules.alerting_v2 import (AlertConfigV2, DetectionPoint,
                                       evaluate_all)

HISTORY_FIELDS = ["tiger_id", "timestamp", "station_id", "lat", "lon",
                  "confidence"]

# Camera stations used when the caller does not supply its own layout.
DEFAULT_CAMERAS = {
    "S01": (21.6800, 79.2900),
    "S02": (21.6500, 79.3500),
    "S03": (21.6100, 79.4200),
    "S04": (21.7200, 79.3800),
}


def load_history_csv(csv_path: Path) -> list:
    """Load historical verified detections as lat/lon DetectionPoints.

    History is stored in lat/lon and consumed in lat/lon — no UTM round-trip,
    which is what silently corrupted coordinates in the previous version.
    """
    out = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append(DetectionPoint(
                    station_id=row["station_id"],
                    timestamp=row["timestamp"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    tiger_id=row["tiger_id"],
                    confidence=float(row.get("confidence") or 0.9)))
            except (KeyError, TypeError, ValueError):
                continue        # skip malformed rows rather than crash a run
    return out


def append_history_csv(csv_path: Path, points: list):
    """Append this run's new detections, writing a header if the file is new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(HISTORY_FIELDS)
        for p in points:
            w.writerow([p.tiger_id, p.timestamp, p.station_id,
                        round(p.lat, 5), round(p.lon, 5),
                        round(p.confidence, 3)])


def load_alert_state(path: Path) -> dict:
    """Read persisted debounce state: {alert_key: consecutive_run_count}."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    keys = state.get("previous_alert_keys", {})
    return {k: int(v) for k, v in keys.items() if isinstance(v, (int, float))}


def save_alert_state(path: Path, keys: dict, now: datetime):
    Path(path).write_text(json.dumps(
        {"updated": now.isoformat(), "previous_alert_keys": keys}, indent=1))


def window_counts(points: list, now: datetime, window_days: int,
                  n_windows: int = 6, step_days: int = 7) -> dict:
    """Per-tiger detection counts over past windows of `window_days` each.

    alerting_v2.eval_activity_anomaly compares these against the count in the
    *current* window, so the historical buckets must span the same number of
    days — otherwise the z-score compares a month against a week. Windows are
    stepped by `step_days` (so they overlap) to get the >=5 buckets the
    evaluator requires out of ~2-3 months of history.
    """
    ages = {}
    for p in points:
        try:
            ts = datetime.fromisoformat(p.timestamp.replace("Z", "+00:00")[:26])
        except ValueError:
            continue
        age = (now - ts.replace(tzinfo=None)).days
        if age >= 0:
            ages.setdefault(p.tiger_id, []).append(age)
    out = {}
    for tid, tiger_ages in ages.items():
        buckets = []
        for i in range(1, n_windows + 1):        # i=0 is the current window
            start = i * step_days
            buckets.append(sum(1 for a in tiger_ages
                               if start <= a < start + window_days))
        out[tid] = list(reversed(buckets))       # oldest first
    return out


def summarise_range(report: dict) -> dict:
    """Flatten a safe_home_range report for the run report / dashboard.

    Handles insufficient_data and degenerate_geometry results, which carry no
    area or centroid fields at all.
    """
    base = {
        "quality": report.get("quality", "unknown"),
        "n_input": report.get("n_input", 0),
        "n_used": report.get("n_used", 0),
        "n_outliers": report.get("n_outliers", 0),
        "n_rejected_coordinates": report.get("n_rejected_coordinates", 0),
    }
    if report.get("reason"):
        base["reason"] = report["reason"]
    if "centroid_lat" not in report:
        base["usable"] = False
        return base
    base.update({
        "usable": True,
        "mcp_area_km2": report.get("mcp_area_km2"),
        "core_mcp_50_area_km2": report.get("core_mcp_50_area_km2"),
        "kde95_area_km2": report.get("kde95_area_km2"),
        "centroid_lat": report.get("centroid_lat"),
        "centroid_lon": report.get("centroid_lon"),
        "spread_max_km": report.get("spread_max_km"),
        "confidence_note": report.get("uncertainty", {}).get("confidence_note"),
    })
    return base


def run_pipeline(ingest_dir: Path, output_dir: Path,
                 cameras: dict = None, now: datetime = None,
                 history_csv: Path = None, cfg: AlertConfigV2 = None):
    now = now or datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_dir = Path(ingest_dir)
    cameras = cameras or DEFAULT_CAMERAS      # station_id -> (lat, lon)
    cfg = cfg or AlertConfigV2()
    history_csv = Path(history_csv) if history_csv else output_dir / "detection_history.csv"
    state_path = output_dir / "alert_state.json"

    blank = BlankFilter()
    reid = TigerReID()

    results = {"started": now.isoformat(), "stages": {},
               "models": {"blank_filter": blank.ckpt_path.name,
                          "blank_filter_band": [blank.lo, blank.hi],
                          "reid_confirm_dist": reid.CONFIRM_DIST,
                          "reid_enroll_dist": reid.ENROLL_DIST}}
    counts = {"total": 0, "animal": 0, "empty": 0, "review": 0}
    verified = []          # detections that passed modules 1+2

    # ---------- Module 1 ----------
    stage1 = []
    for img in sorted(ingest_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        counts["total"] += 1
        verdict = blank.triage(img)
        stage1.append({"file": img.name, **verdict})
        counts[verdict["verdict"]] += 1
        if verdict["verdict"] == "animal":
            verified.append({"file": img.name,
                             "confidence": verdict["animal_confidence"]})

    results["stages"]["module1_blank_triage"] = {
        "counts": counts,
        "animal_files": [s["file"] for s in stage1 if s["verdict"] == "animal"],
        "quarantined": [s["file"] for s in stage1 if s["verdict"] == "review"],
        "archived_blanks": [s["file"] for s in stage1 if s["verdict"] == "empty"],
    }

    # quarantine: copy review images to a safe quarantine folder (deletion-safe)
    qdir = output_dir / "quarantine"
    qdir.mkdir(exist_ok=True)
    for s in stage1:
        if s["verdict"] == "review":
            shutil.copy(ingest_dir / s["file"], qdir / s["file"])

    # ---------- Module 2 ----------
    stage2 = []
    new_points = []
    identity_records = []
    enrolled_new = 0
    for v in verified:
        rec = reid.identify(ingest_dir / v["file"])
        # assign station from filename pattern STATIONID_xxx.jpg
        m = re.match(r"^([A-Za-z]+\d*)", v["file"])
        station_id = m.group(1) if m else "S00"
        entry = {"file": v["file"], "station_id": station_id, **rec}
        stage2.append(entry)
        cam = cameras.get(station_id)
        if rec["decision"] == "known_identity":
            if cam is None:
                entry["note"] = f"unknown station {station_id}; not localised"
                continue
            new_points.append(DetectionPoint(
                station_id=station_id, timestamp=now.isoformat(),
                lat=cam[0], lon=cam[1], tiger_id=rec["top_match"],
                confidence=v["confidence"]))
        elif rec["decision"] == "new_identity":
            enrolled_new += 1
            entry["note"] = "enrolled as new identity (auto-enroll)"
            identity_records.append({
                "tiger_id": rec.get("top_match") or f"NEW-{v['file']}",
                "station_id": station_id,
                "min_distance": rec.get("cosine_distance", 1.0)})

    results["stages"]["module2_tiger_reid"] = {
        "processed": len(stage2),
        "known_identity": sum(1 for s in stage2 if s["decision"] == "known_identity"),
        "new_identity": sum(1 for s in stage2 if s["decision"] == "new_identity"),
        "human_review": sum(1 for s in stage2 if s["decision"] == "human_review"),
        "new_identities_enrolled": enrolled_new,
        "detail": stage2,
    }

    # ---------- Module 3 ----------
    history = load_history_csv(history_csv)
    if new_points:
        append_history_csv(history_csv, new_points)
    points = history + new_points

    by_tiger = {}
    for p in points:
        by_tiger.setdefault(p.tiger_id, []).append(p)

    raw_ranges, ranges_by_tiger = {}, {}
    for tid, pts in sorted(by_tiger.items()):
        report = safe_home_range(pts)["home_range"]
        raw_ranges[tid] = report
        ranges_by_tiger[tid] = summarise_range(report)

    population = safe_home_range(points)["home_range"] if points else {}
    results["stages"]["module3_home_range"] = {
        "per_tiger": ranges_by_tiger,
        "territorial_overlap": overlap_report_safe(raw_ranges),
        "population": summarise_range(population) if points else {"usable": False},
    }

    # ---------- Module 4 ----------
    previous_keys = load_alert_state(state_path)
    alerts, updated_keys = evaluate_all(
        tiger_detections=by_tiger,
        other_tigers=list(by_tiger),
        identity_records=identity_records,
        weekly_counts=window_counts(points, now, cfg.current_window_days),
        now=now, cfg=cfg, previous_alert_keys=previous_keys)
    save_alert_state(state_path, updated_keys, now)

    results["stages"]["module4_alerts"] = [a.as_dict() for a in alerts]
    results["stages"]["module4_debounce"] = {
        "confirmation_windows": cfg.confirmation_windows,
        "previous_alert_keys": previous_keys,
        "updated_alert_keys": updated_keys,
        "state_file": str(state_path),
    }

    out = output_dir / "pipeline_report.json"
    json.dump(results, open(out, "w"), indent=1, default=str)
    print(json.dumps(results, indent=1, default=str)[:3000])
    print(f"\nReport saved to {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", required=True)
    ap.add_argument("--output", default="./run_output")
    ap.add_argument("--history", default=None,
                    help="detection history CSV (default: <output>/detection_history.csv)")
    args = ap.parse_args()
    run_pipeline(Path(args.ingest), Path(args.output),
                 history_csv=Path(args.history) if args.history else None)
