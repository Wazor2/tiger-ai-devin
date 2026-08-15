"""Pench Tiger Reserve — Camera-Trap Triage & Movement Intelligence Pipeline.

End-to-end runner that chains the four modules over an ingestion folder:

  Module 1  blank_filter.TriageImage   ->  animal | empty | review
  Module 2  detector + reid.Identify   ->  known_identity | new_identity | human_review
  Module 3  occupancy.HomeRange        ->  MCP + 95% AKDE, overlap mapping
  Module 4  alerting.DeviationCheck    ->  alerts after artefact filtering

Usage:
    python3 -m pench.pipeline --ingest ./ingest_demo --output ./run_output
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pench.model_serving import BlankFilter, TigerReID
from pench.modules.occupancy import (Detection, compute_home_range,
                                     lonlat_to_utm, utm_to_lonlat, overlap_report)
from pench.modules.alerting import StationStatus, check_deviation, AlertConfig


def load_history_csv(csv_path: Path) -> list:
    """Load historical verified detections: tiger_id,timestamp,station_id,lat,lon,confidence."""
    import csv
    out = []
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            e, n = lonlat_to_utm(float(row["lon"]), float(row["lat"]))
            out.append(Detection(row["station_id"], row["timestamp"], e, n,
                                 tiger_id=row["tiger_id"],
                                 confidence=float(row["confidence"])))
    return out


def run_pipeline(ingest_dir: Path, output_dir: Path,
                 cameras: dict = None, now: datetime = None,
                 history_csv: Path = None):
    now = now or datetime.now()
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_dir = Path(ingest_dir)
    cameras = cameras or {}  # station_id -> (lat, lon)

    blank = BlankFilter()
    reid = TigerReID()

    results = {"started": now.isoformat(), "stages": {}}
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
    detections = []
    enrolled_new = 0
    for v in verified:
        rec = reid.identify(ingest_dir / v["file"])
        # assign station from filename pattern STATIONID_xxx.jpg
        import re
        m = re.match(r"^([A-Za-z]+\d*)", v["file"])
        station_id = m.group(1) if m else "S00"
        entry = {"file": v["file"], **rec}
        stage2.append(entry)
        if rec["decision"] == "known_identity":
            cam = cameras.get(station_id)
            if cam is None:
                continue
            e, n = lonlat_to_utm(cam[1], cam[0])  # (lon, lat)
            detections.append(Detection(station_id, now.isoformat(), e, n,
                                        tiger_id=rec["top_match"],
                                        confidence=v["confidence"]))
        elif rec["decision"] == "new_identity":
            enrolled_new += 1
            entry["note"] = "enrolled as new identity (auto-enroll)"

    results["stages"]["module2_tiger_reid"] = {
        "processed": len(stage2),
        "known_identity": sum(1 for s in stage2 if s["decision"] == "known_identity"),
        "new_identity": sum(1 for s in stage2 if s["decision"] == "new_identity"),
        "human_review": sum(1 for s in stage2 if s["decision"] == "human_review"),
        "detail": stage2,
    }

    # ---------- Module 3 ----------
    stations = [StationStatus(sid, now.isoformat()) for sid in cameras]
    # merge historical detections with today's new detections
    if history_csv:
        history = load_history_csv(history_csv)
        detections = history + detections
        # append today's detections to the history for the next run
        import csv as _csv
        with open(history_csv, "a") as f:
            w = _csv.writer(f)
            for d in [x for x in detections if x.tiger_id in
                      {d2.tiger_id for d2 in detections[-len(history)-10:]} and x.timestamp == now.isoformat()]:
                lon, lat = utm_to_lonlat(d.easting, d.northing)
                w.writerow([d.tiger_id, d.timestamp, d.station_id,
                            round(lat, 5), round(lon, 5), round(d.confidence, 3)])
    ranges_by_tiger = {}
    tigers = sorted({d.tiger_id for d in detections})
    for tid in tigers:
        tdets = [d for d in detections if d.tiger_id == tid]
        r = compute_home_range(tid, tdets)
        ranges_by_tiger[tid] = {
            "n_detections": r.n_detections,
            "mcp_area_km2": round(r.mcp_area_km2, 1),
            "core_mcp_50_area_km2": round(r.core_mcp_50_area_km2, 1),
            "kde95_area_km2": round(r.kde95_area_km2, 1),
            "centroid_lat": round(r.centroid_lat, 5),
            "centroid_lon": round(r.centroid_lon, 5),
            "spread_max_km": round(r.spread_max_km, 1),
        }
    full_range = compute_home_range("population", detections) if detections else None
    results["stages"]["module3_home_range"] = {
        "per_tiger": ranges_by_tiger,
        "territorial_overlap": overlap_report([]),
        "population_mcp_km2": round(full_range.mcp_area_km2, 1) if full_range else 0,
    }

    # ---------- Module 4 ----------
    alerts = []
    for tid in tigers:
        tdets = [d for d in detections if d.tiger_id == tid]
        alerts.extend(check_deviation(tid, tdets, stations, now))
    results["stages"]["module4_alerts"] = [
        {"tiger_id": a.tiger_id, "type": a.alert_type, "severity": a.severity,
         "detail": a.detail} for a in alerts
    ]

    results["stages"]["module2_tiger_reid"]["new_identities_enrolled"] = enrolled_new

    out = output_dir / "pipeline_report.json"
    json.dump(results, open(out, "w"), indent=1, default=str)
    print(json.dumps(results, indent=1, default=str)[:3000])
    print(f"\nReport saved to {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", required=True)
    ap.add_argument("--output", default="./run_output")
    args = ap.parse_args()
    run_pipeline(Path(args.ingest), Path(args.output))
