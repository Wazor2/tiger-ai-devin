"""FastAPI backend for the live demo: capture + inference + API in one process.

    python -m pench.server --source dummy --inference dummy   # plumbing only
    python -m pench.server --source folder --folder demo/ingest
    python -m pench.server --source webcam --device 1         # phone via Iriun

Dashboard on http://127.0.0.1:8000/ ; the capture loop runs in a background
thread and pushes results into a LiveStore that the endpoints read.
"""
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from pench.live_capture import CaptureLoop, build_source, probe_devices
from pench.live_infer import build_inference
from pench.live_map import render_map
from pench.live_store import DEMO_LATLON, DEMO_STATION, LiveStore

PROJECT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT / "static"
MAP_INTERVAL = 30.0        # seconds between home-range map regenerations

app = FastAPI(title="Pench Tiger — live monitoring console")
runtime = {"store": None, "capture": None, "inference": None,
           "started": None, "source": None, "map_thread": None}


# ------------------------------------------------------------------ capture
def handle_capture(frame_bgr, meta: dict):
    """Called by the capture thread for every triggered frame."""
    import cv2
    from PIL import Image

    t0 = time.perf_counter()
    store: LiveStore = runtime["store"]
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    frame_path = store.frames_dir / f"{stamp}.jpg"
    cv2.imwrite(str(frame_path), frame_bgr)
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    result = runtime["inference"].classify(image)
    reid = result.get("reid")

    point = identity_record = None
    if reid and reid["decision"] == "known_identity" and reid.get("tiger_id"):
        point = store.make_point(reid["tiger_id"],
                                 result.get("animal_confidence", 0.9), now)
    elif reid and reid["decision"] == "new_identity":
        identity_record = {"tiger_id": reid.get("tiger_id") or f"NEW-{stamp}",
                           "station_id": DEMO_STATION,
                           "min_distance": reid.get("cosine_distance", 1.0)}

    event = {
        "timestamp": now.isoformat(timespec="seconds"),
        "verdict": result["verdict"],
        "animal_confidence": result.get("animal_confidence"),
        "quarantine_band": result.get("quarantine_band"),
        "model_version": result.get("model_version"),
        "reid": reid,
        "frame_url": f"/media/frames/{frame_path.name}",
        "station_id": DEMO_STATION,
        "lat": DEMO_LATLON[0], "lon": DEMO_LATLON[1],
        "trigger": meta.get("trigger"),
        "motion_score": meta.get("motion_score"),
        "source_file": meta.get("source_file"),
        "inference_ms": result.get("inference_ms"),
        "recorded": point is not None,
    }
    event["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    store.add_event(event, point=point, identity_record=identity_record, now=now)


def map_refresher(stop: threading.Event, interval: float = MAP_INTERVAL):
    store: LiveStore = runtime["store"]
    cameras = {DEMO_STATION: DEMO_LATLON}
    while not stop.is_set():
        try:
            render_map(store.raw_ranges(), cameras, store.map_path)
            store.map_generation += 1
        except Exception as exc:
            runtime["map_error"] = f"{type(exc).__name__}: {exc}"
        stop.wait(interval)


def start_runtime(source_kind="dummy", inference_kind="dummy", device=0,
                  folder=None, hold=6.0, output=PROJECT / "run_output",
                  reseed=False):
    store = LiveStore(output)
    store.seed_baseline(force=reseed)
    runtime["store"] = store
    runtime["inference"] = build_inference(inference_kind)
    runtime["source"] = {"kind": source_kind, "device": device,
                         "folder": str(folder) if folder else None,
                         "inference": inference_kind}
    src = build_source(source_kind, device_index=device, folder=folder, hold=hold)
    loop = CaptureLoop(src, handle_capture)
    loop.start()
    runtime["capture"] = loop
    stop = threading.Event()
    thread = threading.Thread(target=map_refresher, args=(stop,),
                              name="map-refresher", daemon=True)
    thread.start()
    runtime["map_thread"] = (thread, stop)
    runtime["started"] = datetime.now().isoformat(timespec="seconds")
    return store


@app.on_event("shutdown")
def _shutdown():
    if runtime.get("capture"):
        runtime["capture"].stop()
    if runtime.get("map_thread"):
        runtime["map_thread"][1].set()


# ---------------------------------------------------------------- endpoints
def _store() -> LiveStore:
    store = runtime.get("store")
    if store is None:
        raise HTTPException(503, "capture runtime not started")
    return store


@app.get("/api/latest")
def api_latest():
    store = _store()
    return {"latest": store.latest(),
            "capture": runtime["capture"].stats if runtime["capture"] else {},
            "source": runtime["source"],
            "map_url": f"/media/home_range_map.png?g={store.map_generation}"}


@app.get("/api/alerts")
def api_alerts():
    return {"alerts": _store().alert_feed()}


@app.get("/api/home_range/{tiger_id}")
def api_home_range(tiger_id: str):
    report = _store().home_range(tiger_id)
    if not report:
        raise HTTPException(404, f"no detections for tiger {tiger_id}")
    return report


@app.get("/api/home_ranges")
def api_home_ranges():
    store = _store()
    return {"ranges": store.all_ranges(), "overlap": store.overlap}


@app.get("/api/history")
def api_history(limit: int = 50):
    store = _store()
    return {"detections": store.history_rows(limit),
            "events": store.recent_events(25)}


@app.get("/api/status")
def api_status():
    store = _store()
    return {"started": runtime["started"], "source": runtime["source"],
            "capture": runtime["capture"].stats if runtime["capture"] else {},
            "blank_model": getattr(runtime["inference"], "blank_model", None),
            "quarantine_band": getattr(runtime["inference"], "band", None),
            "tracked_tigers": sorted(store.all_ranges()),
            "map_generation": store.map_generation,
            "map_error": runtime.get("map_error")}


@app.get("/api/devices")
def api_devices():
    return {"devices": probe_devices()}


@app.get("/")
def index():
    if (STATIC_DIR / "index.html").exists():
        return RedirectResponse("/static/index.html")
    return {"service": "pench live console", "dashboard": "not built yet",
            "endpoints": ["/api/latest", "/api/alerts", "/api/history",
                          "/api/home_range/{tiger_id}", "/api/status"]}


def mount_static(output_dir):
    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/media", StaticFiles(directory=str(output_dir)), name="media")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dummy",
                    choices=["dummy", "folder", "webcam"])
    ap.add_argument("--inference", default="models", choices=["dummy", "models"])
    ap.add_argument("--device", type=int, default=0, help="webcam index")
    ap.add_argument("--folder", default=None, help="stills folder for --source folder")
    ap.add_argument("--hold", type=float, default=6.0,
                    help="seconds each still is held in folder mode")
    ap.add_argument("--output", default=str(PROJECT / "run_output"))
    ap.add_argument("--reseed", action="store_true",
                    help="rewrite the synthetic baseline history at boot")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mount_static(output)
    start_runtime(args.source, args.inference, args.device,
                  args.folder, args.hold, output, args.reseed)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
