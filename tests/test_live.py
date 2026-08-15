"""Regression tests for the live demo layer (capture loop, store, API).

These cover the failure modes that would show up on stage: a capture loop that
dies on a bad frame or a callback error, a synthetic baseline that invents
alerts, and endpoints that 500 before the first frame arrives.
"""
import time
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pench import live_store, server
from pench.live_capture import CaptureLoop, DummySource, build_source, motion_score
from pench.live_infer import DummyInference
from pench.live_map import render_map
from pench.live_store import LiveStore, nearest_station


# --------------------------------------------------------------- capture loop
class ScriptedSource(DummySource):
    """Emits a fixed list of frames, then None forever."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.exhausted = False

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        self.exhausted = True
        return None


def _frame(value: int) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


def _run(loop: CaptureLoop, timeout: float = 3.0):
    loop.start()
    loop.join(timeout)
    loop.stop()


def test_motion_score_zero_for_identical_frames():
    assert motion_score(_frame(30), _frame(30)) == 0.0
    assert motion_score(_frame(0), _frame(200)) > 100


def test_first_frame_is_captured_immediately():
    seen = []
    src = ScriptedSource([_frame(30)] * 3)
    _run(CaptureLoop(src, lambda f, m: seen.append(m), min_gap=5.0, poll=0.01))
    # the demo must show something at once, so the first frame always fires
    assert [m["trigger"] for m in seen] == ["interval"]


def test_scene_change_triggers_capture_and_min_gap_suppresses_duplicates():
    seen = []
    src = ScriptedSource([_frame(0), _frame(255), _frame(0)])
    # force_interval disabled: only motion can trigger here
    _run(CaptureLoop(src, lambda f, m: seen.append(m), min_gap=1.0,
                     force_interval=1e12, poll=0.01))
    assert [m["trigger"] for m in seen] == ["motion"]
    assert seen[0]["motion_score"] > 6.0


def test_static_scene_still_captures_on_interval():
    seen = []
    src = ScriptedSource([_frame(30)] * 6)
    _run(CaptureLoop(src, lambda f, m: seen.append(m), min_gap=0.0,
                     force_interval=0.0, poll=0.01))
    assert seen and all(m["trigger"] == "interval" for m in seen)


def test_callback_error_does_not_kill_the_loop():
    calls = {"n": 0}

    def boom(frame, meta):
        calls["n"] += 1
        raise RuntimeError("inference exploded")

    src = ScriptedSource([_frame(30)] * 4)
    loop = CaptureLoop(src, boom, min_gap=0.0, force_interval=0.0, poll=0.01)
    _run(loop)
    assert calls["n"] >= 2
    assert loop.stats["errors"] == calls["n"]
    assert "RuntimeError" in loop.stats["last_error"]


def test_dummy_source_frames_are_valid_bgr():
    frame = build_source("dummy").read()
    assert frame.shape[2] == 3 and frame.dtype == np.uint8


def test_folder_source_reports_current_file(tmp_path):
    import cv2
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(tmp_path / name), _frame(60))
    src = build_source("folder", folder=tmp_path, hold=0.05)
    assert src.current_name() in ("a.jpg", "b.jpg")
    time.sleep(0.06)
    assert src.read() is not None


def test_build_source_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_source("telepathy")


# ---------------------------------------------------------------- live store
@pytest.fixture
def store(tmp_path):
    s = LiveStore(tmp_path / "out")
    s.seed_baseline(now=datetime(2026, 8, 15, 9, 0))
    return s


def test_nearest_station_picks_the_closest_camera():
    assert nearest_station(*live_store.DEMO_LATLON) == "S01"
    assert nearest_station(21.6110, 79.3300) == "S03"


def test_seed_baseline_is_deterministic_and_dense_enough(store, tmp_path):
    rows = store.history_rows(limit=1000)
    assert len(rows) >= 30
    per_tiger = {t: 0 for t in live_store.BASELINE_TIGERS}
    for row in rows:
        per_tiger[row["tiger_id"]] += 1
    assert all(n >= 30 for n in per_tiger.values())

    twin = LiveStore(tmp_path / "twin")
    twin.seed_baseline(now=datetime(2026, 8, 15, 9, 0))
    assert twin.history_rows(1000) == rows


def test_seed_baseline_is_idempotent(store):
    before = len(store.history_rows(1000))
    store.seed_baseline(now=datetime(2026, 8, 15, 9, 0))
    assert len(store.history_rows(1000)) == before


def test_seeded_history_raises_only_the_intended_core_shift(store):
    alerts = store.alert_feed()
    assert [(a["tiger_id"], a["alert_type"]) for a in alerts] == [("252", "core_shift")]


def test_seeded_detections_never_exceed_the_movement_threshold(store):
    by_tiger = {}
    for row in store.history_rows(1000):
        by_tiger.setdefault(row["tiger_id"], []).append(row)
    for rows in by_tiger.values():
        rows.sort(key=lambda r: r["timestamp"])
        for a, b in pairwise(rows):
            dt = (datetime.fromisoformat(b["timestamp"])
                  - datetime.fromisoformat(a["timestamp"])).total_seconds() / 86400
            km = 111 * ((a["lat"] - b["lat"]) ** 2
                        + (0.93 * (a["lon"] - b["lon"])) ** 2) ** 0.5
            assert km / max(dt, 1e-6) <= live_store.LIVE_MOVEMENT_SPEED_KM_PER_DAY


def test_home_ranges_are_usable_for_every_seeded_tiger(store):
    ranges = store.all_ranges()
    assert set(ranges) == set(live_store.BASELINE_TIGERS)
    for report in ranges.values():
        assert report["usable"] and report["mcp_area_km2"] > 0


def test_confirmed_detection_extends_history_and_recomputes(store):
    now = datetime(2026, 8, 15, 10, 0)
    before = len(store.history_rows(1000))
    point = store.make_point("172", 0.97, now)
    store.add_event({"timestamp": now.isoformat(), "verdict": "animal"},
                    point=point, now=now)
    assert len(store.history_rows(1000)) == before + 1
    assert store.latest()["seq"] == 1
    assert store.home_range("172")["last_seen"] == now.isoformat()


def test_repeat_sightings_within_a_visit_are_not_persisted_twice(store):
    now = datetime(2026, 8, 15, 10, 0)
    before = len(store.history_rows(1000))
    for i in range(5):
        t = now + timedelta(seconds=20 * i)
        store.add_event({"timestamp": t.isoformat(), "verdict": "animal"},
                        point=store.make_point("172", 0.97, t), now=t)
    assert len(store.history_rows(1000)) == before + 1
    assert store.latest()["dedup"] == "same_visit"
    assert store.latest()["recorded"] is False

    later = now + timedelta(seconds=live_store.VISIT_DEDUP_SECONDS + 1)
    store.add_event({"timestamp": later.isoformat(), "verdict": "animal"},
                    point=store.make_point("172", 0.97, later), now=later)
    assert len(store.history_rows(1000)) == before + 2


def test_visit_dedup_keeps_the_demo_alert_story_intact(store):
    now = datetime(2026, 8, 15, 10, 0)
    for i in range(30):
        t = now + timedelta(seconds=10 * i)
        store.add_event({"timestamp": t.isoformat(), "verdict": "animal"},
                        point=store.make_point("172", 0.97, t), now=t)
    assert [(a["tiger_id"], a["alert_type"]) for a in store.alert_feed()] \
        == [("252", "core_shift")]


def test_unconfirmed_event_is_shown_but_not_recorded(store):
    before = len(store.history_rows(1000))
    store.add_event({"timestamp": datetime.now().isoformat(), "verdict": "review"})
    assert len(store.history_rows(1000)) == before
    assert store.latest()["verdict"] == "review"


def test_restart_rebuilds_state_and_escalates_via_debounce(store, tmp_path):
    assert store.state_path.exists()
    reopened = LiveStore(tmp_path / "out")
    kept = reopened.seed_baseline(now=datetime(2026, 8, 15, 9, 1))
    # existing history is reused, not re-seeded, but ranges and alerts are
    # recomputed so the dashboard is populated before the first live frame
    assert kept == len(store.history_rows(1000))
    assert reopened.all_ranges()
    severities = {a["alert_type"]: a["severity"] for a in reopened.alert_feed()}
    assert severities["core_shift"] == "escalated"


# ----------------------------------------------------------------- inference
def test_dummy_inference_cycles_through_every_demo_state():
    inf = DummyInference()
    seen = [inf.classify(None) for _ in range(4)]
    assert {r["verdict"] for r in seen} == {"animal", "empty", "review"}
    decisions = {r["reid"]["decision"] for r in seen if r.get("reid")}
    assert "known_identity" in decisions and "human_review" in decisions


# ----------------------------------------------------------------------- map
def test_render_map_skips_unusable_ranges(tmp_path, store):
    out = tmp_path / "map.png"
    ranges = dict(store.raw_ranges())
    ranges["BROKEN"] = {"usable": False, "quality": "insufficient_data"}
    render_map(ranges, {"S01": live_store.DEMO_LATLON}, out)
    assert out.exists() and out.stat().st_size > 0


# ----------------------------------------------------------------------- api
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "runtime",
                        {"store": None, "capture": None, "inference": None,
                         "started": None, "source": None, "map_thread": None})
    with TestClient(server.app) as c:
        yield c


def test_endpoints_report_503_before_the_runtime_starts(client):
    for path in ("/api/latest", "/api/alerts", "/api/history", "/api/status",
                 "/api/home_range/172"):
        assert client.get(path).status_code == 503


def test_endpoints_serve_the_seeded_state(tmp_path, client):
    server.runtime.update({
        "store": LiveStore(tmp_path / "api"), "inference": DummyInference(),
        "capture": None, "started": "now",
        "source": {"kind": "dummy", "inference": "dummy"}})
    server.runtime["store"].seed_baseline(now=datetime(2026, 8, 15, 9, 0))

    assert client.get("/api/latest").json()["latest"] == {}
    assert client.get("/api/alerts").json()["alerts"][0]["alert_type"] == "core_shift"
    assert client.get("/api/home_range/172").json()["usable"] is True
    assert client.get("/api/home_range/T-999").status_code == 404
    history = client.get("/api/history?limit=5").json()
    assert len(history["detections"]) == 5 and history["events"] == []
    assert client.get("/api/status").json()["tracked_tigers"] == ["172", "252", "85"]
