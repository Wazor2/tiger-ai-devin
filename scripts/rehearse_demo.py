"""Timed rehearsal of the three demo beats against the real HTTP API.

Replays demo/rehearsal (known tiger -> empty frame -> unknown tiger) through the
running server, polls /api/latest the way the dashboard does, and reports how
long each beat took to become visible. Run the server first:

    python -m pench.server --source folder --folder demo/rehearsal \
        --inference models --hold 6 --port 8020
    python scripts/rehearse_demo.py --port 8020

Fails loudly if a beat classifies differently than the demo script promises, so
a bad rehearsal cannot be mistaken for a good one.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

POLL = 0.5
BEATS = [
    ("01_known_tiger_172.jpg", "animal", "known_identity", "172"),
    ("02_empty_jungle.jpg", "empty", None, None),
    ("03_unknown_tiger.jpg", "animal", "human_review", None),
]


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.load(resp)


def wait_for_beat(base: str, source_file: str, deadline: float) -> tuple:
    """Poll until /api/latest reports a capture of `source_file`."""
    while time.time() < deadline:
        latest = get(base, "/api/latest")["latest"]
        if latest.get("source_file") == source_file:
            return latest, time.time()
        time.sleep(POLL)
    return None, time.time()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="seconds to wait for all three beats")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    status = get(base, "/api/status")
    print(f"model      : {status['blank_model']} band {status['quarantine_band']}")
    print(f"source     : {status['source']['kind']} {status['source']['folder']}")
    print(f"tracked    : {', '.join(status['tracked_tigers'])}\n")

    start = time.time()
    deadline = start + args.timeout
    failures, rows = [], []
    for source_file, want_verdict, want_decision, want_tiger in BEATS:
        t_wait = time.time()
        latest, seen_at = wait_for_beat(base, source_file, deadline)
        if latest is None:
            failures.append(f"{source_file}: never appeared on /api/latest")
            print(f"MISSING    {source_file}")
            continue
        reid = latest.get("reid") or {}
        got = (latest["verdict"], reid.get("decision"), reid.get("tiger_id"))
        ok = (got[0] == want_verdict and got[1] == want_decision
              and (want_tiger is None or got[2] == want_tiger))
        if not ok:
            failures.append(f"{source_file}: expected "
                            f"{(want_verdict, want_decision, want_tiger)}, got {got}")
        rows.append({
            "source_file": source_file, "verdict": latest["verdict"],
            "animal_confidence": latest.get("animal_confidence"),
            "reid_decision": reid.get("decision"), "tiger_id": reid.get("tiger_id"),
            "cosine_distance": reid.get("cosine_distance"),
            "trigger": latest.get("trigger"),
            "inference_ms": latest.get("inference_ms"),
            "pipeline_ms": latest.get("latency_ms"),
            "api_visible_s": round(seen_at - t_wait, 2),
            "recorded": latest.get("recorded"),
            "pass": ok,
        })
        print(f"{'PASS' if ok else 'FAIL'}       {source_file}: "
              f"{latest['verdict']} conf={latest.get('animal_confidence')} "
              f"reid={reid.get('decision')}/{reid.get('tiger_id')} "
              f"dist={reid.get('cosine_distance')} "
              f"pipeline={latest.get('latency_ms')}ms "
              f"visible_in={round(seen_at - t_wait, 2)}s")

    alerts = get(base, "/api/alerts")["alerts"]
    print("\nalerts     : " + (", ".join(
        f"{a['tiger_id']}/{a['alert_type']}/{a['severity']}" for a in alerts)
        or "none"))
    total = round(time.time() - start, 1)
    print(f"total      : {total}s for {len(rows)}/3 beats")

    out = Path("run_output/rehearsal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"beats": rows, "alerts": alerts,
                               "total_s": total, "failures": failures}, indent=2))
    for line in failures:
        print("FAILURE:", line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
