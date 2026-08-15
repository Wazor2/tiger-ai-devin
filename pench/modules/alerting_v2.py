"""Module 4 v2 — Deterministic alerting with debounce/persistence (Task 6).

Extends the existing alerting module with:
- Five deterministic alert types: core_shift, territorial_overlap,
  new_identity, unusual_movement, activity_anomaly
- Debounce / persistence: an alert must be sustained across
  `confirmation_windows` consecutive runs (or exceed a hard severity
  threshold) before it is escalated to high severity
- Every alert carries the full required field set:
  tiger_id, timestamp/window, evidence, metric, value, threshold,
  severity, confidence, explanation

Design: all alert evaluation is pure/deterministic (no randomness, no
external state), so it can be unit-tested exactly.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from .occupancy import lonlat_to_utm, utm_to_lonlat
from .occupancy_safe import safe_home_range
from .alerting import AlertConfig, StationStatus


@dataclass
class DetectionPoint:
    """Minimal detection record used by the v2 alerting module."""
    station_id: str
    timestamp: str
    lat: float
    lon: float
    tiger_id: str
    confidence: float = 0.9


@dataclass
class Alert:
    """Full-field alert record (Task 6 requirement)."""
    tiger_id: str
    alert_type: str          # core_shift | territorial_overlap | new_identity |
                             # unusual_movement | activity_anomaly
    window: str              # e.g. "2026-08-01..2026-08-14"
    timestamp: str           # run/evaluation time
    evidence: str            # human-readable evidence summary
    metric: str              # the measured quantity, e.g. "centroid_shift_km"
    value: float             # measured value
    threshold: float         # triggering threshold
    severity: str            # info | warning | critical | escalated
    confidence: float        # alert confidence (0..1), derived from evidence strength
    explanation: str         # recommended action / interpretation

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class AlertConfigV2(AlertConfig):
    confirmation_windows: int = 2     # debounce: windows needed to escalate
    overlap_area_km2: float = 50.0    # territorial overlap alert threshold
    new_identity_min_distance: float = 0.316   # mirrors TigerReID.CONFIRM_DIST
    movement_speed_km_per_day: float = 3.0    # unusual movement
    activity_z_threshold: float = 2.0         # activity anomaly (stddev units)


def _parse_ts(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts[:26])


def _window_label(now, days):
    return f"{(now - timedelta(days=days)).date()}..{now.date()}"


def _dist_km(p1, p2):
    e1, n1 = lonlat_to_utm(float(p1[1]), float(p1[0]))
    e2, n2 = lonlat_to_utm(float(p2[1]), float(p2[0]))
    return ((e2 - e1) ** 2 + (n2 - n1) ** 2) ** 0.5 / 1000


# ---------------------------------------------------------------------------
# deterministic alert evaluators
# ---------------------------------------------------------------------------
def eval_core_shift(points_baseline, points_current, now, cfg):
    """core_shift alert from two point lists."""
    if len(points_baseline) < cfg.min_current_detections or \
       len(points_current) < cfg.min_current_detections:
        return None
    hb = safe_home_range(points_baseline, min_detections=cfg.min_current_detections)
    hc = safe_home_range(points_current, min_detections=cfg.min_current_detections)
    if hb["home_range"].get("quality") == "insufficient_data" or \
       hc["home_range"].get("quality") == "insufficient_data" or \
       "centroid_lat" not in hb["home_range"] or \
       "centroid_lat" not in hc["home_range"]:
        return None
    b = (hb["home_range"]["centroid_lat"], hb["home_range"]["centroid_lon"])
    c = (hc["home_range"]["centroid_lat"], hc["home_range"]["centroid_lon"])
    d = _dist_km(b, c)
    if d <= cfg.buffer_km:
        return None
    sev = "warning" if d < 10 else "critical"
    return Alert(
        tiger_id=points_current[0].tiger_id,
        alert_type="core_shift",
        window=_window_label(now, cfg.current_window_days),
        timestamp=now.isoformat(),
        evidence=(f"baseline centroid ({b[0]:.4f},{b[1]:.4f}) vs current "
                  f"({c[0]:.4f},{c[1]:.4f}); {len(points_baseline)} baseline + "
                  f"{len(points_current)} current verified detections"),
        metric="centroid_shift_km", value=round(d, 2),
        threshold=cfg.buffer_km, severity=sev,
        confidence=min(1.0, d / (2 * cfg.buffer_km)),
        explanation="Verify with field patrol; check for conflict or "
                    "displacement before escalating to critical.",
    )


def eval_territorial_overlap(points_a, points_b, now, cfg, tiger_id):
    """territorial_overlap between two tigers' current detections."""
    if len(points_a) < cfg.min_current_detections or \
       len(points_b) < cfg.min_current_detections:
        return None
    ha = safe_home_range(points_a, min_detections=cfg.min_current_detections)
    hb_ = safe_home_range(points_b, min_detections=cfg.min_current_detections)
    if ha["home_range"].get("quality") == "insufficient_data" or \
       hb_["home_range"].get("quality") == "insufficient_data" or \
       "centroid_lat" not in ha["home_range"] or \
       "centroid_lat" not in hb_["home_range"]:
        return None
    area = hb_["home_range"].get("kde95_area_km2", 0) or 0
    # overlap approximated via centroid proximity when polygons unavailable
    ca = (ha["home_range"]["centroid_lat"], ha["home_range"]["centroid_lon"])
    cb = (hb_["home_range"]["centroid_lat"], hb_["home_range"]["centroid_lon"])
    d = _dist_km(ca, cb)
    overlap_est = max(0.0, area - 3.14 * (d / 2) ** 2) if d < 2 * (area / 3.14) ** 0.5 else 0.0
    if overlap_est < cfg.overlap_area_km2:
        return None
    sev = "warning" if overlap_est < 100 else "critical"
    return Alert(
        tiger_id=tiger_id, alert_type="territorial_overlap",
        window=_window_label(now, cfg.current_window_days),
        timestamp=now.isoformat(),
        evidence=f"estimated overlap with other tiger ≈{overlap_est:.0f} km2 "
                 f"(centroids {d:.1f} km apart; tiger A range {ha['home_range'].get('kde95_area_km2'):.0f} km2)",
        metric="overlap_area_km2", value=round(overlap_est, 1),
        threshold=cfg.overlap_area_km2, severity=sev,
        confidence=min(1.0, overlap_est / 200),
        explanation="High likelihood of territorial conflict or a female "
                    "with cub; monitor both individuals.",
    )


def eval_new_identity(identity_record, now, cfg):
    """new_identity alert when a tiger is enrolled as a new catalogue ID."""
    d = identity_record.get("min_distance", 1.0)
    if d >= cfg.new_identity_min_distance:
        return Alert(
            tiger_id=identity_record["tiger_id"], alert_type="new_identity",
            window=_window_label(now, 1), timestamp=now.isoformat(),
            evidence=(f"best match distance {d:.3f} > {cfg.new_identity_min_distance}; "
                      f"image at {identity_record.get('station_id')}"),
            metric="min_cosine_distance", value=round(d, 4),
            threshold=cfg.new_identity_min_distance, severity="info",
            confidence=round(min(1.0, max(0.0, (d - cfg.new_identity_min_distance) + 0.5)), 2),
            explanation="New individual enrolled into catalogue; capture "
                        "additional flank photos to confirm identity.",
        )
    return None


def eval_unusual_movement(points, now, cfg):
    """unusual_movement: detection-to-detection speed exceeds threshold."""
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: _parse_ts(p.timestamp))
    best = None
    for i in range(1, len(pts)):
        dt = (_parse_ts(pts[i].timestamp) - _parse_ts(pts[i - 1].timestamp)).total_seconds() / 86400
        if dt <= 0:
            continue
        d = _dist_km((pts[i - 1].lat, pts[i - 1].lon), (pts[i].lat, pts[i].lon))
        spd = d / dt
        if spd > cfg.movement_speed_km_per_day and (best is None or spd > best[0]):
            best = (spd, d, dt, pts[i - 1], pts[i])
    if best is None:
        return None
    spd, d, dt, p0, p1 = best
    sev = "warning" if spd < 2 * cfg.movement_speed_km_per_day else "critical"
    return Alert(
        tiger_id=p1.tiger_id, alert_type="unusual_movement",
        window=_window_label(now, cfg.current_window_days),
        timestamp=now.isoformat(),
        evidence=(f"{d:.1f} km in {dt:.1f} days between {p0.station_id} and "
                  f"{p1.station_id} ({spd:.1f} km/day)"),
        metric="movement_speed_km_per_day", value=round(spd, 2),
        threshold=cfg.movement_speed_km_per_day, severity=sev,
        confidence=min(1.0, spd / (3 * cfg.movement_speed_km_per_day)),
        explanation="Check timestamps for camera-clock errors first; if "
                    "valid, possible chase, dispersal or translocation.",
    )


def eval_activity_anomaly(counts_history, current_count, now, cfg, tiger_id):
    """activity_anomaly: current detection count deviates from history mean."""
    if len(counts_history) < 5:
        return None
    import statistics
    mean = statistics.mean(counts_history)
    sd = statistics.stdev(counts_history) if len(counts_history) > 1 else 1.0
    if sd <= 0:
        sd = max(1.0, mean * 0.3)
    z = (current_count - mean) / sd
    if abs(z) < cfg.activity_z_threshold:
        return None
    sev = "warning" if abs(z) < 3 else "critical"
    direction = "surge" if z > 0 else "drop"
    return Alert(
        tiger_id=tiger_id, alert_type="activity_anomaly",
        window=_window_label(now, cfg.current_window_days),
        timestamp=now.isoformat(),
        evidence=f"{current_count} detections this window vs history "
                 f"mean {mean:.1f} ± {sd:.1f} ({len(counts_history)} windows)",
        metric="activity_z_score", value=round(z, 2),
        threshold=cfg.activity_z_threshold, severity=sev,
        confidence=min(1.0, abs(z) / 5),
        explanation=(f"Detection activity {direction} — check for camera "
                     "malfunction, baiting, or real behaviour change."),
    )


# ---------------------------------------------------------------------------
# debounce / persistence
# ---------------------------------------------------------------------------
def apply_debounce(alerts_this_run, previous_alert_keys, cfg):
    """Escalate alerts sustained across confirmation_windows runs.

    `previous_alert_keys`: {alert_key: consecutive_count} from prior runs.
    Returns (alert_objects, updated_keys).
    """
    updated = {}
    out = []
    for a in alerts_this_run:
        key = f"{a.tiger_id}:{a.alert_type}"
        count = previous_alert_keys.get(key, 0) + 1
        updated[key] = count
        if count >= cfg.confirmation_windows and a.severity != "critical":
            a.severity = "escalated"
            a.explanation = (f"Sustained across {count} consecutive evaluation "
                             f"windows. " + a.explanation)
        out.append(a)
    return out, updated


def evaluate_all(tiger_detections, other_tigers, identity_records,
                 weekly_counts, now, cfg, previous_alert_keys):
    """Run every deterministic evaluator; apply debounce. Returns alerts."""
    alerts = []
    for tid, pts in tiger_detections.items():
        base = [p for p in pts
                if (_parse_ts(now.isoformat()) - _parse_ts(p.timestamp)).days
                >= cfg.current_window_days]
        cur = [p for p in pts
               if (_parse_ts(now.isoformat()) - _parse_ts(p.timestamp)).days
               < cfg.current_window_days]
        a = eval_core_shift(base, cur, now, cfg)
        if a:
            alerts.append(a)
        a = eval_unusual_movement(pts, now, cfg)
        if a:
            alerts.append(a)
        if tid in weekly_counts:
            a = eval_activity_anomaly(weekly_counts[tid], len(cur), now, cfg, tid)
            if a:
                alerts.append(a)
    if len(tiger_detections) >= 2:
        ids = list(tiger_detections.items())
        for (ta, pa), (tb, pb) in [(ids[0], ids[1])]:
            a = eval_territorial_overlap(pa, pb, now, cfg, ta)
            if a:
                alerts.append(a)
    for rec in identity_records:
        a = eval_new_identity(rec, now, cfg)
        if a:
            alerts.append(a)
    alerts, keys = apply_debounce(alerts, previous_alert_keys, cfg)
    return alerts, keys


if __name__ == "__main__":
    now = datetime(2026, 8, 14)
    cfg = AlertConfigV2()
    base = [DetectionPoint(f"S{i%8:02d}", (now - timedelta(days=30 + i)).isoformat(),
                           21.65, 79.35, "T1", 0.9) for i in range(12)]
    cur = [DetectionPoint(f"S{i%5:02d}", (now - timedelta(days=1, hours=i * 8)).isoformat(),
                          21.65, 79.43, "T1", 0.9) for i in range(6)]
    a = eval_core_shift(base, cur, now, cfg)
    print(a.as_dict() if a else "no core_shift")
    u = eval_unusual_movement(cur + [DetectionPoint("S09", (now - timedelta(days=2)).isoformat(),
                                                   21.5, 79.0, "T1", 0.9)], now, cfg)
    print(u.as_dict() if u else "no unusual_movement")
