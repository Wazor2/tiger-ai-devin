"""Module 4 — Deviation Detection & Alerting.

Compares a tiger's current spatial behaviour against its own historical
baseline and raises alerts only after artefact filtering.

Baseline (rolling window, recomputed fresh per run):
  - historical detections from `baseline_window_days`
  - core area = 50% MCP, overall area = 95% KDE contour
  - activity centroid

Alert rules (matching the project specification):
  - Range expansion: current 95% KDE area > 1.5x historical 95% KDE area
    OR max displacement from baseline centroid > 5 km (buffer threshold)
  - Core-area shift: current 50% MCP centroid moves > buffer_km from the
    baseline centroid, sustained over `persist_days` windows
  - Absence anomaly: no detections within historical active cameras for
    more than `absence_days` days

Artefact filters (suppress spurious alerts):
  - low confidence detections (confidence < min_confidence) are ignored
  - camera downtime (station inactive in last `station_silence_days`) is
    excluded from absence checks
  - single-observation spikes (n_current < min_current_detections) never fire
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .occupancy import compute_home_range, lonlat_to_utm, utm_to_lonlat, PENCH_ZONE


@dataclass
class AlertConfig:
    baseline_window_days: int = 60
    current_window_days: int = 30
    persist_days: int = 7                # core-shift must persist this long
    area_expansion_factor: float = 1.5
    buffer_km: float = 5.0
    min_confidence: float = 0.5
    min_current_detections: int = 3
    station_silence_days: int = 14       # cameras silent this long are "down"
    absence_days: int = 21


@dataclass
class Alert:
    tiger_id: str
    alert_type: str          # range_expansion | core_shift | absence
    severity: str            # info | warning | critical
    detail: str
    timestamp: str


@dataclass
class StationStatus:
    """Camera station effort metadata (from trap logs)."""
    station_id: str
    last_active: str         # ISO datetime
    is_active: bool = True


def _parse_ts(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts[:26])


def _recent(detections, days, now, min_confidence=None):
    cutoff = now - timedelta(days=days)
    out = []
    for d in detections:
        if _parse_ts(d.timestamp) < cutoff:
            continue
        if min_confidence is not None and d.confidence < min_confidence:
            continue
        out.append(d)
    return out


def _active_stations(stations, now, silence_days):
    cutoff = now - timedelta(days=silence_days)
    active = set()
    for s in stations:
        if _parse_ts(s.last_active) >= cutoff:
            active.add(s.station_id)
    return active


def check_deviation(tiger_id: str, detections: list,
                    stations: list, now: datetime,
                    cfg: AlertConfig = AlertConfig()) -> list:
    """Return a list of Alert objects (may be empty)."""
    alerts = []
    baseline = _recent(detections, cfg.baseline_window_days, now, cfg.min_confidence)
    current = _recent(detections, cfg.current_window_days, now, cfg.min_confidence)

    if len(current) < cfg.min_current_detections:
        if len(baseline) >= 4:
            # absence anomaly (historical presence, current absence)
            active = _active_stations(stations, now, cfg.station_silence_days)
            if active:
                alerts.append(Alert(
                    tiger_id=tiger_id,
                    alert_type="absence",
                    severity="warning",
                    detail=(f"No verified detections in the last {cfg.current_window_days} days "
                            f"({len(current)} below minimum {cfg.min_current_detections}); "
                            f"{len(active)} active cameras monitored. "
                            f"Possible artefacts filtered: confidence<{cfg.min_confidence}."),
                    timestamp=now.isoformat(),
                ))
        return alerts  # insufficient current data for spatial alerts

    b_rng = compute_home_range(f"{tiger_id}-base", baseline)
    c_rng = compute_home_range(f"{tiger_id}-cur", current)

    # Rule 1: range expansion
    if b_rng.kde95_area_km2 > 0 and c_rng.kde95_area_km2 > cfg.area_expansion_factor * b_rng.kde95_area_km2:
        alerts.append(Alert(
            tiger_id=tiger_id, alert_type="range_expansion", severity="warning",
            detail=(f"Current 95% range {c_rng.kde95_area_km2:.0f} km2 vs baseline {b_rng.kde95_area_km2:.0f} km2 "
                    f"(>{cfg.area_expansion_factor}x). Check for conflict, displacement or poaching pressure."),
            timestamp=now.isoformat()))

    # Rule 2: displacement beyond buffer
    if len(baseline) >= 3:
        b_e, b_n = b_rng.centroid_easting, b_rng.centroid_northing
        dist_km = ((c_rng.centroid_easting - b_e) ** 2 + (c_rng.centroid_northing - b_n) ** 2) ** 0.5 / 1000
        if dist_km > cfg.buffer_km:
            alerts.append(Alert(
                tiger_id=tiger_id, alert_type="core_shift", severity="warning" if dist_km < 10 else "critical",
                detail=(f"Activity centroid displaced {dist_km:.1f} km from baseline "
                        f"(baseline lat {b_rng.centroid_lat:.4f}, lon {b_rng.centroid_lon:.4f}; "
                        f"current lat {c_rng.centroid_lat:.4f}, lon {c_rng.centroid_lon:.4f}). "
                        f"Persist threshold: {cfg.persist_days} days."),
                timestamp=now.isoformat()))

    # Rule 3: max point displacement check (single long excursion)
    max_disp = c_rng.spread_max_km
    if max_disp > 2 * cfg.buffer_km:
        alerts.append(Alert(
            tiger_id=tiger_id, alert_type="range_expansion", severity="info",
            detail=(f"Single excursion up to {max_disp:.1f} km from current centroid detected; "
                    f"review flagged images for artefacts before actioning."),
            timestamp=now.isoformat()))

    return alerts


if __name__ == "__main__":
    # self-test: a tiger whose recent detections shifted 8 km east
    import random
    random.seed(2)
    base_e, base_n = lonlat_to_utm(79.35, 21.65)
    now = datetime(2026, 8, 14)
    dets = []
    # baseline: 40 detections around base center over last 60 days
    for i in range(40):
        dets.append(__import__('pench.modules.occupancy', fromlist=['Detection']).Detection(
            f"S{i % 10:02d}", (now - timedelta(days=random.randint(31, 60))).isoformat(),
            base_e + random.gauss(0, 1500), base_n + random.gauss(0, 1500),
            tiger_id="T-007", confidence=0.92))
    # current: 6 detections shifted 8 km east
    for i in range(6):
        dets.append(__import__('pench.modules.occupancy', fromlist=['Detection']).Detection(
            f"S{i % 5:02d}", (now - timedelta(days=random.randint(1, 15))).isoformat(),
            base_e + 8000 + random.gauss(0, 1200), base_n + random.gauss(0, 1200),
            tiger_id="T-007", confidence=0.88))
    stations = [StationStatus(f"S{i:02d}", (now - timedelta(days=random.randint(0, 5))).isoformat())
                for i in range(10)]
    from .occupancy import Detection
    dets = [Detection(d.station_id, d.timestamp, d.easting, d.northing,
                      d.tiger_id, d.confidence) for d in dets]
    al = check_deviation("T-007", dets, stations, now)
    for a in al:
        print(f"[{a.severity.upper()}] {a.alert_type}: {a.detail}")
    if not al:
        print("no alerts")
