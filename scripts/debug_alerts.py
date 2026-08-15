from datetime import datetime, timedelta
import numpy as np
from pench.modules.occupancy_safe import safe_home_range
from pench.modules.alerting_v2 import (_dist_km, AlertConfigV2, DetectionPoint,
                                       eval_territorial_overlap, eval_unusual_movement)

def _parse_ts(ts):
    return datetime.fromisoformat(ts[:26])

now = datetime(2026, 8, 14)
cfg = AlertConfigV2()
rng = np.random.default_rng(7)

def jit(lat, lon, s=0.004):
    return (lat + rng.normal(0, s), lon + rng.normal(0, s))

# unusual movement: per-pair distances/speeds
mix = [DetectionPoint(f"S{i%10:02d}", (now - timedelta(days=14 + i * 96)).isoformat(),
                      *jit(21.65, 79.35), "T1", 0.9) for i in range(3)] + \
      [DetectionPoint("S09", (now - timedelta(hours=6)).isoformat(), 21.5, 79.0, "T1", 0.9)]
pts = sorted(mix, key=lambda p: _parse_ts(p.timestamp))
print("threshold km/day:", cfg.movement_speed_km_per_day)
for i in range(1, len(pts)):
    dt = (_parse_ts(pts[i].timestamp) - _parse_ts(pts[i - 1].timestamp)).total_seconds() / 86400
    d = _dist_km((pts[i - 1].lat, pts[i - 1].lon), (pts[i].lat, pts[i].lon))
    print(i, round(dt, 2), round(d, 2), round(d / dt, 2) if dt > 0 else None)

# territory: ranges and overlap estimate
pa = [DetectionPoint(f"S{i%10:02d}", (now - timedelta(days=1 + i * 4)).isoformat(),
                     *jit(21.65, 79.35), "T1", 0.9) for i in range(8)]
ha = safe_home_range(pa, min_detections=5)
hb = safe_home_range(
    [DetectionPoint(f"S{i%10:02d}", (now - timedelta(days=1 + i * 4)).isoformat(),
                    *jit(21.651, 79.351), "T1", 0.9) for i in range(8)],
    min_detections=5)
ca = (ha["home_range"]["centroid_lat"], ha["home_range"]["centroid_lon"])
cb = (hb["home_range"]["centroid_lat"], hb["home_range"]["centroid_lon"])
d = _dist_km(ca, cb)
area = hb["home_range"].get("kde95_area_km2", 0) or 0
overlap = max(0.0, area - 3.14 * (d / 2) ** 2) if d < 2 * (area / 3.14) ** 0.5 else 0.0
print("kde95_area_km2:", area, "centroid_dist_km:", round(d, 2), "overlap_est:", round(overlap, 1))
