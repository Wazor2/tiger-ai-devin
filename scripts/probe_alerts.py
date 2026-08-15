import json
from datetime import datetime, timedelta
from pench.modules.alerting_v2 import AlertConfigV2, DetectionPoint, eval_core_shift
from pench.modules.occupancy_safe import safe_home_range

now = datetime(2026, 8, 14)
cfg = AlertConfigV2()


def pts(lat, lon, da, n, step=4):
    return [DetectionPoint(f"S{i % 10:02d}",
                           (now - timedelta(days=da + i * step)).isoformat(),
                           lat, lon, "T1", 0.9) for i in range(n)]


base = pts(21.65, 79.35, 35, 12, step=4)
cur_far = pts(21.65, 79.42, 1, 6, step=4)
for label, p in [("base", base), ("cur_far", cur_far)]:
    r = safe_home_range(p)
    print(label, "->", json.dumps(r["home_range"], indent=1, default=str)[:400])
a = eval_core_shift(base, cur_far, now, cfg)
print("core_shift far:", a.as_dict() if a else "None")

# add scatter so points are not collinear
cur_scattered = pts(21.65, 79.42, 1, 6, step=4)
cur_scattered[2].lat = 21.66
cur_scattered[4].lon = 79.43
a2 = eval_core_shift(base, cur_scattered, now, cfg)
print("core_shift scattered:", a2.as_dict() if a2 else "None")
