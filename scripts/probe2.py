import json
import traceback
from datetime import datetime, timedelta

from shapely.geometry import MultiPoint, Polygon

from pench.modules.alerting_v2 import DetectionPoint
from pench.modules.occupancy import lonlat_to_utm

now = datetime(2026, 8, 14)
pts = [DetectionPoint(f"S{i % 10:02d}",
                      (now - timedelta(days=1 + i * 4)).isoformat(),
                      21.65, 79.42, "T1", 0.9) for i in range(6)]
import numpy as np
kept = np.array([lonlat_to_utm(float(p.lon), float(p.lat)) for p in pts])
print("kept shape:", kept.shape, kept.dtype)

try:
    mp = MultiPoint([tuple(p) for p in kept])
    print("MultiPoint ok:", mp)
except Exception as e:
    print("MultiPoint err:", e)
    traceback.print_exc()

try:
    from scipy.spatial import ConvexHull
    hull_idx = ConvexHull(kept).vertices
    print("hull vertices:", hull_idx)
    hull = kept[hull_idx]
    mp = Polygon(MultiPoint([tuple(p) for p in hull]).convex_hull)
    print("MCP polygon ok, area:", mp.area)
except Exception as e:
    print("hull err:", e)
    traceback.print_exc()
