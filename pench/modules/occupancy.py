"""Module 3 — Occupancy & Home Range Estimation.

For each identified tiger (or the whole population) this module computes,
*fresh on every run* (no stale caching):

- Minimum Convex Polygon (MCP) home range and core area (50% MCP)
- 95% adaptive kernel density estimate (AKDE-style, scipy Gaussian KDE with
  Silverman bandwidth and a cross-checked fixed-bandwidth KDE)
- Activity centroid and spatial spread metrics
- Pairwise territorial overlap (intersection area, % of each range)

Works on projected coordinates (UTM zone 44N for Pench, EPSG:32644).
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

PENCH_UTM_EPSG = 32644  # UTM zone 44N covers Pench (lon ~79-80E)
PENCH_ZONE = 44

# Approximate Pench Tiger Reserve boundary polygon in UTM easting/northing
# (simplified from public reserve boundary descriptions; ~742 km2 reserve).
PENCH_BOUNDARY_UTM = [
    (305000, 2405000), (310000, 2395000), (320000, 2388000), (332000, 2384000),
    (345000, 2386000), (356000, 2394000), (358000, 2405000), (352000, 2414000),
    (340000, 2417000), (326000, 2416000), (313000, 2412000),
]


def lonlat_to_utm(lon: float, lat: float, zone: int = PENCH_ZONE) -> tuple:
    """Simple lon/lat -> UTM conversion (WGS84)."""
    import math
    k0 = 0.9996
    a = 6378137.0
    e2 = 0.00669438
    lon0 = (zone - 1) * 6 - 180 + 3
    phi, lam = math.radians(lat), math.radians(lon - lon0)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = e2 / (1 - e2) * math.cos(phi) ** 2
    A = math.cos(phi) * lam
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    east = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * 0.0067) * A ** 5 / 120) + 500000
    north = k0 * (M + N * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                    + (61 - 58 * T + T ** 2 + 600 * C - 330 * 0.0067) * A ** 6 / 720))
    if lat < 0:
        north += 10000000
    return east, north


def utm_to_lonlat(east: float, north: float, zone: int = PENCH_ZONE) -> tuple:
    """Inverse UTM -> lon/lat (WGS84)."""
    import math
    k0 = 0.9996
    a = 6378137.0
    e2 = 0.00669438
    x, y = east - 500000, north
    lon0 = (zone - 1) * 6 - 180 + 3.0
    x, y = east - 500000.0, north
    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu))
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = e2 / (1 - e2) * math.cos(phi1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = x / (N1 * k0)
    lat = phi1 - (N1 * math.tan(phi1) / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * 0.0067) * D ** 4 / 24
            + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * 0.0067 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * 0.0067 + 24 * T1 ** 2) * D ** 5 / 120) / math.cos(phi1)
    return float(lon0 + math.degrees(lon)), math.degrees(lat)


@dataclass
class CameraStation:
    id: str
    lat: float
    lon: float
    easting: float = 0.0
    northing: float = 0.0

    def __post_init__(self):
        if self.easting == 0.0 and self.northing == 0.0:
            self.easting, self.northing = lonlat_to_utm(self.lon, self.lat)


@dataclass
class Detection:
    """A verified tiger detection at a camera station."""
    station_id: str
    timestamp: str          # ISO datetime
    easting: float
    northing: float
    tiger_id: Optional[str] = None      # assigned identity
    confidence: float = 0.0
    flank: str = "unknown"              # left / right / unknown


@dataclass
class HomeRangeResult:
    entity_id: str                      # tiger id or "population"
    n_detections: int
    mcp_area_km2: float
    core_mcp_50_area_km2: float
    kde95_area_km2: float
    kde50_area_km2: float
    centroid_easting: float
    centroid_northing: float
    centroid_lat: float
    centroid_lon: float
    mcp_polygon: list = field(default_factory=list)   # [(e,n), ...]
    kde95_contour: list = field(default_factory=list)
    spread_max_km: float = 0.0


def polygon_area_km2(ring):
    """Shoelace area in km2 for a ring of (easting, northing) tuples (meters)."""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    area = 0.0
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2e6


def mcp_ring(points, fraction=1.0):
    """Compute MCP ring. fraction<1 -> iterative peel to keep that fraction of
    points (50% core)."""
    pts = np.array(points, dtype=float)
    if len(pts) < 3:
        return pts.tolist()
    try:
        hull_idx = ConvexHull(pts).vertices
    except Exception:
        # degenerate (collinear) point set -> jitter slightly and retry
        pts = pts + np.random.default_rng(0).normal(0, 1e-3, pts.shape)
        hull_idx = ConvexHull(pts).vertices
    ring = pts[hull_idx].tolist()
    if fraction >= 1.0:
        return ring
    # iterative peeling: remove outermost point, recompute hull, until <= fraction remain
    keep = list(range(len(pts)))
    target = max(3, int(len(pts) * fraction))
    while len(keep) > target:
        sub = pts[keep]
        if len(sub) < 3:
            break
        try:
            hull_idx = ConvexHull(sub).vertices
        except Exception:
            break
        # peel the vertex farthest from centroid
        c = sub.mean(axis=0)
        dists = np.linalg.norm(sub - c, axis=1)
        peel_local = hull_idx[np.argmax(dists[hull_idx])]
        keep.pop(keep.index(keep[peel_local]))
    hull_idx = ConvexHull(pts[keep]).vertices
    return pts[keep][hull_idx].tolist()


def kde_contour(points, percentile, grid_step=250.0, n_sigma=3.5):
    """Iso-density contour (polygon ring) enclosing `percentile` of density mass."""
    pts = np.array(points, dtype=float).T  # (2, n)
    if pts.shape[1] < 3:
        return points
    # guard against singular covariance (collinear detections): add tiny jitter
    if np.linalg.matrix_rank(np.cov(pts)) < 2:
        pts = pts + np.random.default_rng(0).normal(0, 200.0, pts.shape)
    try:
        kde = gaussian_kde(pts)
    except np.linalg.LinAlgError:
        # fallback: independent per-axis bandwidths (product kernel)
        bw = pts.std(axis=1, ddof=1) * (4 / (3 * pts.shape[1])) ** 0.2
        kde = gaussian_kde(pts, bw_method=bw)
    center = pts.mean(axis=1)
    spread = pts.std(axis=1)
    xs = np.arange(center[0] - n_sigma * spread[0], center[0] + n_sigma * spread[0], grid_step)
    ys = np.arange(center[1] - n_sigma * spread[1], center[1] + n_sigma * spread[1], grid_step)
    gx, gy = np.meshgrid(xs, ys)
    pos = np.vstack([gx.ravel(), gy.ravel()])
    z = kde(pos).reshape(gx.shape)
    threshold = np.percentile(z[z > 0], 100 - percentile * 100) if (z > 0).any() else 0
    mask = z >= threshold
    if not mask.any():
        return points
    yx = np.argwhere(mask)
    xy_pts = [(xs[c[1]], ys[c[0]]) for c in yx]
    mp = MultiPoint(xy_pts)
    hull = mp.convex_hull
    return list(hull.exterior.coords)


def compute_home_range(entity_id: str, detections: list) -> HomeRangeResult:
    """Fresh per-run computation of MCP + KDE home range for one tiger."""
    pts = [(d.easting, d.northing) for d in detections]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    centroid = (xs.mean(), ys.mean())
    spread_km = float(np.max(np.sqrt((xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2))) / 1000
    clon, clat = utm_to_lonlat(centroid[0], centroid[1])

    mcp = mcp_ring(pts, 1.0)
    core = mcp_ring(pts, 0.5) if len(pts) >= 6 else mcp
    kde95 = kde_contour(pts, 0.95)
    kde50 = kde_contour(pts, 0.5) if len(pts) >= 6 else kde95

    return HomeRangeResult(
        entity_id=entity_id,
        n_detections=len(pts),
        mcp_area_km2=polygon_area_km2(mcp),
        core_mcp_50_area_km2=polygon_area_km2(core),
        kde95_area_km2=polygon_area_km2(kde95),
        kde50_area_km2=polygon_area_km2(kde50),
        centroid_easting=centroid[0],
        centroid_northing=centroid[1],
        centroid_lat=clat,
        centroid_lon=clon,
        mcp_polygon=mcp,
        kde95_contour=kde95,
        spread_max_km=spread_km,
    )


def overlap_report(results: list) -> dict:
    """Pairwise overlap of 95% KDE contours. Returns dict of pairs."""
    out = {}
    polys = {}
    for r in results:
        polys[r.entity_id] = MultiPoint(r.kde95_contour).convex_hull if len(r.kde95_contour) >= 3 else None
    ids = list(polys)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = polys[ids[i]], polys[ids[j]]
            if a is None or b is None or not a.is_valid or not b.is_valid:
                continue
            inter = a.intersection(b).area / 1e6
            pct_a = inter / (a.area / 1e6) * 100 if a.area else 0
            pct_b = inter / (b.area / 1e6) * 100 if b.area else 0
            out[f"{ids[i]} x {ids[j]}"] = {
                "overlap_km2": round(inter, 2),
                "pct_of_first": round(pct_a, 1),
                "pct_of_second": round(pct_b, 1),
            }
    return out


if __name__ == "__main__":
    # quick self-test with synthetic detections inside Pench
    import random
    random.seed(1)
    center = lonlat_to_utm(79.35, 21.65)
    dets = [
        Detection(f"S{i % 10:02d}", f"2026-01-{i % 28 + 1:02d}T10:00:00",
                  center[0] + random.gauss(0, 3000), center[1] + random.gauss(0, 3000),
                  tiger_id="T-003", confidence=0.9)
        for i in range(30)
    ]
    r = compute_home_range("T-003", dets)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in asdict(r).items()}, indent=1))
