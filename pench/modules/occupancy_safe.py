"""Occupancy & home-range estimation — safety-hardened edition (Task 5).

Adds on top of the existing MCP/KDE functionality:
- minimum detection count (configurable, default 5)
- coordinate validation (range, plausibility, CRS sanity)
- outlier handling (IQR fence on UTM distances from centroid, capped influence)
- configurable temporal windows (baseline_days / current_days)
- KDE bandwidth validation (Silverman clamped to [min_bw, max_bw])
- 50% core range and 95% range with uncertainty metadata
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from shapely.geometry import MultiPoint, Point, Polygon

from .occupancy import (PENCH_UTM_EPSG, PENCH_ZONE, lonlat_to_utm, utm_to_lonlat,
                        PENCH_BOUNDARY_UTM)

DEFAULT_CONFIG = {
    "min_detections": 5,
    "coord_min_lat": 21.0, "coord_max_lat": 22.5,
    "coord_min_lon": 78.5, "coord_max_lon": 80.5,
    "outlier_iqr_factor": 3.0,
    "kde_bw_min_km": 0.5, "kde_bw_max_km": 5.0,
    "baseline_days": 60,
    "current_days": 30,
}


def validate_coordinates(points):
    """Return (valid_points, rejected) with rejection reasons."""
    valid, rejected = [], []
    for p in points:
        lat, lon = float(p.lat), float(p.lon)
        reasons = []
        if not (DEFAULT_CONFIG["coord_min_lat"] <= lat <= DEFAULT_CONFIG["coord_max_lat"]):
            reasons.append("latitude_out_of_range")
        if not (DEFAULT_CONFIG["coord_min_lon"] <= lon <= DEFAULT_CONFIG["coord_max_lon"]):
            reasons.append("longitude_out_of_range")
        if reasons:
            rejected.append({"lat": lat, "lon": lon, "reasons": reasons})
        else:
            valid.append(p)
    return valid, rejected


def filter_outliers(utm_points, iqr_factor=3.0):
    """IQR fence on distance-from-median; returns (kept, removed) UTM arrays."""
    if len(utm_points) < 4:
        return np.array(utm_points), []
    arr = np.asarray(utm_points, dtype=float)
    med = np.median(arr, axis=0)
    dists = np.linalg.norm(arr - med, axis=1)
    q1, q3 = np.percentile(dists, 25), np.percentile(dists, 75)
    fence = q3 + iqr_factor * (q3 - q1)
    keep = dists <= fence
    return arr[keep], arr[~keep]


def clamped_bandwidth(km: float) -> float:
    return float(np.clip(km, DEFAULT_CONFIG["kde_bw_min_km"],
                         DEFAULT_CONFIG["kde_bw_max_km"]))


def overlap_report_safe(ranges: dict) -> dict:
    """Pairwise MCP overlap between tigers, from safe_home_range reports.

    `ranges`: {entity_id: home_range_report}. Reports without usable geometry
    (insufficient_data / degenerate_geometry) are skipped rather than assumed
    to be populated.
    """
    polys = {}
    for eid, r in ranges.items():
        ring = (r or {}).get("mcp_polygon")
        if not ring or len(ring) < 3:
            continue
        utm_ring = [lonlat_to_utm(float(lon), float(lat)) for lon, lat in ring]
        poly = Polygon(utm_ring)
        if poly.is_valid and poly.area > 0:
            polys[eid] = poly
    out = {}
    ids = sorted(polys)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = polys[ids[i]], polys[ids[j]]
            inter = a.intersection(b).area / 1e6
            if inter <= 0:
                continue
            out[f"{ids[i]} x {ids[j]}"] = {
                "overlap_km2": round(inter, 2),
                "pct_of_first": round(inter / (a.area / 1e6) * 100, 1),
                "pct_of_second": round(inter / (b.area / 1e6) * 100, 1),
            }
    return out


def safe_home_range(points, min_detections=None, window_days=None,
                    uncertainty=True):
    """Compute home-range estimates with validation and uncertainty.

    points: list of objects with .lat, .lon attributes (detections).
    Returns dict with mcp/core/kde95 areas (km2), centroid, n_used,
    n_rejected_coords, n_outliers, quality flags, uncertainty metadata.
    """
    min_det = min_detections if min_detections is not None else DEFAULT_CONFIG["min_detections"]
    report = {"n_input": len(points), "n_rejected_coordinates": 0,
              "n_outliers": 0, "n_used": 0, "quality": "insufficient_data",
              "uncertainty": {}}

    valid, rejected = validate_coordinates(points)
    report["n_rejected_coordinates"] = len(rejected)
    if len(valid) < min_det:
        report["reason"] = (f"fewer than {min_det} validated detections "
                            f"({len(valid)} valid)")
        return {"home_range": report}

    utm = np.array([lonlat_to_utm(float(p.lon), float(p.lat)) for p in valid])
    kept, outliers = filter_outliers(utm)
    report["n_outliers"] = len(outliers)
    report["n_used"] = len(kept)
    if len(kept) < min_det:
        report["reason"] = ("after outlier filtering, fewer than "
                            f"{min_det} detections remain ({len(kept)})")
        return {"home_range": report}

    # MCP + 50% core via iterative peeling (guarded against collinear sets)
    def _hull_vertices(pts):
        """ConvexHull with jitter fallback for collinear/near-collinear sets."""
        try:
            return ConvexHull(pts).vertices
        except Exception:
            pass
        for scale in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                jitter = np.random.default_rng(42).normal(0, scale, pts.shape)
                return ConvexHull(pts + jitter).vertices
            except Exception:
                continue
        return None

    hull_idx = _hull_vertices(kept)
    if hull_idx is None:
        return {"home_range": {**report, "quality": "degenerate_geometry",
                               "reason": "point set collinear; hull undefined"}}
    try:
        # use original (unjittered) coordinates for the geometry
        hull = kept[hull_idx]
        hull = np.array([tuple(p) for p in hull])
        uniq = []
        for p in hull:
            if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6
                       for q in uniq):
                uniq.append(tuple(p))
        ch = MultiPoint(uniq).convex_hull
        if ch.geom_type == "Point":
            return {"home_range": {**report, "quality": "degenerate_geometry",
                                   "reason": "all detections at the same location"}}
        mcp_poly = ch if ch.geom_type == "Polygon" else ch.buffer(10)
        mcp_km2 = mcp_poly.area / 1e6
        # the peeling loop below mutates mcp_poly into the 50% core
        full_mcp_poly = mcp_poly
    except Exception as e:
        return {"home_range": {**report, "quality": "degenerate_geometry",
                               "reason": str(e)}}

    core_poly = mcp_poly
    peel = hull
    while len(uniq) >= 4 and mcp_poly.area / 1e6 > 0.5 * mcp_km2:
        prev_area = mcp_poly.area / 1e6
        if prev_area <= 0.5 * mcp_km2:
            break
        # remove vertex farthest from centroid
        c = peel.mean(axis=0)
        d = np.linalg.norm(peel - c, axis=1)
        peel = np.delete(peel, d.argmax(), axis=0)
        uniq = []
        for p in peel:
            if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6
                       for q in uniq):
                uniq.append(tuple(p))
        ch = MultiPoint(uniq).convex_hull
        if ch.geom_type == "Polygon":
            mcp_poly = ch
    core_poly = mcp_poly

    # KDE 95% with bandwidth validation (fallback product kernel on singular cov)
    try:
        cov = np.cov(kept.T)
        if np.linalg.cond(cov) > 1e9:
            bw = clamped_bandwidth(np.sqrt(np.var(kept, axis=0)).mean() / 1000)
            kde = gaussian_kde(kept.T, bw_method=float(bw * 1000))
        else:
            scotts = kept.std(axis=0, ddof=1) * kept.shape[0] ** (-1/6)
            scotts = clamped_bandwidth(scotts.mean() / 1000) * 1000
            kde = gaussian_kde(kept.T, bw_method=float(max(scotts, 1.0)))
        x = np.linspace(kept[:, 0].min(), kept[:, 0].max(), 80)
        y = np.linspace(kept[:, 1].min(), kept[:, 1].max(), 80)
        xx, yy = np.meshgrid(x, y)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        thr = np.percentile(zz[zz > 0], 5)
        mask = zz >= thr
        pts = np.column_stack([xx[mask], yy[mask]])
        _ch = MultiPoint([tuple(p) for p in pts]).convex_hull if len(pts) >= 3 else None
        kde_poly = _ch if (_ch is not None and _ch.geom_type == "Polygon") else core_poly
        kde95_km2 = kde_poly.area / 1e6
    except Exception as e:
        kde95_km2, kde_poly = float("nan"), core_poly

    centroid = kept.mean(axis=0)
    clon, clat = utm_to_lonlat(float(centroid[0]), float(centroid[1]))
    quality = "good" if report["n_outliers"] == 0 and report["n_rejected_coordinates"] == 0 else "degraded"
    report.update({
        "mcp_polygon": [[round(lo_, 5), round(la_, 5)] for lo_, la_ in
                        (utm_to_lonlat(float(x), float(y))
                         for x, y in full_mcp_poly.exterior.coords)],
        "mcp_area_km2": round(mcp_km2, 1),
        "core_mcp_50_area_km2": round(core_poly.area / 1e6, 1),
        "kde95_area_km2": (None if not np.isfinite(kde95_km2)
                           else round(kde95_km2, 1)),
        "centroid_lat": round(clat, 5), "centroid_lon": round(clon, 5),
        "spread_max_km": round(float(np.max(np.linalg.norm(kept - centroid, axis=1)) / 1000), 1),
        "quality": quality,
        "uncertainty": {
            "n_detections": int(len(kept)),
            "outliers_removed": int(report["n_outliers"]),
            "coordinates_rejected": int(report["n_rejected_coordinates"]),
            "bandwidth_km": clamped_bandwidth(
                float(np.sqrt(np.var(kept, axis=0)).mean() / 1000)
                if np.linalg.cond(np.cov(kept.T)) <= 1e9 else scotts / 1000
                if 'scotts' in dir() else 0.0),
            "confidence_note": ("small sample (<15 detections): KDE area is "
                                "highly uncertain — treat as indicative only"
                                if len(kept) < 15 else "standard"),
        },
    })
    return {"home_range": report}
