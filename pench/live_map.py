"""Home-range map rendered to a PNG for the dashboard to <img> in.

Regenerated on a timer rather than drawn live in the browser: a static image
carries the same information to a judge with none of the JS-map risk.
"""
from pathlib import Path

from pench.modules.occupancy import PENCH_BOUNDARY_UTM, utm_to_lonlat

COLORS = ["#e74c3c", "#3498db", "#9b59b6", "#f1c40f", "#1abc9c"]


def render_map(raw_ranges: dict, cameras: dict, out_path,
               title: str = "Pench TR - estimated tiger home ranges"):
    """Draw MCP polygons + centroids. Skips ranges without usable geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#11170f")
    ax.set_facecolor("#11170f")
    boundary = [utm_to_lonlat(e, n) for e, n in PENCH_BOUNDARY_UTM]
    ax.add_patch(MplPolygon(boundary, closed=True, facecolor="#1d2a17",
                            edgecolor="#4a6b3a", linewidth=1.5,
                            label="Pench TR (approx.)"))

    drawn = 0
    for (tid, report), color in zip(sorted(raw_ranges.items()), COLORS * 4):
        if report.get("centroid_lat") is None:
            continue
        poly = report.get("mcp_polygon") or []
        if len(poly) >= 4:
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=color,
                                    edgecolor=color, alpha=0.22, linewidth=1.4))
        area = report.get("mcp_area_km2")
        label = f"{tid}" + (f"  MCP {area:.0f} km2" if area else "  MCP n/a")
        ax.plot(report["centroid_lon"], report["centroid_lat"], "o",
                color=color, markersize=8, label=label)
        drawn += 1

    for sid, (lat, lon) in (cameras or {}).items():
        ax.plot(lon, lat, "s", color="#cfd8c5", markersize=5)
        ax.annotate(sid, (lon, lat), textcoords="offset points", xytext=(5, -10),
                    fontsize=7, color="#cfd8c5")

    if not drawn:
        ax.text(0.5, 0.5, "no usable home range yet", transform=ax.transAxes,
                ha="center", color="#8a9a80", fontsize=13)

    ax.set_title(title, color="#e8f0e0", fontsize=11)
    for spine in ax.spines.values():
        spine.set_color("#3a4a32")
    ax.tick_params(colors="#8a9a80", labelsize=7)
    ax.grid(alpha=0.15, color="#4a6b3a")
    if drawn:
        leg = ax.legend(loc="lower left", fontsize=7, facecolor="#1d2a17",
                        edgecolor="#3a4a32")
        for text in leg.get_texts():
            text.set_color("#e8f0e0")
    fig.tight_layout()
    tmp = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    tmp.replace(out_path)        # atomic: the dashboard never reads a half file
    return out_path
