"""Demo runner for the Pench pipeline.

1. Builds a small synthetic ingestion folder:
   - 6 "animal" frames = real tiger flank images from the ATRW dataset
     (resized/padded onto camera-trap-style backgrounds with timestamps)
   - 4 blank frames  = dark night-vision style empty frames (gradient noise)
   - 2 borderline frames = very low-contrast dark images (should land in review)
   Camera stations use realistic coordinates around Pench Tiger Reserve.
2. Builds a synthetic history file (historical verified detections for three
   tigers) so home-range and alert modules have a baseline.
3. Runs the four-module pipeline: current detections are appended to the
   history and Modules 3-4 operate on the combined dataset.
4. Draws a home-range map (MCP + 95% KDE contour) with camera stations.
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pench.pipeline import run_pipeline
from pench.modules.occupancy import PENCH_BOUNDARY_UTM

random.seed(42)
PROJECT = Path(__file__).resolve().parent.parent
ATRW = PROJECT / "datasets" / "atrw" / "train"
CCT_ANIMAL = PROJECT / "datasets" / "cct_subset" / "train" / "animal"
DEMO = PROJECT / "demo"
INGEST = DEMO / "ingest"
OUTPUT = DEMO / "output"

# Three demo tigers with distinct activity centers inside Pench TR.
# IDs match the ATRW identity numbers in the trained Re-ID catalogue so the
# demo's flank photos get confirmed as known identities.
TIGERS = {
    "85": {"center": (21.6853, 79.2520), "spread": 1800.0},   # S01 area
    "172": {"center": (21.6312, 79.4155), "spread": 2200.0},  # S02 area
    "252": {"center": (21.6110, 79.3300), "spread": 1500.0},  # S03 area (will shift)
}
# tiger 252: last-30-days detections are shifted 8 km east -> triggers core_shift alert
T252_CURRENT_SHIFT = (0.072, 0.0)  # ~8 km lon shift, current window only

CAMERAS = {
    "S01": (21.6853, 79.2520),
    "S02": (21.6312, 79.4155),
    "S03": (21.6110, 79.3300),
    "S04": (21.6690, 79.2910),
}


def find_atrw_tiger_imgs(n=8):
    imgs = sorted(p for p in ATRW.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if len(imgs) < n:
        raise RuntimeError(f"ATRW images not found at {ATRW} ({len(imgs)} files)")
    step = max(1, len(imgs) // n)
    return [imgs[i * step] for i in range(n)]


def stamp(img, text, color="white"):
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 10), text, fill=color, font=font)
    return img


def make_animal_frame(src, station, t, size=(640, 480)):
    """Composite an ATRW flank image onto a real camera-trap background so
    the triage classifier sees a realistic frame."""
    bg_src = random.choice(list(CCT_ANIMAL.iterdir())) if CCT_ANIMAL.exists() else None
    if bg_src:
        canvas = Image.open(bg_src).convert("RGB").resize(size)
    else:
        canvas = Image.new("RGB", size, (60, 70, 45))
    im = Image.open(src).convert("RGB")
    im.thumbnail((420, 300))
    alpha = Image.new("L", im.size, 255)
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2), alpha)
    return stamp(canvas, f"{station}  {t.strftime('%Y-%m-%d %H:%M:%S')}")


def make_blank_frame(t, dark=False, size=(640, 480)):
    arr = np.random.normal(18 if dark else 30, 6 if dark else 9, (size[1], size[0], 3))
    ys, xs = np.mgrid[0:size[1], 0:size[0]]
    vig = 1 - 0.4 * (((xs - size[0] / 2) / size[0]) ** 2 + ((ys - size[1] / 2) / size[1]) ** 2)
    arr *= vig[..., None]
    im = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    return stamp(im, f"S03  {t.strftime('%Y-%m-%d %H:%M:%S')}")


def build_history(now, out_path):
    """Synthetic historical verified detections (from trap review logs)."""
    from pench.modules.occupancy import lonlat_to_utm
    rows = []
    for tid, cfg in TIGERS.items():
        lat, lon = cfg["center"]
        spread = cfg["spread"]
        # baseline window: 60-30 days ago, 24 detections (tiger 252 unshifted)
        for i in range(24):
            t = now - timedelta(days=random.randint(31, 60), hours=random.randint(0, 23))
            elat = random.gauss(0, spread / 111000)
            elon = random.gauss(0, spread / (111000 * np.cos(np.radians(lat))))
            rows.append(f"{tid},{t.isoformat()},S{random.randint(1,4):02d},"
                        f"{round(lat + elat, 5)},{round(lon + elon, 5)},0.91")
        # current window: last 30 days; tiger 252 shifted 8 km east -> alert
        shift = T252_CURRENT_SHIFT if tid == "252" else (0.0, 0.0)
        for i in range(7):
            t = now - timedelta(days=random.randint(1, 25), hours=random.randint(0, 23))
            elat = random.gauss(0, spread / 111000)
            elon = random.gauss(0, spread / (111000 * np.cos(np.radians(lat)))) + shift[0]
            rows.append(f"{tid},{t.isoformat()},S{random.randint(1,4):02d},"
                        f"{round(lat + elat, 5)},{round(lon + elon, 5)},0.93")
    out_path.write_text("tiger_id,timestamp,station_id,lat,lon,confidence\n" + "\n".join(rows))
    print("history written:", out_path, f"({len(rows)} rows)")


def main():
    INGEST.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for d in (INGEST, OUTPUT):
        for p in d.iterdir():
            p.unlink() if p.is_file() else None

    tigers = find_atrw_tiger_imgs(6)
    now = datetime(2026, 8, 14, 6, 30)

    # current ingest frames: two per demo tiger (real flank photos)
    for i, src in enumerate(tigers):
        t = now - timedelta(hours=random.randint(2, 60))
        tid = list(TIGERS.keys())[i % 3]
        # station near the tiger's activity center
        station = {"85": "S01", "172": "S02", "252": "S03"}[tid]
        p = INGEST / f"{station}_{i:02d}_{tid}.jpg"
        make_animal_frame(src, station, t).save(p, quality=90)
        print("animal:", p.name)

    for j in range(4):
        t = now - timedelta(hours=random.randint(1, 48))
        p = INGEST / f"S03_{10 + j:02d}_blank.jpg"
        make_blank_frame(t).save(p, quality=85)
        print("blank  :", p.name)

    for k in range(2):
        t = now - timedelta(hours=random.randint(1, 20))
        p = INGEST / f"S04_{20 + k:02d}_borderline.jpg"
        make_blank_frame(t, dark=True).save(p, quality=80)
        print("border :", p.name)

    history_path = OUTPUT / "detection_history.csv"
    build_history(now, history_path)

    results = run_pipeline(INGEST, OUTPUT, CAMERAS, now,
                           history_csv=history_path)

    draw_map(results, CAMERAS, OUTPUT / "home_range_map.png")
    print("\nDemo complete. Outputs in", OUTPUT)


def draw_map(results, cameras, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from pench.modules.occupancy import utm_to_lonlat

    hr = results["stages"]["module3_home_range"]["per_tiger"]
    fig, ax = plt.subplots(figsize=(10, 8))
    boundary = [utm_to_lonlat(e, n) for e, n in PENCH_BOUNDARY_UTM]
    ax.add_patch(MplPolygon(boundary, closed=True, fill=True,
                            facecolor="#d8e8c8", edgecolor="#4a6b3a",
                            linewidth=2, alpha=0.6, label="Pench TR boundary (approx.)"))
    colors = ["#c0392b", "#2980b9", "#8e44ad"]
    for (tid, r), c in zip(hr.items(), colors):
        ax.plot(r["centroid_lon"], r["centroid_lat"], "o", color=c, markersize=9,
                label=f"{tid}  MCP {r['mcp_area_km2']:.0f} km2 / KDE95 {r['kde95_area_km2']:.0f} km2")
        ax.annotate(tid, (r["centroid_lon"], r["centroid_lat"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9, color=c)
    for sid, (lat, lon) in cameras.items():
        ax.plot(lon, lat, "s", color="black", markersize=7)
        ax.annotate(sid, (lon, lat), textcoords="offset points", xytext=(6, -12),
                    fontsize=9, color="black")
    alerts = results["stages"]["module4_alerts"]
    title = "Pench TR — Estimated Tiger Home Ranges (this run)"
    if alerts:
        title += "\n" + " | ".join(f"{a['tiger_id']}: {a['type']} ({a['severity']})" for a in alerts)
    ax.set_title(title)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("map saved:", out_path)


if __name__ == "__main__":
    main()
