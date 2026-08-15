"""Check the demo cards still classify correctly when filmed off a screen.

The rehearsal path is: card on a laptop screen -> phone camera -> Iriun ->
webcam frame. That round trip costs resolution, sharpness and contrast, so the
cards are re-scored through a simulation of it before the rehearsal is trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from pench.model_serving import BlankFilter, TigerReID

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "demo" / "demo_cards"

# (label, webcam long edge, blur radius, brightness, contrast, jpeg quality)
CONDITIONS = [
    ("clean", 1280, 0.0, 1.0, 1.0, 90),
    ("typical", 1280, 0.8, 0.92, 0.9, 70),
    ("harsh", 960, 1.4, 0.78, 0.78, 55),
]


def degrade(img: Image.Image, edge: int, blur: float, bright: float,
            contrast: float, quality: int, tmp: Path) -> Image.Image:
    out = img.copy()
    scale = edge / max(out.size)
    out = out.resize((round(out.width * scale), round(out.height * scale)),
                     Image.LANCZOS)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    out = ImageEnhance.Brightness(out).enhance(bright)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out.save(tmp, quality=quality)
    return Image.open(tmp).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="keep the degraded frames under demo/demo_cards/_sim")
    args = ap.parse_args()

    manifest_path = CARDS / "manifest.json"
    if not manifest_path.exists():
        print("run scripts/build_demo_cards.py first")
        return 1
    manifest = json.loads(manifest_path.read_text())

    blank, reid = BlankFilter(), TigerReID()
    sim_dir = CARDS / "_sim"
    sim_dir.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    for card in manifest["cards"]:
        path = ROOT / card["card"]
        img = Image.open(path).convert("RGB")
        per_card = {}
        for label, edge, blur, bright, contrast, quality in CONDITIONS:
            tmp = sim_dir / f"{path.stem}_{label}.jpg"
            frame = degrade(img, edge, blur, bright, contrast, quality, tmp)
            triage = blank.triage(frame)
            row = {"verdict": triage["verdict"],
                   "animal_confidence": triage["animal_confidence"]}
            if triage["verdict"] in ("animal", "review"):
                res = reid.identify(frame)
                row["reid_decision"] = res["decision"]
                row["top1"] = [res["top3"][0][0], round(res["top3"][0][1], 4)]
            per_card[label] = row
            if not args.save:
                tmp.unlink(missing_ok=True)
        results[card["card"]] = per_card
        name = f'{card["beat"]}{" " + card["tiger_id"] if card.get("tiger_id") else ""}'
        print(name)
        for label, row in per_card.items():
            print("   ", label, row)

    if not args.save:
        sim_dir.rmdir()
    manifest["screen_simulation"] = results
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
