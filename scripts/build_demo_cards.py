"""Select and export the images used for the live demo rehearsal.

Produces `demo/demo_cards/` (one image per demo beat) plus a manifest recording
the measured blank-filter and Re-ID response for each card, so the rehearsal is
based on verified behaviour rather than hope.

Beats (PENCH_LIVE_DEMO_SPEC "demo content"):
  1. known tiger   - ATRW catalogue image that the Re-ID confirms outright
  2. empty jungle  - camera-trap frame the blank filter rejects
  3. unknown tiger - tiger that is not in the catalogue -> unidentified
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pench.model_serving import BlankFilter, TigerReID

ROOT = Path(__file__).resolve().parents[1]
ATRW = ROOT / "datasets" / "atrw_dl"
CCT_EMPTY = ROOT / "datasets" / "cct_subset" / "val" / "empty"
OUT = ROOT / "demo" / "demo_cards"

KNOWN_IDS = ("172", "85", "252")
CARD_SIZE = (1600, 1200)


def load_train_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    with open(ATRW / "reid_list_train.csv", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                index[row[0].strip()].append(row[1].strip())
    return index


def as_card(src: Path, dest: Path) -> None:
    """Scale-and-crop to 4:3 so the subject fills the frame.

    Letterboxing is deliberately avoided: the padding shrinks the tiger inside
    the 224px model input and measurably destroys the Re-ID match.
    """
    img = Image.open(src).convert("RGB")
    tw, th = CARD_SIZE
    scale = max(tw / img.width, th / img.height)
    img = img.resize((max(tw, round(img.width * scale)),
                      max(th, round(img.height * scale))), Image.LANCZOS)
    left, top = (img.width - tw) // 2, (img.height - th) // 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.crop((left, top, left + tw, top + th)).save(dest, quality=95)


def pick_known(reid: TigerReID, index: dict[str, list[str]], tid: str,
               limit: int) -> tuple[Path, dict] | None:
    best: tuple[float, Path, dict] | None = None
    for name in index.get(tid, [])[:limit]:
        path = ATRW / "train" / name
        if not path.exists():
            continue
        res = reid.identify(Image.open(path).convert("RGB"))
        top = res["top3"][0]
        if top[0] != tid or res["decision"] != "known_identity":
            continue
        if best is None or top[1] < best[0]:
            best = (top[1], path, res)
    return (best[1], best[2]) if best else None


def pick_unknown(reid: TigerReID, blank: BlankFilter,
                 limit: int) -> tuple[Path, dict] | None:
    """Furthest-from-catalogue tiger that the blank filter still calls an animal.

    The beat only reads as "a tiger we don't know" if the blank filter passes the
    frame first, so an unambiguous animal beats a marginally larger distance.
    """
    names = sorted(p.name for p in (ATRW / "test").glob("*.jpg"))
    random.Random(7).shuffle(names)
    best: tuple[tuple[int, float], Path, dict] | None = None
    for name in names[:limit]:
        path = ATRW / "test" / name
        img = Image.open(path).convert("RGB")
        res = reid.identify(img)
        if res["decision"] == "known_identity":
            continue
        animal = blank.triage(img)["verdict"] == "animal"
        key = (1 if animal else 0, res["top3"][0][1])
        if best is None or key > best[0]:
            best = (key, path, res)
    return (best[1], best[2]) if best else None


def pick_empty(blank: BlankFilter, limit: int) -> tuple[Path, dict] | None:
    if not CCT_EMPTY.exists():
        return None
    best: tuple[float, Path, dict] | None = None
    for path in sorted(CCT_EMPTY.glob("*.jpg"))[:limit]:
        res = blank.triage(Image.open(path).convert("RGB"))
        if res["verdict"] != "empty":
            continue
        if best is None or res["animal_confidence"] < best[0]:
            best = (res["animal_confidence"], path, res)
    return (best[1], best[2]) if best else None


def measure(blank: BlankFilter, reid: TigerReID, card: Path) -> dict:
    img = Image.open(card).convert("RGB")
    triage = blank.triage(img)
    out = {"verdict": triage["verdict"],
           "animal_confidence": triage["animal_confidence"]}
    if triage["verdict"] in ("animal", "review"):
        res = reid.identify(img)
        out["reid_decision"] = res["decision"]
        out["top3"] = [[t[0], round(t[1], 4)] for t in res["top3"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=40,
                    help="images to score per candidate pool")
    args = ap.parse_args()

    if not (ATRW / "reid_list_train.csv").exists():
        print(f"ATRW re-id train split missing under {ATRW}")
        return 1

    blank, reid = BlankFilter(), TigerReID()
    index = load_train_index()
    cards: list[dict] = []

    for tid in KNOWN_IDS:
        hit = pick_known(reid, index, tid, args.scan)
        if hit is None:
            print(f"no confident catalogue image for tiger {tid}")
            continue
        src, _ = hit
        dest = OUT / f"known_tiger_{tid}.jpg"
        as_card(src, dest)
        cards.append({"beat": "known_tiger", "tiger_id": tid,
                      "source": str(src.relative_to(ROOT)),
                      "card": str(dest.relative_to(ROOT)),
                      "expected": "animal -> known_identity"})

    hit = pick_empty(blank, args.scan)
    if hit is not None:
        dest = OUT / "empty_jungle.jpg"
        as_card(hit[0], dest)
        cards.append({"beat": "empty", "source": str(hit[0].relative_to(ROOT)),
                      "card": str(dest.relative_to(ROOT)),
                      "expected": "empty -> discarded"})

    hit = pick_unknown(reid, blank, args.scan)
    if hit is not None:
        dest = OUT / "unknown_tiger.jpg"
        as_card(hit[0], dest)
        cards.append({"beat": "unknown_tiger",
                      "source": str(hit[0].relative_to(ROOT)),
                      "card": str(dest.relative_to(ROOT)),
                      "expected": "animal -> unidentified / new tiger"})

    for card in cards:
        card["measured"] = measure(blank, reid, ROOT / card["card"])
        print(card["beat"], card.get("tiger_id", ""), card["measured"])

    manifest = OUT / "manifest.json"
    manifest.write_text(json.dumps(
        {"confirm_dist": TigerReID.CONFIRM_DIST,
         "blank_band": [blank.lo, blank.hi],
         "blank_model": getattr(blank, "version", "unknown"),
         "cards": cards}, indent=2))
    print("wrote", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
