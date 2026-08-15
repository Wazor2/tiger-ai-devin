"""Rebuild the Caltech Camera Traps subset used to train/validate the blank filter.

The shipped repo has no `datasets/` folder, so the blank-filter thresholds cannot
be re-swept without pulling the data back down. This script reconstructs a
distribution-matched subset with the same shape as the one referenced by
`scripts/train_blank_v2.py` (train 1200/class, val 300/class) from the public
LILA mirror.

Layout produced:
    datasets/cct_subset/train/{animal,empty}
    datasets/cct_subset/val/{animal,empty}      (val_local 200 + val_official 100 per class)
    datasets/cct_subset/split_meta.json

Note: the original split used `random.shuffle(list(set(locations)))`, whose input
order depends on PYTHONHASHSEED, so the image list here is distribution-matched
but not byte-identical to the split the shipped checkpoint was validated on.

Usage:
    python -m scripts.fetch_cct_subset [--val-only] [--workers 16]
"""
import argparse
import io
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT / "datasets"
CCT_META = DATASETS / "cct_meta"
SUB = DATASETS / "cct_subset"

LILA = "https://lilawildlife.blob.core.windows.net/lila-wildlife"
LABELS_ZIP = f"{LILA}/caltechcameratraps/labels/caltech_camera_traps.json.zip"
SPLITS_URL = f"{LILA}/caltechcameratraps/CaltechCameraTrapsSplits_v0.json"
IMAGES_BASE = f"{LILA}/caltech-unzipped/cct_images"

EMPTY_CATEGORY_ID = 30
SEED = 7


def _get(url: str) -> bytes:
    with urlopen(url, timeout=120) as r:
        return r.read()


def fetch_metadata() -> tuple:
    CCT_META.mkdir(parents=True, exist_ok=True)
    meta_path = CCT_META / "caltech_camera_traps.json"
    splits_path = CCT_META / "CaltechCameraTrapsSplits_v0.json"
    if not meta_path.exists():
        print("downloading label metadata (~9 MB zipped)...", flush=True)
        with zipfile.ZipFile(io.BytesIO(_get(LABELS_ZIP))) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            meta_path.write_bytes(z.read(name))
    if not splits_path.exists():
        print("downloading location splits...", flush=True)
        splits_path.write_bytes(_get(SPLITS_URL))
    return json.loads(meta_path.read_text()), json.loads(splits_path.read_text())


def choose(imgs_by_loc, labels_map, locs, per_class):
    """Sample `per_class` animal and empty images, matched by location.

    Both classes are drawn round-robin from the same locations, and only from
    locations that contain both classes. Filling one class from one set of
    locations and the other class from a different set lets a classifier win on
    background cues alone, which is how the shipped v2 checkpoint ended up
    below chance on a location-disjoint validation split.
    """
    per_loc = {}
    for loc in sorted(locs, key=str):
        buckets = {"animal": [], "empty": []}
        for img in imgs_by_loc.get(loc, []):
            cat = labels_map.get(img["id"])
            if cat is None:
                continue        # unannotated image: label unknown, skip
            buckets["empty" if cat == EMPTY_CATEGORY_ID else "animal"].append(img)
        if buckets["animal"] and buckets["empty"]:
            rng = random.Random(f"{SEED}:{loc}")
            for b in buckets:
                rng.shuffle(buckets[b])
            per_loc[loc] = buckets

    order = sorted(per_loc)
    random.Random(SEED).shuffle(order)
    pool = {"animal": [], "empty": []}
    progress = True
    while progress and (len(pool["animal"]) < per_class or
                        len(pool["empty"]) < per_class):
        progress = False
        for loc in order:
            for bucket in ("animal", "empty"):
                if len(pool[bucket]) < per_class and per_loc[loc][bucket]:
                    pool[bucket].append(per_loc[loc][bucket].pop())
                    progress = True
    return pool


def download_all(tasks, workers):
    done = {"ok": 0, "skip": 0, "fail": 0}

    def one(task):
        dst, file_name = task
        if dst.exists() and dst.stat().st_size > 1000:
            done["skip"] += 1
            return
        for _ in range(3):
            try:
                data = _get(f"{IMAGES_BASE}/{file_name}")
                if len(data) > 1000:
                    dst.write_bytes(data)
                    done["ok"] += 1
                    return
            except Exception:
                pass
        done["fail"] += 1

    print(f"downloading {len(tasks)} images with {workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, tasks), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} {done}", flush=True)
    print("download finished:", done, flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-only", action="store_true",
                    help="skip the 2400-image training split")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    meta, splits_doc = fetch_metadata()
    splits = splits_doc["splits"] if "splits" in splits_doc else splits_doc

    labels_map = {a["image_id"]: a["category_id"] for a in meta["annotations"]}
    imgs_by_loc = {}
    for img in meta["images"]:
        imgs_by_loc.setdefault(img["location"], []).append(img)
    print(f"metadata: {len(meta['images'])} images, {len(imgs_by_loc)} locations",
          flush=True)

    train_locs = sorted({str(x) for x in splits["train"]})
    official_val_locs = sorted({str(x) for x in splits["val"]})
    imgs_by_loc = {str(k): v for k, v in imgs_by_loc.items()}

    rng = random.Random(SEED)
    shuffled = list(train_locs)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * 0.15))
    holdout_locs, core_locs = set(shuffled[:n_holdout]), set(shuffled[n_holdout:])
    print(f"core train locs={len(core_locs)} holdout val locs={len(holdout_locs)} "
          f"official val locs={len(official_val_locs)}", flush=True)

    core = choose(imgs_by_loc, labels_map, core_locs, 1200)
    holdout = choose(imgs_by_loc, labels_map, holdout_locs, 200)
    official = choose(imgs_by_loc, labels_map, official_val_locs, 100)

    train_root, val_root = SUB / "train", SUB / "val"
    for root in (train_root, val_root):
        for bucket in ("animal", "empty"):
            (root / bucket).mkdir(parents=True, exist_ok=True)

    tasks = []
    if not args.val_only:
        for bucket, imgs in core.items():
            for img in imgs:
                tasks.append((train_root / bucket / img["file_name"].split("/")[-1],
                              img["file_name"]))
    for chosen in (holdout, official):
        for bucket, imgs in chosen.items():
            for img in imgs:
                tasks.append((val_root / bucket / img["file_name"].split("/")[-1],
                              img["file_name"]))

    stats = download_all(tasks, args.workers)

    (SUB / "split_meta.json").write_text(json.dumps({
        "seed": SEED,
        "core_locs": sorted(core_locs),
        "holdout_val_locs": sorted(holdout_locs),
        "official_val_locs": official_val_locs,
        "train_counts": {k: len(v) for k, v in core.items()},
        "val_local_counts": {k: len(v) for k, v in holdout.items()},
        "val_official_counts": {k: len(v) for k, v in official.items()},
        "download": stats,
        "note": "distribution-matched rebuild; not byte-identical to the "
                "original split (PYTHONHASHSEED-dependent shuffle)",
    }, indent=2))
    for root in (train_root, val_root):
        counts = {b: len(list((root / b).glob("*.jpg"))) for b in ("animal", "empty")}
        print(root.name, counts, flush=True)


if __name__ == "__main__":
    sys.exit(main())
