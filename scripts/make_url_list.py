"""Generate wget command list for the chosen CCT subset (idempotent)."""
import random
from pathlib import Path

import ijson

DATASETS = Path("/home/ubuntu/project/datasets")
CCT = DATASETS / "cct"
OUT = DATASETS / "cct_subset"
AZURE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"

random.seed(42)

import json
splits = json.load(open(DATASETS / "cct_splits.json"))["splits"]
train_locs = set(splits["train"])
val_locs = set(splits["val"])

labels_map = {}
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "annotations.item"):
        labels_map[obj["image_id"]] = obj["category_id"]

imgs_by_loc = {}
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for img in ijson.items(f, "images.item"):
        imgs_by_loc.setdefault(img["location"], []).append(img)


def pick(subset_locs, per_class):
    chosen = {"animal": [], "empty": []}
    for loc in subset_locs:
        for img in imgs_by_loc.get(loc, []):
            bucket = "animal" if labels_map.get(img["id"], 30) != 30 else "empty"
            if len(chosen[bucket]) < per_class:
                chosen[bucket].append(img)
    return chosen


tasks = []
for split, per in [("train", 1200), ("val", 300)]:
    chosen = pick(train_locs if split == "train" else val_locs, per)
    for bucket, imgs in chosen.items():
        (OUT / split / bucket).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            dst = OUT / split / bucket / img["file_name"].split("/")[-1]
            if dst.exists() and dst.stat().st_size > 1000:
                continue
            tasks.append(f"{AZURE}/{img['file_name']} -O {dst}")

(DATASETS / "url_list.txt").write_text("\n".join(tasks) + "\n")
print(len(tasks), "pending downloads")
