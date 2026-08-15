"""Prepare a balanced subset of Caltech Camera Traps for blank-image training.

v3: streaming JSON parse + httpx downloads with retries.
"""
import random
import subprocess
from pathlib import Path

import httpx
import ijson
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASETS = Path("/home/ubuntu/project/datasets")
CCT = DATASETS / "cct"
OUT = DATASETS / "cct_subset"
AZURE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"

random.seed(42)

splits = __import__("json").load(open(DATASETS / "cct_splits.json"))["splits"]
train_locs = set(splits["train"])
val_locs = set(splits["val"])

print("parsing labels...", flush=True)
labels_map = {}
cats = {}
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "annotations.item"):
        labels_map[obj["image_id"]] = obj["category_id"]
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "categories.item"):
        cats[obj["id"]] = obj["name"]

print("parsing images...", flush=True)
imgs_by_loc = {}
count = 0
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for img in ijson.items(f, "images.item"):
        count += 1
        imgs_by_loc.setdefault(img["location"], []).append(img)
print("parsed", count, "images")


def pick(subset_locs, per_class):
    chosen = {"animal": [], "empty": []}
    for loc in subset_locs:
        for img in imgs_by_loc.get(loc, []):
            bucket = "animal" if labels_map.get(img["id"], 30) != 30 else "empty"
            if len(chosen[bucket]) < per_class:
                chosen[bucket].append(img)
    return chosen


train_chosen = pick(train_locs, 1200)
val_chosen = pick(val_locs, 300)
for k in train_chosen:
    print(f"train {k}: {len(train_chosen[k])}")
for k in val_chosen:
    print(f"val {k}: {len(val_chosen[k])}")

for split, chosen in [("train", train_chosen), ("val", val_chosen)]:
    for bucket in ("animal", "empty"):
        (OUT / split / bucket).mkdir(parents=True, exist_ok=True)

# (split, bucket, img) task list
tasks = []
for split, chosen in [("train", train_chosen), ("val", val_chosen)]:
    for bucket, imgs in chosen.items():
        for img in imgs:
            tasks.append((split, bucket, img))
print("total downloads:", len(tasks))

client = httpx.Client(timeout=90, follow_redirects=True)


def download(args):
    split, bucket, img = args
    dst = OUT / split / bucket / img["file_name"].split("/")[-1]
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    url = f"{AZURE}/{img['file_name']}"
    for _ in range(3):
        try:
            r = client.get(url)
            if r.status_code == 200 and len(r.content) > 1000:
                dst.write_bytes(r.content)
                return True
        except Exception:
            pass
    dst.unlink(missing_ok=True)
    return False


ok = fail = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(download, t): t for t in tasks}
    for fut in as_completed(futs):
        if fut.result():
            ok += 1
        else:
            fail += 1
        if (ok + fail) % 250 == 0:
            print(f"  {ok+fail}/{len(tasks)} ok={ok} fail={fail}", flush=True)
print(f"finished ok={ok} fail={fail}")
