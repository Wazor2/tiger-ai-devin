"""Rebuild the dataset with a distribution-matched validation split.

The official CCT location split isolates entire geographic locations for val,
which makes blank-vs-animal validation unrealistically hard (illumination,
vegetation shifts). We instead create a 'val' split held out from the training
locations (15% of train locations), keeping the official split available as
'test-loc' for robustness evaluation later.

Downloads holdout images the same way as before.
"""
import json
import random
from pathlib import Path

import ijson

DATASETS = Path("/home/ubuntu/project/datasets")
CCT = DATASETS / "cct"
SUB = DATASETS / "cct_subset"
AZURE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"

random.seed(7)

splits = json.load(open(DATASETS / "cct_splits.json"))["splits"]
all_train_locs = list(set(splits["train"]))

print("building maps...")
labels_map = {}
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "annotations.item"):
        labels_map[obj["image_id"]] = obj["category_id"]

imgs_by_loc = {}
with open(CCT / "caltech_images_20210113.json", "rb") as f:
    for img in ijson.items(f, "images.item"):
        imgs_by_loc.setdefault(img["location"], []).append(img)

# 15% of train locations become the local val split
random.shuffle(all_train_locs)
n_holdout = max(1, int(len(all_train_locs) * 0.15))
holdout_locs = set(all_train_locs[:n_holdout])
core_locs = set(all_train_locs[n_holdout:])
print(f"core train locs: {len(core_locs)}, holdout val locs: {len(holdout_locs)}")

# For robustness: also download a smaller official-val set from trans locations
official_val_locs = set(splits["val"])


def choose(locs, per_class):
    """Choose up to per_class images per bucket, sampled across locations."""
    pool = {"animal": [], "empty": []}
    for loc in locs:
        for img in imgs_by_loc.get(loc, []):
            bucket = "animal" if labels_map.get(img["id"], 30) != 30 else "empty"
            if len(pool[bucket]) < per_class:
                pool[bucket].append(img)
    for bucket in pool:
        random.shuffle(pool[bucket])
        pool[bucket] = pool[bucket][:per_class]
    return pool


def download(tasks, out_root):
    import httpx
    pending = [t for t in tasks]
    print(f"downloading {len(pending)} images into {out_root}...")
    client = httpx.Client(timeout=90, follow_redirects=True)
    ok = fail = 0
    for split, bucket, img in pending:
        dst = out_root / split / bucket / img["file_name"].split("/")[-1]
        if dst.exists() and dst.stat().st_size > 1000:
            ok += 1
            continue
        url = f"{AZURE}/{img['file_name']}"
        done = False
        for _ in range(2):
            try:
                r = client.get(url)
                if r.status_code == 200 and len(r.content) > 1000:
                    dst.write_bytes(r.content)
                    ok += 1
                    done = True
                    break
            except Exception:
                pass
        if not done:
            dst.unlink(missing_ok=True)
            fail += 1
        if (ok + fail) % 100 == 0:
            print(f"  {ok+fail}/{len(pending)} ok={ok} fail={fail}", flush=True)
    print(f"done ok={ok} fail={fail}")


core_chosen = choose(core_locs, 1200)
holdout_chosen = choose(holdout_locs, 200)
official_chosen = choose(official_val_locs, 100)

SUB.mkdir(parents=True, exist_ok=True)
train_root = SUB / "train"
holdout_root = SUB / "val_local"
off_root = SUB / "val_official"

for root, chosen in [(train_root, core_chosen), (holdout_root, holdout_chosen), (off_root, official_chosen)]:
    for bucket in ("animal", "empty"):
        (root / bucket).mkdir(parents=True, exist_ok=True)

def build_tasks(chosen):
    tasks = []
    for bucket, imgs in chosen.items():
        for img in imgs:
            tasks.append((".", bucket, img))
    return tasks

# Download each with correct target root
def dl_tasks(chosen, root):
    out = []
    for bucket, imgs in chosen.items():
        for img in imgs:
            out.append((root, bucket, img))
    return out

download(dl_tasks(core_chosen, train_root) + dl_tasks(holdout_chosen, holdout_root) + dl_tasks(official_chosen, off_root), SUB)

# Save mapping for later training use
meta = {
    "core_locs": sorted(core_locs),
    "holdout_locs": sorted(holdout_locs),
    "official_val_locs": sorted(official_val_locs),
    "train_counts": {k: len(v) for k, v in core_chosen.items()},
    "val_local_counts": {k: len(v) for k, v in holdout_chosen.items()},
    "val_official_counts": {k: len(v) for k, v in official_chosen.items()},
}
(SUB / "split_meta.json").write_text(json.dumps(meta, indent=2))
print("meta saved:", meta["train_counts"], meta["val_local_counts"], meta["val_official_counts"])
