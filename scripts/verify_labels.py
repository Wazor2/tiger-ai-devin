"""Verify ground-truth labels in COCO JSON for a sample from each bucket."""
from pathlib import Path
import json
import random
import ijson

DATASETS = Path("/home/ubuntu/project/datasets")
p = DATASETS / "cct_subset"

print("building id->cat map...")
labels_map = {}
with open(DATASETS / "cct" / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "annotations.item"):
        labels_map[obj["image_id"]] = obj["category_id"]
cats = {1: "opossum", 3: "raccoon", 5: "squirrel", 6: "bobcat", 7: "skunk",
        8: "dog", 9: "coyote", 10: "rabbit", 11: "bird", 14: "lizard",
        16: "cat", 21: "badger", 30: "empty", 33: "car", 34: "deer",
        37: "cow", 39: "pig", 40: "mountain_lion", 51: "fox", 66: "bat", 97: "insect", 99: "rodent"}

# image_id -> filename requires second pass; instead check files by filename hash
# CCT filenames are UUIDs; COCO 'file_name' field matches. Build id->fname map via
# streaming with a target set for efficiency.
random.seed(1)
targets = {}
for split in ["train", "val"]:
    for label in ["animal", "empty"]:
        d = p / split / label
        files = list(d.iterdir())
        for f in random.sample(files, min(10, len(files))):
            targets[f.name] = (split, label)

print("building fname->image_id map...")
fname2id = {}
with open(DATASETS / "cct" / "caltech_images_20210113.json", "rb") as f:
    for img in ijson.items(f, "images.item"):
        fn = img["file_name"]
        if fn in targets:
            fname2id[fn] = img["id"]
        if len(fname2id) == len(targets):
            break

for fn, (split, label) in sorted(targets.items()):
    img_id = fname2id.get(fn)
    if img_id is None:
        print(f"{fn}: NOT IN COCO JSON!")
        continue
    cat = labels_map.get(img_id, 30)
    gt = "animal" if cat != 30 else "empty"
    flag = "OK " if gt == label else "MISMATCH"
    print(f"[{flag}] {split}/{label}: {fn[:30]} -> gt={cats.get(cat, cat)}")
