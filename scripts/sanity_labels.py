"""Independent sanity: for ALL files in val dirs, check true label from COCO
JSON and count directory-vs-truth mismatches."""
from pathlib import Path
import ijson

DATASETS = Path("/home/ubuntu/project/datasets")
CCT = DATASETS / "cct_subset"

print("building fname2id and labels...")
fname2id = {}
labels_map = {}
with open(DATASETS / "cct" / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "images.item"):
        fname2id[obj["file_name"]] = obj["id"]
with open(DATASETS / "cct" / "caltech_images_20210113.json", "rb") as f:
    for obj in ijson.items(f, "annotations.item"):
        labels_map[obj["image_id"]] = obj["category_id"]

for split in ["train", "val"]:
    for label in ["animal", "empty"]:
        d = CCT / split / label
        files = list(d.iterdir())
        ok = bad = miss = 0
        for f in files:
            img_id = fname2id.get(f.name)
            if img_id is None:
                miss += 1
                continue
            true_cat = labels_map.get(img_id, 30)
            true_label = "animal" if true_cat != 30 else "empty"
            if true_label == label:
                ok += 1
            else:
                bad += 1
        print(f"{split}/{label}: n={len(files)} label_ok={ok} misplaced={bad} missing_in_json={miss}")
