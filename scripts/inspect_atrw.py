"""Inspect ATRW Re-ID annotation structure."""
import json
import csv
from pathlib import Path

BASE = Path("/home/ubuntu/project/datasets/atrw")

with open(BASE / "reid_list_train.csv") as f:
    rows = list(csv.reader(f))
ids = [r[0] for r in rows]
print("train CSV rows:", len(rows), "unique ids:", len(set(ids)))
print("sample:", rows[:3])

with open(BASE / "reid_keypoints_train.json") as f:
    d = json.load(f)
print("train keypoints top-level keys:", list(d.keys()))
for k, v in d.items():
    print(f"  {k}: type={type(v).__name__}", end="")
    if isinstance(v, list):
        print(f" len={len(v)}")
        print("  first:", json.dumps(v[0])[:300])
    elif isinstance(v, dict):
        print(f" keys(first): {list(v.keys())[:5]}")
    else:
        print(f" value={str(v)[:200]}")

# test set
with open(BASE / "reid_list_test.csv") as f:
    trows = list(csv.reader(f))
print("\ntest CSV rows:", len(trows))

with open(BASE / "reid_keypoints_test.json") as f:
    td = json.load(f)
print("test keypoints top-level keys:", list(td.keys()))
for k, v in td.items():
    print(f"  {k}: type={type(v).__name__}", end="")
    if isinstance(v, list):
        print(f" len={len(v)}")
    elif isinstance(v, dict):
        print(f" dict w/ {len(v)} entries")
    else:
        print()

# README
print("\nREADME:")
print((BASE / "README.md").read_text()[:1500])
