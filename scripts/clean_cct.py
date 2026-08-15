"""Remove empty/truncated/non-JPEG files from the CCT subset."""
from pathlib import Path
from PIL import Image

p = Path("/home/ubuntu/project/datasets/cct_subset")
bad = 0
for f in p.rglob("*"):
    if f.is_file():
        if f.stat().st_size < 500:
            f.unlink(); bad += 1; continue
        try:
            Image.open(f).verify()
        except Exception:
            f.unlink(); bad += 1

print("removed", bad, "bad files")
