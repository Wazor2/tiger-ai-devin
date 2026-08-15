"""Check whether the animal/empty split actually matches image content."""
from pathlib import Path
from PIL import Image
import numpy as np

p = Path("/home/ubuntu/project/datasets/cct_subset")
for split in ["train", "val"]:
    for label in ["animal", "empty"]:
        d = p / split / label
        files = sorted(d.iterdir())
        print(f"== {split}/{label} ({len(files)} files)")
        for f in files[:6]:
            im = np.asarray(Image.open(f).convert("L").resize((64, 64)))
            print(f"  {f.name[:24]} std={im.std():5.1f} mean={im.mean():5.1f}")
