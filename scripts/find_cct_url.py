"""Probe possible blob URL prefixes for CCT images (GCP/Azure mirrors)."""
import subprocess
import sys

fn = "59a49a65-23d2-11e8-a6a3-ec086b02610b.jpg"

candidates = [
    f"https://lilawildlife.blob.core.windows.net/lila-wildlife/cct_images/{fn}",
    f"https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/{fn}",
    f"https://lilawildlife.blob.core.windows.net/lila-wildlife/caltechcameratraps/cct_images/{fn}",
    f"https://storage.googleapis.com/public-datasets-lila/cct_images/{fn}",
    f"https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/{fn}",
    f"http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/cct_images/{fn}",
    f"http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/caltech-unzipped/cct_images/{fn}",
]

for url in candidates:
    r = subprocess.run(
        ["curl", "-sIL", "--max-time", "20", url],
        capture_output=True, text=True)
    lines = r.stdout.strip().split("\n")
    status = [l for l in lines if l.startswith("HTTP/")]
    last = status[-1] if status else "NO RESPONSE"
    print(last, "->", url)
