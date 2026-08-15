"""Honest Re-ID benchmark — strict gallery/probe split (fix-spec item 6, first bullet).

The existing benchmark (scripts/bench_reid.py, models/reid_benchmark.json) scores
every image against a catalogue whose centroids were computed from that same
image, so Rank-1 76.4% / open-set ROC-AUC 0.820 are optimistic by construction.

This script splits each identity's images into a gallery half (centroids only)
and a probe half (queries only), so a query never contributes to the centroid it
is scored against, and holds whole identities out of the catalogue to measure
open-set rejection. Results land in models/reid_benchmark_strict.json.

    python scripts/bench_reid_strict.py [--dataset datasets/atrw_dl] [--limit N]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from pench.model_serving import MODELS, TigerReID

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT = Path(__file__).resolve().parent.parent
SEED = 17
MIN_IMAGES = 4          # identities below this can't be split gallery/probe
UNKNOWN_FRACTION = 0.25  # share of eligible identities held fully out of catalogue


def load_labels(dataset: Path) -> dict:
    """filename -> identity, restricted to files that actually exist."""
    out = {}
    with open(dataset / "reid_list_train.csv", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) != 2:
                continue
            tid, fname = row[0].strip(), row[1].strip()
            path = dataset / "train" / fname
            if tid.isdigit() and path.exists():
                out[path] = tid
    return out


def split_identities(labels: dict) -> tuple:
    """(known: id -> (gallery, probe)), (unknown: id -> probes)."""
    by_id = defaultdict(list)
    for path, tid in labels.items():
        by_id[tid].append(path)
    eligible = sorted(tid for tid, ps in by_id.items() if len(ps) >= MIN_IMAGES)
    rng = np.random.default_rng(SEED)
    n_unknown = max(1, int(len(eligible) * UNKNOWN_FRACTION))
    unknown_ids = set(rng.choice(eligible, size=n_unknown, replace=False))

    known, unknown = {}, {}
    for tid in eligible:
        paths = sorted(by_id[tid])
        rng.shuffle(paths)
        if tid in unknown_ids:
            unknown[tid] = paths
        else:
            half = len(paths) // 2
            known[tid] = (paths[:half], paths[half:])
    return known, unknown


@torch.no_grad()
def embed_all(reid: TigerReID, paths: list, batch: int = 32) -> np.ndarray:
    vecs = []
    for i in range(0, len(paths), batch):
        xs = torch.stack([reid.transform(Image.open(p).convert("RGB"))
                          for p in paths[i:i + batch]]).to(reid.device)
        vecs.append(F.normalize(reid.model(xs), dim=1).cpu().numpy())
    return np.concatenate(vecs).astype("float32")


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PROJECT / "datasets" / "atrw_dl"))
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per identity (smoke runs)")
    ap.add_argument("--out", default=str(MODELS / "reid_benchmark_strict.json"))
    ap.add_argument("--ckpt", default=str(MODELS / "tiger_reid.pth"),
                    help="checkpoint to score (candidates are not promoted here)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    labels = load_labels(dataset)
    if not labels:
        print(f"no labelled images under {dataset}")
        return 1
    known, unknown = split_identities(labels)
    if args.limit:
        known = {t: (g[:args.limit], p[:args.limit]) for t, (g, p) in known.items()}
        unknown = {t: p[:args.limit] for t, p in unknown.items()}

    reid = TigerReID(ckpt_path=Path(args.ckpt))
    print(f"ckpt={Path(args.ckpt).name} device={reid.device} identities: {len(known)} catalogue / "
          f"{len(unknown)} held-out", flush=True)

    # --- catalogue from gallery halves only ---------------------------------
    cat_ids, cat_vecs = [], []
    for tid, (gallery, _) in sorted(known.items()):
        cat_ids.append(tid)
        cat_vecs.append(embed_all(reid, gallery).mean(axis=0))
    cat = unit(np.array(cat_vecs, dtype="float32"))

    # --- known probes -------------------------------------------------------
    probe_paths, probe_ids = [], []
    for tid, (_, probes) in sorted(known.items()):
        probe_paths += probes
        probe_ids += [tid] * len(probes)
    probe_vecs = unit(embed_all(reid, probe_paths))
    probe_ids = np.array(probe_ids)
    print(f"embedded {len(probe_paths)} known probes", flush=True)

    dist = 1 - probe_vecs @ cat.T             # (n_probes, n_identities)
    order = dist.argsort(axis=1)
    ranked = np.array(cat_ids)[order]
    hit = ranked == probe_ids[:, None]
    rank1 = float(hit[:, 0].mean())
    rank5 = float(hit[:, :5].any(axis=1).mean())
    rank10 = float(hit[:, :10].any(axis=1).mean())
    # one relevant centroid per query, so AP == 1/rank of the correct identity
    ap_ = 1.0 / (hit.argmax(axis=1) + 1)
    m_ap = float(np.where(hit.any(axis=1), ap_, 0.0).mean())

    known_min = dist.min(axis=1)
    per_id = defaultdict(list)
    for tid, ok in zip(probe_ids, hit[:, 0], strict=True):
        per_id[tid].append(bool(ok))

    # --- unknown probes (identities absent from the catalogue) --------------
    unk_paths = [p for ps in unknown.values() for p in ps]
    unk_vecs = unit(embed_all(reid, unk_paths))
    unknown_min = (1 - unk_vecs @ cat.T).min(axis=1)
    print(f"embedded {len(unk_paths)} unknown probes", flush=True)

    confirm = TigerReID.CONFIRM_DIST
    scores = np.concatenate([known_min, unknown_min])
    truth = np.concatenate([np.zeros(len(known_min)), np.ones(len(unknown_min))])
    idx = np.argsort(-scores)
    y = truth[idx]
    n_u, n_k = int(y.sum()), int(len(y) - y.sum())
    roc_auc = float(np.trapezoid(np.cumsum(y) / max(1, n_u),
                                 np.cumsum(1 - y) / max(1, n_k)))

    sweep = []
    for t in np.linspace(float(scores.min()), float(scores.max()), 201):
        acc = float((known_min < t).mean())
        rej = float((unknown_min >= t).mean())
        sweep.append({"t": round(float(t), 4), "known_acceptance": round(acc, 4),
                      "unknown_rejection": round(rej, 4),
                      "youden": round(acc + rej - 1, 4)})
    best = max(sweep, key=lambda r: r["youden"])

    counts = Counter(labels.values())
    result = {
        "protocol": {
            "note": ("gallery/probe disjoint per identity; queries never "
                     "contribute to the centroid they are scored against"),
            "seed": SEED, "min_images_per_identity": MIN_IMAGES,
            "unknown_fraction": UNKNOWN_FRACTION,
            "catalogue_identities": len(cat_ids),
            "held_out_identities": sorted(unknown),
            "n_known_probes": len(probe_paths), "n_unknown_probes": len(unk_paths),
            "checkpoint": Path(args.ckpt).name,
            "dataset": str(dataset),
        },
        "closed_set": {
            "rank1": round(rank1, 4), "rank5": round(rank5, 4),
            "rank10": round(rank10, 4), "map": round(m_ap, 4),
            "per_identity_rank1": {t: round(float(np.mean(v)), 3)
                                   for t, v in sorted(per_id.items())},
            "weak_identities_lt_10_images": sorted(
                t for t in per_id if counts[t] < 10),
        },
        "open_set": {
            "confirm_dist": confirm,
            "known_acceptance_at_confirm": round(float((known_min < confirm).mean()), 4),
            "unknown_rejection_at_confirm": round(float((unknown_min >= confirm).mean()), 4),
            "roc_auc": round(roc_auc, 4),
            "known_min_distance": {"mean": round(float(known_min.mean()), 4),
                                   "p95": round(float(np.percentile(known_min, 95)), 4)},
            "unknown_min_distance": {"mean": round(float(unknown_min.mean()), 4),
                                     "p05": round(float(np.percentile(unknown_min, 5)), 4)},
            "best_threshold": best,
            "sweep": sweep,
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSTRICT closed-set : rank1={rank1:.4f} rank5={rank5:.4f} mAP={m_ap:.4f}")
    print(f"STRICT open-set   : roc_auc={roc_auc:.4f} "
          f"known_acc@{confirm}={result['open_set']['known_acceptance_at_confirm']} "
          f"unk_rej@{confirm}={result['open_set']['unknown_rejection_at_confirm']}")
    print(f"best threshold    : t={best['t']} acc={best['known_acceptance']} "
          f"rej={best['unknown_rejection']}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
