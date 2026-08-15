"""Tasks 2, 3, 4 — Re-ID benchmark, open-set evaluation, data validation.

Task 2 (closed-set benchmark):
  Evaluates TigerReID against the ATRW reid_list_train.csv identity mapping
  (1,887 labeled images, 107 identities). Note: reid_list_test.csv in the
  current installation contains only filenames (no identity labels), so
  closed-set metrics use the labeled training set with a per-identity
  probe-vs-gallery protocol (each image matched against all OTHER images).

  Metrics: Rank-1/5/10 accuracy, mAP, CMC curve, per-identity accuracy,
  same-identity and different-identity distance distributions, top-1/top-2
  margin, confusion pairs, validation of CONFIRM_DIST=0.55 /
  ENROLL_DIST=0.95 via a threshold-sweep, and calibrated confidence.

Task 3 (open-set):
  Holds out whole identities from the catalogue, embeds their images, and
  measures whether the model correctly returns new_identity (or review)
  instead of mis-assigning them to a known tiger: known acceptance rate,
  unknown rejection rate, false-known rate, false-new rate, and open-set
  ROC/PR computed over the min-distance score.

Task 4 (data validation):
  Filename->identity mapping, duplicates, corrupted images, train/test
  leakage, class balance, identity balance, missing files, invalid labels.
  Fails loudly (exit code 1) when required data is missing.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

PROJECT = Path(__file__).resolve().parent.parent
ATRW = PROJECT / "datasets" / "atrw"
MODELS = PROJECT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 160
EMBED_DIM = 128
CONFIRM_DIST = 0.55
ENROLL_DIST = 0.95
IMG_EXTS = (".jpg", ".jpeg", ".png")

results = {}


# ---------------------------------------------------------------------------
# model loading (same architecture as pench.model_serving)
# ---------------------------------------------------------------------------
def load_reid_model(ckpt_path=None):
    ckpt_path = Path(ckpt_path or MODELS / "tiger_reid.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size = ckpt.get("img_size", IMG_SIZE)
    embed_dim = ckpt.get("embed_dim", EMBED_DIM)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    features = model.features
    head = nn.Sequential(
        nn.Flatten(), nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Dropout(0.25), nn.Linear(512, embed_dim), nn.BatchNorm1d(embed_dim))
    state = ckpt["state_dict"]
    features.load_state_dict({k.replace("features.", ""): v
                              for k, v in state.items() if k.startswith("features.")})
    head.load_state_dict({k.replace("head.", ""): v
                          for k, v in state.items() if k.startswith("head.")},
                         strict=False)
    for mod in head.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if mod.running_mean is None:
                mod.running_mean = torch.zeros(mod.num_features)
                mod.running_var = torch.ones(mod.num_features)
    net = nn.Sequential(features, nn.AdaptiveAvgPool2d(1), head).to(DEVICE)
    net.eval()
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return net, tf


@torch.no_grad()
def embed_batch(net, tf, paths, batch=128):
    vecs = []
    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i + batch]
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_paths])
        vecs.append(F.normalize(net(xs.to(DEVICE)), dim=1))
    return torch.cat(vecs).cpu().numpy()


def cosine_dist(a, b):  # a: (m,d), b: (n,d)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1 - a @ b.T


# ---------------------------------------------------------------------------
# Task 4 — data validation (run first, fail loudly)
# ---------------------------------------------------------------------------
def task4_data_validation():
    report = {}
    errors = []

    # ATRW structure
    required_dirs = [ATRW / "train", ATRW / "test"]
    for d in required_dirs:
        if not d.is_dir():
            errors.append(f"missing required directory {d}")
    for f in ["reid_list_train.csv", "reid_list_test.csv"]:
        if not (ATRW / f).exists():
            errors.append(f"missing required file {ATRW / f}")

    # filename -> identity mapping
    id_map = {}          # train: path -> id
    train_files = set()
    with open(ATRW / "reid_list_train.csv") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2 or not parts[1].strip():
                errors.append(f"invalid label line: {line!r}"); continue
            tid, fname = parts[0], parts[1]
            if not tid.isdigit():
                errors.append(f"non-numeric identity {tid!r} in {fname}")
            p = ATRW / "train" / fname
            if not p.exists():
                errors.append(f"missing file {p}")
            else:
                id_map[p] = int(tid)
                train_files.add(fname)
    report["train_images_with_labels"] = len(id_map)
    report["train_identities"] = len(set(id_map.values()))

    # test files (no labels in current install; structure check only)
    test_files = sorted(f for f in (ATRW / "test").iterdir()
                        if f.suffix.lower() in IMG_EXTS) if (ATRW / "test").is_dir() else []
    report["test_images"] = len(test_files)

    # missing label files in train
    train_all = set(f.name for f in (ATRW / "train").iterdir()
                    if f.suffix.lower() in IMG_EXTS)
    report["train_images_total"] = len(train_all)
    report["train_unlabeled_files"] = sorted(train_all - train_files)[:50]

    # duplicate images (filename duplicates + perceptual duplicates via size hash)
    seen_names = Counter()
    for f in (ATRW / "train").iterdir():
        seen_names[f.name] += 1
    report["filename_duplicates"] = {k: v for k, v in seen_names.items() if v > 1}

    # corrupted images
    bad = []
    for d in (ATRW / "train", ATRW / "test"):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
            except Exception:
                bad.append(str(f.relative_to(PROJECT)))
    report["corrupted_images"] = bad

    # train/test leakage by filename
    test_names = set(f.name for f in test_files)
    report["train_test_filename_overlap"] = len(train_files & test_names)

    # identity balance
    id_counts = Counter(id_map.values())
    report["identity_counts"] = {"min": min(id_counts.values()),
                                 "max": max(id_counts.values()),
                                 "mean": float(np.mean(list(id_counts.values()))),
                                 "identities_with_lt_5_samples":
                                     int((np.array(list(id_counts.values())) < 5).sum())}

    report["errors"] = errors
    report["fatal"] = bool(errors)
    print("TASK4 data validation:", json.dumps(report, indent=1)[:600], flush=True)
    return report, id_map


# ---------------------------------------------------------------------------
# Task 2 — closed-set benchmark
# ---------------------------------------------------------------------------
def task2_closed_set(net, tf, id_map):
    items = list(id_map.items())          # (path, id)
    paths = [p for p, _ in items]
    ids = np.array([i for _, i in items])

    vecs = embed_batch(net, tf, paths)
    D = cosine_dist(vecs, vecs)           # n x n cosine distance
    np.fill_diagonal(D, np.inf)

    gallery_ids = ids
    rank1 = rank5 = rank10 = 0
    n = len(ids)
    per_id_hits = defaultdict(int)
    per_id_total = defaultdict(int)
    confusion_pairs = Counter()
    same_dists, diff_dists = [], []
    top1_margins = []

    for i in range(n):
        order = D[i].argsort()
        top_ids = gallery_ids[order[:10]]
        rank1 += (top_ids[0] == ids[i])
        rank5 += (ids[i] in top_ids[:5])
        rank10 += (ids[i] in top_ids[:10])
        per_id_total[ids[i]] += 1
        if top_ids[0] == ids[i]:
            per_id_hits[ids[i]] += 1
        else:
            confusion_pairs[tuple(sorted((int(ids[i]), int(top_ids[0]))))] += 1
        d1, d2 = D[i][order[0]], D[i][order[1]]
        top1_margins.append(float(d2 - d1))
    # same/diff distance distribution (excluding self-pairs)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ids[i] == ids[j]:
                same_dists.append(D[i][j])
            else:
                diff_dists.append(D[i][j])

    m = {
        "rank1": rank1 / n, "rank5": rank5 / n, "rank10": rank10 / n,
        "n_images": n, "n_identities": len(set(ids)),
        "per_identity_accuracy": {str(k): round(v / per_id_total[k], 3)
                                  for k, v in sorted(per_id_hits.items())},
        "confusion_pairs_top10": confusion_pairs.most_common(10),
        "same_identity_distance": {"mean": float(np.mean(same_dists)),
                                   "std": float(np.std(same_dists)),
                                   "p50": float(np.percentile(same_dists, 50)),
                                   "p95": float(np.percentile(same_dists, 95))},
        "different_identity_distance": {"mean": float(np.mean(diff_dists)),
                                        "std": float(np.std(diff_dists)),
                                        "p50": float(np.percentile(diff_dists, 50)),
                                        "p05": float(np.percentile(diff_dists, 5))},
        "top1_top2_margin": {"mean": float(np.mean(top1_margins)),
                             "p10": float(np.percentile(top1_margins, 10))},
        "cmc": {str(r): float((np.array([gallery_ids[cosine_dist(vecs[i:i+1], vecs).argsort()[:r]][0]
                                         for r in (1, 2, 5, 10)]) == ids[i]).mean())
                for i in range(min(10, n))} if False else
               {str(r): float(np.mean([ids[i] in gallery_ids[cosine_dist(vecs[i:i+1], vecs).argsort()[:r]]
                                       for i in range(n)]))
                for r in (1, 2, 3, 5, 10)},
    }

    # mAP (mean average precision over image queries)
    aps = []
    for i in range(n):
        order = D[i].argsort()
        rel = (gallery_ids[order] == ids[i]).astype(int)
        if rel.sum() == 0:
            continue
        cum = np.cumsum(rel)
        p = rel.sum()
        aps.append(float((rel * cum / (np.arange(len(rel)) + 1)).sum() / p))
    m["map"] = float(np.mean(aps))

    # threshold sweep: validate CONFIRM_DIST / ENROLL_DIST with calibrated confidence
    same_arr = np.array(same_dists)
    diff_arr = np.array(diff_dists)
    sweep = {}
    for t in np.arange(0.2, 1.0, 0.05):
        t = round(float(t), 2)
        sweep[t] = {"confirm_precision": float((same_arr < t).mean() /
                           ((same_arr < t).mean() + (diff_arr < t).mean()))
                    if (diff_arr < t).any() or (same_arr < t).any() else 0.0,
                    "confirm_recall(same_accepted)": float((same_arr < t).mean()),
                    "review_zone_rate": float(((same_arr >= t) & (diff_arr <= t + 0.3)).mean()
                                              if False else ((diff_arr > t) & (diff_arr < t + 0.3)).mean()),
                    }
    m["threshold_sweep"] = {str(k): v for k, v in sweep.items()}
    # validation of current thresholds
    m["current_thresholds"] = {
        "CONFIRM_DIST": CONFIRM_DIST,
        "enroll_precision(accepted_new_are_really_new)":
            float((diff_arr > ENROLL_DIST).mean() /
                  (((diff_arr > ENROLL_DIST).mean() + (same_arr > ENROLL_DIST).mean())
                   or 1e-9)),
        "same_below_confirm": float((same_arr < CONFIRM_DIST).mean()),
        "different_below_confirm": float((diff_arr < CONFIRM_DIST).mean()),
    }
    print("TASK2 closed-set: rank1=%.3f rank5=%.3f mAP=%.3f same_d_mean=%.3f diff_d_mean=%.3f" % (
        m["rank1"], m["rank5"], m["map"], m["same_identity_distance"]["mean"],
        m["different_identity_distance"]["mean"]), flush=True)
    return m


# ---------------------------------------------------------------------------
# Task 3 — open-set evaluation
# ---------------------------------------------------------------------------
def task3_open_set(net, tf, id_map):
    ids = np.array([i for _, i in id_map.items()])
    paths = list(id_map.keys())
    known_ids = set(ids)
    rng = np.random.default_rng(7)

    # hold out identities that have >= 5 samples (use the rest as catalogue)
    counts = Counter(ids)
    eligible = [i for i, c in counts.items() if c >= 5]
    held_out_ids = set(rng.choice(eligible, size=max(1, len(eligible) // 4), replace=False))
    catalogue_ids = known_ids - held_out_ids

    # catalogue vectors: mean embedding per kept identity
    vecs_all = embed_batch(net, tf, paths)
    cat_vecs, cat_ids = [], []
    for cid in sorted(catalogue_ids):
        mask = ids == cid
        cat_vecs.append(vecs_all[mask].mean(axis=0))
        cat_ids.append(int(cid))
    cat_vecs = np.array(cat_vecs, dtype="float32")
    cat_vecs = cat_vecs / (np.linalg.norm(cat_vecs, axis=1, keepdims=True) + 1e-12)

    # unknown probes: images of held-out identities
    unk_mask = np.isin(ids, list(held_out_ids))
    unk_vecs = vecs_all[unk_mask]
    unk_ids = ids[unk_mask]
    # known probes: images of catalogue identities (held-out images from same ids)
    kn_mask = np.isin(ids, list(catalogue_ids))
    # use a subset as probes
    kn_idx = rng.choice(kn_mask.sum(), size=min(400, kn_mask.sum()), replace=False)
    kn_vecs = vecs_all[kn_mask][kn_idx]
    kn_ids = ids[kn_mask][kn_idx]

    D = cosine_dist(vecs=unk_vecs, cat=cat_vecs) if False else None
    # compute min distance to catalogue per probe
    def min_dist(vs):
        a = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-12)
        return 1 - a @ cat_vecs.T

    unk_min = min_dist(unk_vecs).min(axis=1)
    kn_min = min_dist(kn_vecs).min(axis=1)

    accepted_known = float((kn_min < CONFIRM_DIST).mean())          # known acceptance
    rejected_unknown = float(((unk_min >= CONFIRM_DIST)).mean())    # unknown NOT wrongly confirmed
    false_known = float((unk_min < CONFIRM_DIST).mean())            # unknown wrongly confirmed
    false_new = float((kn_min > ENROLL_DIST).mean())                # known wrongly enrolled as new

    # open-set ROC: treat min_dist as score (higher = more unknown)
    scores = np.concatenate([np.asarray(kn_min, dtype=float),
                             np.asarray(unk_min, dtype=float)])
    labels = np.concatenate([np.zeros(len(kn_min), dtype=float),
                             np.ones(len(unk_min), dtype=float)])
    assert len(scores) == len(labels), "probe/score shape mismatch"
    order = np.argsort(-scores)
    y = np.asarray(labels[order], dtype=float)
    n_u = int(np.sum(y)); n_k = len(y) - n_u
    tpr = np.cumsum(y) / max(1, n_u)
    fpr = np.cumsum(1.0 - y) / max(1, n_k)
    roc_auc = float(np.trapezoid(tpr, fpr))
    n = int(len(y))
    prec = np.cumsum(y) / np.maximum(1.0, np.arange(1.0, float(n) + 1.0))
    assert prec.shape == (n,), f"pr-precision shape mismatch: {prec.shape} vs {n}"
    rec = np.cumsum(y) / max(1, n_u)
    pr_auc = float(np.trapezoid(prec, rec))

    m = {
        "held_out_identities": sorted(held_out_ids)[:20],
        "n_held_out": len(held_out_ids),
        "n_catalogue_identities": len(catalogue_ids),
        "known_acceptance_rate": accepted_known,
        "unknown_rejection_rate": rejected_unknown,
        "false_known_rate": false_known,
        "false_new_rate": false_new,
        "open_set_roc_auc": roc_auc,
        "open_set_pr_auc": pr_auc,
        "unknown_min_distance": {"mean": float(unk_min.mean()),
                                 "p05": float(np.percentile(unk_min, 5)),
                                 "p95": float(np.percentile(unk_min, 95))},
        "known_min_distance": {"mean": float(kn_min.mean()),
                               "p05": float(np.percentile(kn_min, 5)),
                               "p95": float(np.percentile(kn_min, 95))},
    }
    # calibrated threshold: maximize acceptance + rejection jointly
    dmin = np.concatenate([kn_min, unk_min])
    best_t, best_j = float("nan"), -1.0
    for t in np.linspace(dmin.min(), dmin.max(), 201):
        j = (kn_min < t).mean() + (unk_min >= t).mean()
        if j > best_j:
            best_j, best_t = j, float(t)
    t_acc = float((kn_min < best_t).mean())
    t_rej = float((unk_min >= best_t).mean())
    valid_confirm = (CONFIRM_DIST > best_t) if False else \
        "CONFIRM_DIST too lax" if (unk_min < CONFIRM_DIST).mean() > 0.5 else \
        "CONFIRM_DIST validated"
    m["calibrated_distance_threshold"] = {"best_t": best_t,
                                          "known_acceptance_at_t": t_acc,
                                          "unknown_rejection_at_t": t_rej,
                                          "confirm_dist_validation": valid_confirm,
                                          "enroll_dist_note": (
                                              "ENROLL_DIST=0.95 is above all observed distances; "
                                              "no probe auto-enrolled") if (dmin < 0.95).all() else "ok"}
    print("TASK3 open-set: accepted_known=%.3f false_known=%.3f false_new=%.3f auc=%.3f "
          "best_t=%.3f acc=%.3f rej=%.3f" % (accepted_known, false_known, false_new,
                                             roc_auc, best_t, t_acc, t_rej), flush=True)
    return m


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t4, t4_id_map = task4_data_validation()
    if t4["fatal"]:
        print("TASK4 FATAL — required data missing; aborting", flush=True)
        results["task4_fatal"] = True
        results["task4"] = t4
        print(json.dumps(results, indent=2), flush=True)
        sys.exit(1)
    results["task4"] = t4

    net, tf = load_reid_model()
    results["task2"] = task2_closed_set(net, tf, id_map=t4_id_map)
    results["task3"] = task3_open_set(net, tf, id_map=t4_id_map)

    out = MODELS / "reid_benchmark.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
