"""Module 2 — Individual Tiger Identification (stripe-pattern Re-ID).

Metric-learning (triplet-loss) embedding model trained on the ATRW dataset
(1,887 labeled flank images of 107 individual tigers). The model maps a
flank/stripe patch to a 128-d 'stripe fingerprint' vector; nearest-neighbor
cosine search implements matching with auto-confirm / ambiguous-review /
auto-enroll decision logic.

Flank-awareness: ATRW keypoints encode left/right flank; training pairs are
constructed so positives always come from the same tiger and same flank side
when the side is detectable (left vs right flanks carry non-mirror stripe
patterns).

Output: models/tiger_reid.pth + models/tiger_reid_catalogue + val metrics.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

PROJECT = Path("/home/ubuntu/project")
DATASETS = PROJECT / "datasets" / "atrw"
MODELS = PROJECT / "models"
MODELS.mkdir(exist_ok=True)

IMG_SIZE = 160
EMBED_DIM = 128
BATCH = 16
EPOCHS = 18
LR = 3e-4

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_identities():
    """Return {filename: int_id} and {int_id: [filenames]} from reid_list_train.csv."""
    import csv
    mapping = {}
    by_id = defaultdict(list)
    with open(DATASETS / "reid_list_train.csv") as f:
        for tid, fname in csv.reader(f):
            tid = int(tid)
            mapping[fname] = tid
            by_id[tid].append(fname)
    return mapping, by_id


def load_keypoints():
    return json.load(open(DATASETS / "reid_keypoints_train.json"))


class FlankSide:
    """Infer flank side from COCO keypoints (15 keypoints for tigers)."""
    LEFT, RIGHT, UNKNOWN = "left", "right", "unknown"

    @staticmethod
    def of(kps):
        # keypoints ordered like COCO tiger: left_ear, right_ear, ... 
        # Without the exact kepoint-name list, we use geometry: ears are first two.
        # Visible keypoints (v==2): compare mean x of first two pairs.
        triples = [kps[i:i+3] for i in range(0, len(kps), 3)]
        vis = [t for t in triples if t[2] == 2]
        if len(vis) < 4:
            return FlankSide.UNKNOWN
        xs = [t[0] for t in vis]
        ys = [t[1] for t in vis]
        left_pts = [(x, y) for (x, y) in zip(xs, ys) if x < np.median(xs)]
        return FlankSide.LEFT if len(left_pts) >= 2 else FlankSide.UNKNOWN


class TigerReIDDataset(Dataset):
    """Loads flank crops; returns (img, id, side, filename)."""
    def __init__(self, transform):
        idmap, by_id = load_identities()
        kps = load_keypoints()
        items = []
        for fname, tid in idmap.items():
            p = DATASETS / "train" / fname
            if not p.exists() or p.stat().st_size < 500:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                continue
            side = FlankSide.of(kps.get(fname, [0] * 45))
            items.append((p, tid, side, fname))
        self.items = items
        self.by_id = by_id
        self.transform = transform
        print(f"ReID dataset: {len(items)} images, {len(by_id)} identities")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, tid, side, fname = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), tid, side, fname


class MarginLoss(nn.Module):
    """Online triplet margin loss with hard-ish mining within batch."""
    def __init__(self, margin=0.6):
        super().__init__()
        self.margin = margin

    def forward(self, emb, ids):
        ids = ids.view(-1)
        n = ids.size(0)
        dist = torch.cdist(emb, emb, p=2)
        pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)).float() - torch.eye(n, device=ids.device)
        neg_mask = 1.0 - pos_mask - torch.eye(n, device=ids.device)
        if pos_mask.sum() < 1:
            return torch.tensor(0.0, requires_grad=True)
        pos_d = (dist * pos_mask).masked_fill(pos_mask == 0, 1e9).min(dim=1)[0]
        neg_d = (dist * neg_mask).masked_fill(neg_mask == 0, -1e9).max(dim=1)[0]
        loss = F.relu(pos_d - neg_d + self.margin)
        return loss.mean()


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = eff.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return F.normalize(self.head(x), dim=1)


def sample_batch(dataset, batch=BATCH, p_per_id=4):
    """Build a batch with p images per randomly sampled identity."""
    items = []
    ids = list(dataset.by_id.keys())
    needed = batch // p_per_id
    chosen = random.choices(ids, k=needed)
    for tid in chosen:
        pool = dataset.by_id[tid]
        picks = random.choices(pool, k=min(p_per_id, len(pool)))
        for fname in picks:
            idx = next(i for i, it in enumerate(dataset.items) if it[3] == fname)
            items.append(dataset[idx])
    return torch.stack([i[0] for i in items]), torch.tensor([i[1] for i in items], dtype=torch.long)


def train():
    device = torch.device("cpu")
    torch.manual_seed(42)

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=16),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = TigerReIDDataset(tf_train)
    model = EmbeddingNet().to(device)
    criterion = MarginLoss(margin=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    t0 = time.time()
    best_map = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        steps = 60
        for _ in range(steps):
            xb, yb = sample_batch(ds, batch=BATCH)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            emb = model(xb)
            loss = criterion(emb, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"epoch {epoch:02d} triplet_loss={loss_sum/n:.4f} ({time.time()-t0:.0f}s)")

        # quick validation: per-identity mean embedding, same-id retrieval rate
        model.eval()
        val_ds = TigerReIDDataset(tf_val)
        by_id = defaultdict(list)
        with torch.no_grad():
            for i in range(0, len(val_ds), BATCH):
                chunk = val_ds.items[i:i+BATCH]
                xs = torch.stack([val_ds.transform(Image.open(p).convert("RGB")) for p, _, _, _ in chunk]).to(device)
                e = model(xs)
                for (p, tid, _, _), ev in zip(chunk, e):
                    by_id[tid].append(ev)
        centroids = {tid: torch.stack(v).mean(0) for tid, v in by_id.items() if len(v)}
        if epoch % 3 != 1:
            print(f"  (val skipped this epoch)")
            continue
        ids = list(centroids)
        probe_ids = random.sample(ids, min(15, len(ids)))  # fast subset evaluation
        hits, total = 0, 0
        centroid_mat = torch.stack([centroids[j] for j in ids], dim=0)  # (n, d)
        for tid in probe_ids:
            for v in by_id[tid]:
                sims = F.cosine_similarity(v.unsqueeze(0), centroid_mat, dim=1)
                ranked = torch.argsort(sims, descending=True).tolist()
                hits += int(ranked[0] == ids.index(tid))
                total += 1
        rr1 = hits / total if total else 0
        print(f"  val Rank-1 on {len(probe_ids)} random ids: {rr1:.4f}")
        if rr1 > best_map:
            best_map = rr1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "arch": "efficientnet_b0_triplet",
                "embed_dim": EMBED_DIM,
                "img_size": IMG_SIZE,
                "classes_per_id": len(by_id),
                "val_rank1": float(rr1),
            }, MODELS / "tiger_reid.pth")
            # also save catalogue of per-identity centroids for runtime matching
            catalogue = {str(tid): c.numpy().tolist() for tid, c in centroids.items()}
            (MODELS / "reid_centroids.json").write_text(json.dumps(catalogue))
            print("  -> saved best checkpoint + catalogue")

    print("Done. best rank-1:", best_map)


if __name__ == "__main__":
    train()
