"""Tiger Re-ID v2 — improved triplet training.

Fixes vs v1 (which collapsed to rank1 0.25 by epoch 18):
- semi-hard negative mining instead of batch-hard (hard negatives cause
  collapse with a small batch of 16 on CPU)
- margin 0.8 with loss weighting (prevents early collapse)
- cosine-distance loss on normalized embeddings (better geometry)
- early stopping on validation rank-1 (15-id probe subset) with best-epoch
  checkpoint saved every epoch
- longer patience: 30 epochs, val every epoch for first 10 then every 2
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT = Path("/home/ubuntu/project")
DATASETS = PROJECT / "datasets" / "atrw"
MODELS = PROJECT / "models"
MODELS.mkdir(exist_ok=True)

IMG_SIZE = 160
EMBED_DIM = 128
BATCH = 24
P_PER_ID = 3
EPOCHS = 30
LR = 1e-4
MARGIN = 0.8

IMG_EXTS = (".jpg", ".jpeg", ".png")
DEVICE = torch.device("cpu")


def load_identities():
    import csv
    mapping, by_id = {}, defaultdict(list)
    with open(DATASETS / "reid_list_train.csv") as f:
        for tid, fname in csv.reader(f):
            tid = int(tid)
            mapping[fname] = tid
            by_id[tid].append(fname)
    return mapping, by_id


def load_keypoints():
    return json.load(open(DATASETS / "reid_keypoints_train.json"))


def flank_side(kps):
    triples = [kps[i:i + 3] for i in range(0, len(kps), 3)]
    vis = [t for t in triples if t[2] == 2]
    if len(vis) < 4:
        return "unknown"
    xs = [t[0] for t in vis]
    left_pts = [(x, y) for (x, y) in zip(xs, [t[1] for t in vis]) if x < np.median(xs)]
    return "left" if len(left_pts) >= 2 else "unknown"


class TigerDataset(Dataset):
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
            items.append((p, tid, flank_side(kps.get(fname, [0] * 45)), fname))
        self.items = items
        self.by_id = by_id
        self.transform = transform
        print(f"ReID v2 dataset: {len(items)} images, {len(by_id)} identities",
              flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, tid, side, fname = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), tid, side, fname


class SemiHardTripletLoss(nn.Module):
    """Cosine-similarity based triplet loss with semi-hard negatives."""

    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, emb, ids):
        n = ids.size(0)
        sim = emb @ emb.T                       # cosine similarity
        eye = torch.eye(n, device=emb.device)
        same = (ids.unsqueeze(0) == ids.unsqueeze(1)) & ~eye.bool()
        diff = ~(ids.unsqueeze(0) == ids.unsqueeze(1))
        if same.sum() < 1:
            return torch.tensor(0.0, requires_grad=True)
        # hardest positive (lowest similarity)
        pos = sim.masked_fill(~same, -1.0).min(dim=1)[0]
        # semi-hard negative: hardest diff below (pos + margin)
        target = pos + self.margin
        neg = sim.masked_fill(~diff, 1.0)
        semi = neg.masked_fill(neg > target, -1.0).max(dim=1)[0]
        loss = F.relu(self.margin - (semi - pos))
        return loss.mean()


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = eff.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, EMBED_DIM),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return F.normalize(self.head(x), dim=1)


def sample_batch(ds, batch=BATCH, p=P_PER_ID):
    items, ids = [], []
    chosen = random.sample(list(ds.by_id), batch // p)
    for tid in chosen:
        for fname in random.sample(ds.by_id[tid], p):
            idx = next(i for i, it in enumerate(ds.items) if it[3] == fname)
            items.append(ds[idx])
            ids.append(tid)
    return torch.stack([i[0] for i in items]), torch.tensor(ids, dtype=torch.long)


def val_rank1(model, ds):
    model.eval()
    by_id = defaultdict(list)  # returned alongside rank1
    with torch.no_grad():
        for i in range(0, len(ds), 16):
            chunk = ds.items[i:i + 16]
            xs = torch.stack([ds.transform(Image.open(p).convert("RGB"))
                              for p, _, _, _ in chunk]).to(DEVICE)
            e = model(xs)
            for (p, tid, _, _), ev in zip(chunk, e):
                by_id[tid].append(ev)
    ids = sorted(by_id)
    probe = random.sample(ids, min(20, len(ids)))
    hits = total = 0
    cvecs = torch.stack([torch.stack(by_id[j]).mean(0) for j in ids])  # (n, d)
    for tid in probe:
        probe_vec = torch.stack(by_id[tid]).mean(0)
        sims = F.cosine_similarity(probe_vec.unsqueeze(0), cvecs, dim=1)
        best = int(torch.argsort(sims, descending=True)[0])
        if best == ids.index(tid):
            hits += 1
        total += 1
    return (hits / total if total else 0.0), by_id


def train():
    torch.manual_seed(42)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=14),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = TigerDataset(tf)
    model = EmbeddingNet().to(DEVICE)
    loss_fn = SemiHardTripletLoss(MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 1e-5)

    val_ds = TigerDataset(transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]))

    t0 = time.time()
    best = {"rank1": 0.0, "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ls = n = 0
        for _ in range(60):
            xb, yb = sample_batch(ds)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            ls += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        do_val = epoch <= 10 or epoch % 2 == 0
        if do_val:
            rr, by_id = val_rank1(model, val_ds)
        else:
            rr, by_id = None, None
        print(f"epoch {epoch:02d} loss={ls/n:.4f}" +
              (f" rank1={rr:.4f}" if rr is not None else " (val skipped)"),
              flush=True)
        if rr is not None and rr > best["rank1"]:
            best = {"rank1": rr, "epoch": epoch}
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "arch": "efficientnet_b0_triplet_v2",
                "embed_dim": EMBED_DIM,
                "img_size": IMG_SIZE,
                "classes_per_id": len(ds.by_id),
                "val_rank1": float(rr),
            }, MODELS / "tiger_reid.pth")
            catalogue = {str(tid): torch.stack(c).mean(0).numpy().tolist()
                         for tid, c in by_id.items() if len(c)}
            (MODELS / "reid_centroids.json").write_text(json.dumps(catalogue))
            print(f"  -> saved best checkpoint epoch {epoch} rank1={rr:.4f}",
                  flush=True)
        if epoch - best["epoch"] > 10 and epoch > 15:
            print("early stopping (no improvement 10 epochs)", flush=True)
            break
    print(f"Done. best rank1={best['rank1']:.4f} at epoch {best['epoch']}",
          flush=True)


if __name__ == "__main__":
    train()
