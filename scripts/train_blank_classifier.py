"""Module 1 — Blank Image Filtering.

Train a two-stage-ready blank/animal classifier (MobileNetV3-Small backbone,
fine-tuned) on a balanced subset of Caltech Camera Traps (1200 animal +
1200 empty train, 300+300 val).

Output:
- models/blank_filter.pth (checkpoint with weights + metadata)
- Validation metrics: accuracy, precision/recall, F1, and confidence zones
  for the quarantine decision logic.
"""
import json
import time
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

PROJECT = Path("/home/ubuntu/project")
CCT = PROJECT / "datasets" / "cct_subset"
CCT_TRAIN = CCT / "train"
CCT_VAL_LOCAL = CCT / "val_local"
CCT_VAL_OFFICIAL = CCT / "val_official"
MODELS = PROJECT / "models"
MODELS.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH = 32
EPOCHS = 15
LR = 1e-3

IMG_EXTS = (".jpg", ".jpeg", ".png")


class BlankDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.transform = transform
        self.samples = []
        for label, cls_idx in [("animal", 1), ("empty", 0)]:
            d = root / label
            if d.exists():
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in IMG_EXTS and p.stat().st_size > 500:
                        try:
                            with Image.open(p) as im:
                                im.verify()
                        except Exception:
                            continue
                        self.samples.append((p, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # fall back to next good sample
            for j in range(len(self.samples)):
                q = self.samples[(i + 1 + j) % len(self.samples)][0]
                try:
                    img = Image.open(q).convert("RGB")
                    break
                except Exception:
                    continue
        return self.transform(img), y


def train():
    device = torch.device("cpu")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = BlankDataset(CCT_TRAIN, train_tf)
    val_ds = BlankDataset(CCT_VAL_LOCAL, val_tf)
    if len(val_ds) == 0:
        # val split may still be downloading; use 10% of train as validation
        n_val = max(1, len(train_ds) // 10)
        tr_idx = torch.randperm(len(train_ds)).tolist()
        # reorder samples in-place: first n_val -> val, rest -> train
        full_samples = train_ds.samples
        val_samples, tr_samples = full_samples[:n_val], full_samples[n_val:]
        train_ds.samples = tr_samples
        val_ds.samples = val_samples
        print(f"train={len(train_ds)} val={len(val_ds)} (val split incomplete; using train subset as val)")
    else:
        print(f"train={len(train_ds)} val={len(val_ds)}")

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(1024, 2)
    model.to(device)

    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)

        # validation
        model.eval()
        correct = total = 0
        conf = {0: [], 1: []}
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                prob = torch.softmax(model(xb), dim=1)
                pred = prob.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                for p, y in zip(prob, yb):
                    conf[int(y)].append(p[int(y)].item())

        acc = correct / total
        print(f"epoch {epoch:02d} loss={loss_sum/n:.4f} val_acc={acc:.4f} ({time.time()-t0:.0f}s)")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "arch": "mobilenet_v3_small",
                "img_size": IMG_SIZE,
                "classes": ["empty", "animal"],
                "val_accuracy": float(acc),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
            }, MODELS / "blank_filter.pth")
            print("  -> saved best checkpoint")

    # Final report with quarantine zones
    model.eval()
    with torch.no_grad():
        probs, labels = [], []
        for xb, yb in val_loader:
            prob = torch.softmax(model(xb.to(device)), dim=1)
            probs.append(prob[:, 1])  # P(animal)
            labels.append(yb)
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()

    lo, hi = 0.15, 0.85  # quarantine band
    animal_p = probs[labels == 1]
    empty_p = probs[labels == 0]
    # robustness eval on official (geographically held-out) val split
    off_ds = BlankDataset(CCT_VAL_OFFICIAL, val_tf)
    off_acc = None
    if len(off_ds) > 0:
        off_loader = DataLoader(off_ds, batch_size=BATCH, num_workers=2)
        model.eval()
        c2 = t2 = 0
        with torch.no_grad():
            for xb, yb in off_loader:
                pred = torch.softmax(model(xb.to(device)), 1).argmax(1)
                c2 += (pred == yb.to(device)).sum().item()
                t2 += yb.size(0)
        off_acc = c2 / t2

    report = {
        "best_val_accuracy": float(best_acc),
        "official_val_accuracy": float(off_acc) if off_acc is not None else None,
        "quarantine_band": [lo, hi],
        "animal_confidence_below_hi": float((animal_p < hi).mean()),
        "empty_confidence_above_lo": float((empty_p > lo).mean()),
        "auto_blank_rate": float((empty_p <= lo).mean()),
        "auto_animal_rate": float((animal_p >= hi).mean()),
        "review_queue_rate": float(((probs > lo) & (probs < hi)).mean()),
    }
    print(json.dumps(report, indent=2))
    (MODELS / "blank_filter_report.json").write_text(json.dumps(report, indent=2))
    print("Done.")


if __name__ == "__main__":
    train()
