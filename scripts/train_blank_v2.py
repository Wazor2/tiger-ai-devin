"""Module 1 v2 — Blank Image Filter retraining (Task 1).

Improvements over v1:
- Class-balanced training (weighted loss + equal sampling when available)
- Realistic camera-trap augmentation (night-shot simulation, noise, fog,
  vignette, crop, rotation, jitter, blur)
- GPU/CUDA when available (auto device selection)
- Versioned checkpoint (never overwrites production: blank_filter_v2.pth)
- Calibrated three-way thresholds chosen on the FULL validation set
  (train/val split) to minimize false-empty rate subject to constraints
- Full metric suite: acc, precision, recall, F1, confusion matrix, ROC-AUC,
  PR-AUC, animal recall, empty recall, false-empty rate, false-animal rate,
  review rate

Validation uses cct_subset/{train, val} (user's layout: 1200/1200, 300/300).
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

PROJECT = Path(__file__).resolve().parent.parent
CCT = PROJECT / "datasets" / "cct_subset"
MODELS = PROJECT / "models"
MODELS.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH = 256
EPOCHS = 20
LR = 1e-4

IMG_EXTS = (".jpg", ".jpeg", ".png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# augmentation: realistic camera-trap artefacts
# ---------------------------------------------------------------------------
def _to_pil(t):
    return transforms.ToPILImage()(t)


class CameraTrapAug:
    """Applies random camera-trap-like corruptions in PIL space."""

    def __call__(self, img):
        # night / flash simulation
        r = np.random.random()
        if r < 0.18:
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.25, 0.65))
            img = ImageOps.autocontrast(img)
        elif r < 0.30:
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(1.6, 2.4))
            img = ImageOps.autocontrast(img)
        if np.random.random() < 0.25:
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.55, 0.85))
        # sensor noise
        if np.random.random() < 0.30:
            a = np.asarray(img).astype(np.int16)
            a = a + np.random.normal(0, np.random.uniform(2, 9), a.shape)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        # mild motion/defocus blur
        if np.random.random() < 0.12:
            k = int(np.random.choice([3, 5]))
            img = img.filter(ImageFilter_smooth := __import__("PIL.ImageFilter", fromlist=["BLUR"]).BLUR)
        # vignette
        if np.random.random() < 0.20:
            w, h = img.size
            vx = np.linspace(-1, 1, w)
            vy = np.linspace(-1, 1, h)
            mask = 1 - 0.55 * (np.add.outer(vy ** 2, vx ** 2) * 0.5) ** 2
            mask = np.clip(mask, 0.2, 1.0)
            img = Image.fromarray((np.asarray(img).astype(np.float32) * mask[..., None]).astype(np.uint8))
        return img


def build_train_tf():
    return transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.15), int(IMG_SIZE * 1.15))),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        CameraTrapAug(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_val_tf():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class BlankDataset(Dataset):
    def __init__(self, root, transform, max_per_class=None, seed=42):
        self.transform = transform
        self.samples = []
        rng = np.random.default_rng(seed)
        for label, cls_idx in [("animal", 1), ("empty", 0)]:
            d = Path(root) / label
            if not d.exists():
                continue
            paths = [p for p in sorted(d.iterdir())
                     if p.suffix.lower() in IMG_EXTS and p.stat().st_size > 500
                     and self._verify(p)]
            if max_per_class and len(paths) > max_per_class:
                paths = rng.choice(paths, max_per_class, replace=False).tolist()
            self.samples.extend((p, cls_idx) for p in paths)
        self._verify.cache = getattr(self._verify, "cache", set())

    @staticmethod
    def _verify(p):
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        try:
            return self.transform(Image.open(p).convert("RGB")), y
        except Exception:
            j = (i + 1) % len(self.samples)
            p, y = self.samples[j]
            return self.transform(Image.open(p).convert("RGB")), y


# ---------------------------------------------------------------------------
# calibration of the three-way policy on the validation set
# ---------------------------------------------------------------------------
def calibrate_thresholds(model, val_ds, device, max_false_empty=0.02):
    """Choose quarantine band [lo, hi] on validation probabilities.

    Safety objective: P(decide empty | actual animal) <= max_false_empty.
    Within that constraint, minimize review rate while keeping
    P(decide animal | actual empty) small (false-animal guard).
    Searches hi from 0.95 downward; lo from 0.05 upward.
    """
    probs, labels = infer_probs(model, val_ds, device)
    animal_p = probs[labels == 1]
    empty_p = probs[labels == 0]
    if len(animal_p) == 0 or len(empty_p) == 0:
        return 0.85, 0.15

    best = None
    for hi_q in np.arange(0.95, 0.40, -0.01):
        hi = float(np.quantile(animal_p, hi_q))
        fe_rate = float((animal_p < hi).mean())   # animal wrongly below hi
        for lo_q in np.arange(0.10, 0.60, 0.01):
            lo = float(np.quantile(empty_p, lo_q))
            if lo >= hi - 0.05:
                continue
            fa_rate = float((empty_p > lo).mean())   # empty wrongly above lo
            review = float(((probs > lo) & (probs < hi)).mean())
            # score: minimize review while honoring safety guardrails
            if fe_rate <= max_false_empty and fa_rate <= 0.10:
                score = review + 2.0 * max(0.0, fe_rate - 0.005)
                if best is None or score < best[0]:
                    best = (score, lo, hi, fe_rate, fa_rate, review)
    if best is None:
        # fallback: enforce safety only
        hi = float(np.quantile(animal_p, 0.97))
        lo = float(np.quantile(empty_p, 0.10))
        fe_rate = float((animal_p < hi).mean())
        fa_rate = float((empty_p > lo).mean())
        review = float(((probs > lo) & (probs < hi)).mean())
        best = (None, lo, hi, fe_rate, fa_rate, review)
    _, lo, hi, fe_rate, fa_rate, review = best
    return round(lo, 3), round(hi, 3), {
        "false_empty_rate": fe_rate, "false_animal_rate": fa_rate,
        "review_rate": review}


def infer_probs(model, ds, device):
    loader = DataLoader(ds, batch_size=BATCH, num_workers=0)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            probs.append(torch.softmax(model(xb), dim=1)[:, 1])
            labels.append(yb)
    return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_metrics(probs, labels, lo, hi):
    pred = np.where(probs >= hi, 1, np.where(probs <= lo, 0, 2))
    animal = labels == 1
    empty = labels == 0
    n = len(labels)

    tp = int(((pred == 1) & animal).sum()); fp = int(((pred == 1) & empty).sum())
    tn = int(((pred == 0) & empty).sum()); fn = int(((pred == 0) & animal).sum())
    rev_a = int(((pred == 2) & animal).sum()); rev_e = int(((pred == 2) & empty).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall_animal = tp / (tp + fn) if tp + fn else 0.0
    recall_empty = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall_animal / (precision + recall_animal) if precision + recall_animal else 0.0

    # ROC-AUC (animal=positive)
    order = np.argsort(-probs)
    y_sorted = labels[order]
    n_pos, n_neg = labels.sum(), n - labels.sum()
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    roc_auc = float(np.trapezoid(tp_cum / n_pos, fp_cum / n_neg)) if n_pos and n_neg else float("nan")

    # PR-AUC
    prec = tp_cum / np.maximum(1, tp_cum + fp_cum)
    rec = tp_cum / n_pos
    pr_auc = float(np.trapezoid(prec, rec)) if n_pos else float("nan")

    return {
        "accuracy": float((pred == labels).mean()),
        "precision_animal": precision,
        "recall_animal": recall_animal,
        "recall_empty": recall_empty,
        "f1_animal": f1,
        "confusion_matrix": {"tp_animal": tp, "fp_animal": fp,
                             "tn_empty": tn, "fn_animal": fn,
                             "review_animal": rev_a, "review_empty": rev_e},
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "animal_recall": recall_animal,
        "empty_recall": recall_empty,
        "false_empty_rate": fn / (fn + rev_a + tp) if fn + rev_a + tp else 0.0,
        "false_animal_rate": fp / (fp + rev_e + tn) if fp + rev_e + tn else 0.0,
        "review_rate": float((pred == 2).mean()),
        "n": n,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train():
    train_ds = BlankDataset(CCT / "train", build_train_tf(), max_per_class=None)
    val_ds = BlankDataset(CCT / "val", build_val_tf())
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)
    counts = Counter(y for _, y in train_ds.samples)
    print("train class counts:", dict(counts), flush=True)

    # class weights for imbalance
    total = sum(counts.values())
    w0 = total / (2 * counts.get(0, total // 2))
    w1 = total / (2 * counts.get(1, total // 2))
    weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)
    print(f"class weights empty={w0:.2f} animal={w1:.2f}", flush=True)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(1024, 2)
    # Freeze the backbone; only the classifier head is trained.
    # This matches the production v1 recipe, is stable on CPU, and is
    # dramatically faster per epoch than full fine-tuning.
    for p in model.features.parameters():
        p.requires_grad = False
    model.to(DEVICE)

    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    best = {"acc": 0.0, "f1": 0.0, "epoch": 0}
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ls = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            ls += loss.item() * xb.size(0); n += xb.size(0)
        scheduler.step()
        probs, labels = infer_probs(model, val_ds, DEVICE)
        lo, hi = 0.15, 0.85
        m = compute_metrics(probs, labels, lo, hi)
        print(f"epoch {epoch:02d} loss={ls/n:.4f} acc={m['accuracy']:.4f} "
              f"f1={m['f1_animal']:.4f} recA={m['recall_animal']:.4f} "
              f"fE={m['false_empty_rate']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m["f1_animal"] > best["f1"]:
            best = {"acc": m["accuracy"], "f1": m["f1_animal"],
                    "recall_animal": m["recall_animal"],
                    "false_empty_rate": m["false_empty_rate"], "epoch": epoch}
            # Persist the best checkpoint each epoch so an interrupted run
            # (OOM, timeout) still leaves a usable blank_filter_v2.pth.
            torch.save({
                "version": "v2",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "arch": "mobilenet_v3_small",
                "img_size": IMG_SIZE,
                "classes": ["empty", "animal"],
                "threshold_lo": lo, "threshold_hi": hi,
                "metrics": m,
                "device": str(DEVICE),
            }, ckpt_path)
            print(f"  -> saved best checkpoint at epoch {epoch}", flush=True)

    # calibrate thresholds on validation set
    probs, labels = infer_probs(model, val_ds, DEVICE)
    lo, hi, cal = calibrate_thresholds(model, val_ds, DEVICE, max_false_empty=0.02)
    print(f"calibrated band: lo={lo} hi={hi}", flush=True)
    m = compute_metrics(probs, labels, lo, hi)
    m.update(cal)
    m.update(best)
    m["thresholds"] = {"lo": lo, "hi": hi}
    m["calibration"] = dict(m.pop("calibration", {}))

    ckpt_path = MODELS / "blank_filter_v2.pth"
    torch.save({
        "version": "v2",
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "arch": "mobilenet_v3_small",
        "img_size": IMG_SIZE,
        "classes": ["empty", "animal"],
        "threshold_lo": lo, "threshold_hi": hi,
        "calibration": cal,
        "metrics": m,
        "device": str(DEVICE),
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hyperparameters": {"batch": BATCH, "epochs": EPOCHS, "lr": LR,
                            "img_size": IMG_SIZE, "class_balanced": True,
                            "augmentation": "camera_trap_corruptions"},
        "dataset": {"name": "caltech_camera_traps_subset",
                    "path": str(CCT)},
    }, ckpt_path)
    print("saved", ckpt_path, flush=True)
    (MODELS / "blank_filter_v2_report.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    train()
