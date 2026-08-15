"""Shared model inference: blank filter (Module 1) + tiger Re-ID (Module 2).

Provides ready-to-use classes that wrap the trained PyTorch checkpoints.
Runs on CUDA when available (RTX box) and falls back to CPU (field laptop);
set PENCH_DEVICE=cpu to force CPU even on a GPU machine.
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (efficientnet_b0, mobilenet_v3_small,
                                EfficientNet_B0_Weights, MobileNet_V3_Small_Weights)

PROJECT = Path(__file__).resolve().parent.parent
MODELS = PROJECT / "models"


def resolve_device(spec: Optional[str] = None) -> torch.device:
    """PENCH_DEVICE wins, then CUDA if usable, else CPU."""
    spec = spec or os.environ.get("PENCH_DEVICE")
    if spec:
        dev = torch.device(spec)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"PENCH_DEVICE={spec} but no CUDA device is available")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = resolve_device()


class BlankFilter:
    """Module 1: two-zone confidence classifier with quarantine band.

    Returns one of: 'animal' (auto-accept), 'empty' (auto-archive),
    'review' (quarantine zone -> human review).
    """

    # Fallback band, used only if the checkpoint carries no calibrated one.
    # Deliberately wide: over-quarantining is recoverable, auto-archiving a
    # real tiger is not.
    DEFAULT_BAND = (0.15, 0.85)

    def __init__(self, ckpt_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        self.device = torch.device(device) if device is not None else DEVICE
        # Prefer the newest calibrated checkpoint present. v1
        # (blank_filter.pth) has an uncalibrated 33-58% false-empty rate and is
        # only a last resort.
        if ckpt_path is None:
            for name in ("blank_filter_v3.pth", "blank_filter_v2.pth",
                         "blank_filter.pth"):
                if (MODELS / name).exists():
                    ckpt_path = MODELS / name
                    break
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.ckpt_path = ckpt_path
        self.version = ckpt.get("version", "v1")
        self.img_size = ckpt.get("img_size", 224)
        # The quarantine band belongs to the weights it was swept on, so read
        # it from the checkpoint rather than hardcoding one band for every
        # model (see scripts/calibrate_blank_v2.py).
        lo, hi = ckpt.get("threshold_lo"), ckpt.get("threshold_hi")
        if lo is None or hi is None or not 0.0 <= lo < hi <= 1.0:
            lo, hi = self.DEFAULT_BAND
            self.band_source = "default_fallback"
        else:
            self.band_source = "checkpoint"
        self.lo, self.hi = float(lo), float(hi)
        self.calibration = ckpt.get("calibration", {})
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(1024, 2)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(self.device)
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def triage(self, image) -> dict:
        """`image` may be a path or an already-loaded PIL image (live capture)."""
        im = (image if isinstance(image, Image.Image)
              else Image.open(image)).convert("RGB")
        x = self.transform(im).unsqueeze(0).to(self.device)
        p = torch.softmax(self.model(x), 1)[0]
        p_animal = p[1].item()
        if p_animal >= self.hi:
            verdict = "animal"
        elif p_animal <= self.lo:
            verdict = "empty"
        else:
            verdict = "review"
        return {"verdict": verdict,
                "animal_confidence": round(p_animal, 4),
                "quarantine_band": [self.lo, self.hi],
                "model_version": self.version}


class TigerReID:
    """Module 2: stripe-pattern embedding + FAISS index over identity centroids.

    Decision logic:
      - known_identity: top-1 distance < threshold  -> auto-confirm
      - new_identity:    distance > upper threshold -> auto-enroll
      - otherwise: human_review
    """

    # Open-set calibrated cosine-distance thresholds. CONFIRM_DIST comes from the
    # held-out-identity sweep in models/reid_benchmark.json (task3
    # calibrated_distance_threshold.best_t): ROC-AUC 0.820, 74.3% known-acceptance,
    # 77.9% unknown-rejection. The previous 0.55 auto-confirmed 100% of unknown
    # tigers, so no unknown individual could ever be flagged for review.
    CONFIRM_DIST = 0.316
    # NOTE: ENROLL_DIST is still unvalidated — the benchmark observed no probe
    # distance above it, so auto-enrol never fires and unmatched tigers land in
    # human_review instead. Left unchanged deliberately (fix-spec item 6).
    ENROLL_DIST = 0.95

    def __init__(self, ckpt_path: Optional[Path] = None,
                 catalogue_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        self.device = torch.device(device) if device is not None else DEVICE
        ckpt_path = ckpt_path or MODELS / "tiger_reid.pth"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.img_size = ckpt.get("img_size", 160)
        self.embed_dim = ckpt.get("embed_dim", 128)

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        features = model.features
        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
        )
        state = ckpt["state_dict"]
        # load features + head from checkpoint
        feat_state = {k.replace("features.", ""): v for k, v in state.items()
                      if k.startswith("features.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items()
                      if k.startswith("head.")}
        features.load_state_dict(feat_state)
        if not head_state:
            raise RuntimeError("checkpoint has no head weights")
        head.load_state_dict(head_state, strict=False)
        # fill any missing BN running buffers with neutral defaults so the
        # loaded head computes correctly despite minor architecture drift
        for mod in head.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if mod.running_mean is None:
                    mod.running_mean = torch.zeros(mod.num_features)
                    mod.running_var = torch.ones(mod.num_features)
        self.model = nn.Sequential(features, nn.AdaptiveAvgPool2d(1), head)
        self.model.eval().to(self.device)

        # FAISS flat index over stored identity centroids
        catalogue_path = catalogue_path or MODELS / "reid_centroids.json"
        cat = json.load(open(catalogue_path))
        if isinstance(cat, dict) and cat and isinstance(next(iter(cat.values())), dict):
            cat = {k: v["centroid"] for k, v in cat.items()}
        self.identities = sorted(cat.keys())
        vecs = np.array([cat[i] for i in self.identities], dtype="float32")
        if vecs.size:
            import faiss
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self.index.add(vecs / (norms + 1e-12))
        else:
            self.index = None

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, image) -> np.ndarray:
        """`image` may be a path or an already-loaded PIL image (live capture)."""
        im = (image if isinstance(image, Image.Image)
              else Image.open(image)).convert("RGB")
        x = self.transform(im).unsqueeze(0).to(self.device)
        v = F.normalize(self.model(x), dim=1)[0]
        return v.cpu().numpy().astype("float32")

    @torch.no_grad()
    def identify(self, image) -> dict:
        vec = self.embed(image)
        if self.index is None or self.index.ntotal == 0:
            return {"decision": "human_review", "reason": "no identities enrolled",
                    "embedding_norm": round(float(np.linalg.norm(vec)), 4)}
        D, I = self.index.search(vec.reshape(1, -1), 3)
        dist = float(1 - D[0][0])           # cosine distance
        top_id = self.identities[I[0][0]]
        if dist < self.CONFIRM_DIST:
            decision, reason = "known_identity", f"matches {top_id} (cos-dist {dist:.3f})"
        elif dist > self.ENROLL_DIST:
            decision, reason = "new_identity", f"no close match (min cos-dist {dist:.3f})"
        else:
            decision, reason = "human_review", f"ambiguous match (cos-dist {dist:.3f})"
        return {"decision": decision, "reason": reason,
                "top_match": top_id if decision != "new_identity" else None,
                "cosine_distance": round(dist, 4),
                "top3": [(self.identities[I[0][k]], round(float(1 - D[0][k]), 4))
                         for k in range(min(3, len(I[0]))) if I[0][k] >= 0]}

    def enroll(self, tiger_id: str, vec: np.ndarray, catalogue_path: Optional[Path] = None):
        """Auto-enroll a new identity into the catalogue and index."""
        catalogue_path = catalogue_path or MODELS / "reid_centroids.json"
        cat = json.load(open(catalogue_path))
        cat[tiger_id] = {"centroid": vec.tolist(), "n_samples": 1}
        json.dump(cat, open(catalogue_path, "w"), indent=1)
        self.identities.append(tiger_id)
        norms = np.linalg.norm(vec, keepdims=True)
        self.index.add((vec / (norms + 1e-12)).reshape(1, -1))
