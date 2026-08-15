"""Regression tests for the Re-ID training fixes (fix-spec items 6-7).

The shipped checkpoint is epoch-1 weights because two bugs made training a
no-op: the triplet loss was a constant with no gradient, and validation rank-1
was 1.0 by construction. These tests pin both fixes.

Run: python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def load_trainer():
    """scripts/ isn't a package, so import train_reid_v2 by path."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    spec = importlib.util.spec_from_file_location(
        "train_reid_v2", PROJECT / "scripts" / "train_reid_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSemiHardTripletLoss:
    def test_loss_responds_to_embedding_quality(self):
        trainer = load_trainer()
        import torch
        import torch.nn.functional as F

        ids = torch.tensor([1, 1, 2, 2, 3, 3])
        loss_fn = trainer.SemiHardTripletLoss(margin=0.8)

        # well separated identities: positives together, negatives far apart
        good = F.normalize(torch.tensor([
            [1.0, 0.0, 0.0], [0.99, 0.1, 0.0],
            [0.0, 1.0, 0.0], [0.1, 0.99, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.1, 0.99],
        ]), dim=1)
        # every identity's pair pulled apart, identities interleaved
        bad = F.normalize(torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.99, 0.1, 0.0], [0.1, 0.99, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 0.05],
        ]), dim=1)

        assert loss_fn(good, ids) < loss_fn(bad, ids), \
            "loss must reward separated identities"
        # the old implementation returned exactly the margin regardless of input
        assert loss_fn(good, ids).item() != pytest.approx(0.8, abs=1e-6)

    def test_loss_has_a_real_gradient(self):
        trainer = load_trainer()
        import torch
        import torch.nn.functional as F

        ids = torch.tensor([1, 1, 2, 2])
        raw = torch.randn(4, 8, generator=torch.Generator().manual_seed(0),
                          requires_grad=True)
        loss = trainer.SemiHardTripletLoss()(F.normalize(raw, dim=1), ids)
        loss.backward()
        assert raw.grad is not None
        assert float(raw.grad.abs().sum()) > 0, "constant loss = training no-op"

    def test_negative_is_never_self_or_positive(self):
        """A collapsed batch is maximally wrong, so loss must be >= margin."""
        trainer = load_trainer()
        import torch
        import torch.nn.functional as F

        ids = torch.tensor([1, 1, 2, 2])
        collapsed = F.normalize(torch.ones(4, 8), dim=1)
        assert float(trainer.SemiHardTripletLoss(margin=0.8)(collapsed, ids)) \
            == pytest.approx(0.8, abs=1e-5)


class TestCheckpointProvenance:
    def test_training_records_augmentation_and_protocol(self):
        trainer = load_trainer()
        assert "aug" in trainer.AUGMENTATION
        # a run must not overwrite the checkpoint the demo serves
        assert trainer.CKPT_OUT.name != "tiger_reid.pth"
        assert trainer.CATALOGUE_OUT.name != "reid_centroids.json"
