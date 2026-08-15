"""Inference used by the live capture loop.

`ModelInference` is the real thing (BlankFilter -> TigerReID). `DummyInference`
returns canned verdicts with no model loaded, so the capture -> API -> dashboard
plumbing can be proven end to end before any model is in the picture.
"""
import itertools
import time


class DummyInference:
    """Cycles animal / empty / review so every dashboard state gets exercised."""

    name = "dummy"

    CANNED = [
        {"verdict": "animal", "animal_confidence": 0.94,
         "reid": {"decision": "known_identity", "tiger_id": "172",
                  "cosine_distance": 0.21, "reason": "matches 172 (cos-dist 0.210)",
                  "top3": [["172", 0.21], ["85", 0.44], ["252", 0.47]]}},
        {"verdict": "empty", "animal_confidence": 0.03, "reid": None},
        {"verdict": "animal", "animal_confidence": 0.88,
         "reid": {"decision": "human_review", "tiger_id": None,
                  "cosine_distance": 0.42, "reason": "ambiguous match (cos-dist 0.420)",
                  "top3": [["61", 0.42], ["208", 0.45], ["12", 0.46]]}},
        {"verdict": "review", "animal_confidence": 0.55, "reid": None},
    ]

    def __init__(self):
        self._cycle = itertools.cycle(self.CANNED)
        self.blank_model = "dummy"
        self.band = [0.10, 0.80]

    def classify(self, image) -> dict:
        started = time.perf_counter()
        out = dict(next(self._cycle))
        out["quarantine_band"] = self.band
        out["model_version"] = "dummy"
        out["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return out


class ModelInference:
    """BlankFilter, then TigerReID for anything not confidently empty."""

    name = "models"

    def __init__(self):
        from pench.model_serving import BlankFilter, TigerReID
        self.blank = BlankFilter()
        self.reid = TigerReID()
        self.blank_model = self.blank.ckpt_path.name
        self.band = [self.blank.lo, self.blank.hi]

    def classify(self, image) -> dict:
        started = time.perf_counter()
        out = dict(self.blank.triage(image))
        out["reid"] = None
        # Quarantined frames still get identified: a 'review' verdict means the
        # blank filter is unsure, not that there is no tiger.
        if out["verdict"] in ("animal", "review"):
            rec = self.reid.identify(image)
            out["reid"] = {
                "decision": rec["decision"],
                "tiger_id": rec.get("top_match"),
                "cosine_distance": rec.get("cosine_distance"),
                "reason": rec.get("reason"),
                "top3": rec.get("top3", []),
                "confirm_dist": self.reid.CONFIRM_DIST,
            }
        out["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return out


def build_inference(kind: str):
    return DummyInference() if kind == "dummy" else ModelInference()
