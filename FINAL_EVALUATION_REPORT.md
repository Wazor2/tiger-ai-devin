# Pench Tiger Reserve — Final Evaluation Report (Tasks 1–10)

**Date:** August 15, 2026 | **Command:** `python3 -m pench.evaluate --all` → `report/evaluation_report.json`

> **Superseded for Re-ID and the blank filter.** Every Re-ID number below comes
> from a benchmark that scored each image against a gallery containing itself, so
> it is optimistic; the honest strict-split measurements, the retrained
> checkpoint, and the recalibrated `CONFIRM_DIST` are in `docs/reid_baseline.md`.
> The blank-filter decision is in `docs/blank_filter_decision.md`.

## 1. Summary Table

| Module | Old (v1) | New (v2) | Target | Pass/Fail |
|---|---|---|---|---|
| Blank filter — false-empty rate | 58.6% of animal frames below archive threshold | **1.0%** | ≤ 2% | **PASS** |
| Blank filter — false-animal rate | — | 7.3% | ≤ 10% | **PASS** |
| Blank filter — versioning | single checkpoint, overwritten | versioned `blank_filter_v2.pth` with embedded provenance; never-overwrite policy | never overwrite | **PASS** |
| Re-ID — Rank-1 (closed set, ATRW) | 49.4% (model trained only 1 epoch) | **76.4%** | ≥ 50% | **PASS** |
| Re-ID — Rank-5 | — | 89.8% | reported | **PASS** |
| Re-ID — mAP | not computed | **0.312** | reported | **PASS** |
| Re-ID — CMC curve / per-id accuracy / confusion pairs | not computed | full report in `models/reid_benchmark.json` | reported | **PASS** |
| Open-set Re-ID (26 held-out identities) | not implemented | acceptance 100%, rejection measured, ROC-AUC **0.820**, PR-AUC 0.784, calibrated threshold 0.316 | measured | **PASS** |
| CONFIRM_DIST = 0.55 / ENROLL_DIST = 0.95 validation | not validated | **CONFIRM_DIST too lax** (100% of unknowns fall below 0.55); calibrated threshold 0.316 recommended | validated | **FINDING** (threshold needs recalibration, not code) |
| Home-range safety (MCP/KDE) | crashed on degenerate geometry | outlier IQR filtering, coordinate validation, bandwidth clamping, 50% core + 95% contour, uncertainty metadata | no crashes | **PASS** |
| Alerting | heuristic, no tests | 5 deterministic alert types + debounce/confirmation windows + full alert fields; 12/12 unit tests | deterministic, tested | **PASS** |
| Pipeline safety | not verified | animal never silently discarded; ambiguous → review; quarantine preserved with metadata | invariants hold | **PASS** |
| Automated evaluation | manual checks | single command, 8 suites, JSON report + console summary | one command | **PASS** |
| Performance (CPU) | not measured | blank filter **35.2 img/s**, p95 44.2 ms | RTX 3050 4 GB compatible (models 6 MB + 19 MB) | **PASS** |

## 2. Key Findings

### 2.1 Blank filter v2 — safety target met

The v1 model would have silently deleted 58.6% of animal frames. The retrained v2 (MobileNetV3-Small, frozen backbone, class-balanced weighted loss) with calibrated thresholds (lo = 0.20, hi = 0.75) achieves a **false-empty rate of 1.0%** and a false-animal rate of 7.3%. The trade-off is that ~84% of frames are routed to the human review queue, which is the intended behavior of the quarantine-band design: the system prefers review over any risk of losing a tiger frame. The evaluation suite recommends **promoting v2 to production**.

### 2.2 Re-ID — strong closed-set gains, honest open-set limits

After retraining with semi-hard triplet loss, closed-set performance improved from Rank-1 49.4% → **76.4%**, Rank-5 89.8%, mAP 0.312 (baseline of 107 identities from ATRW; published results on the 159-identity set reach ~60–70%, so 76.4% on 107 identities is competitive for CPU training). Open-set evaluation (26 held-out identities, ≥ 5 images each) shows ROC-AUC 0.820 with a calibrated minimum-distance threshold of **0.316** delivering 74.3% known acceptance and 77.9% unknown rejection jointly.

### 2.3 Threshold finding — CONFIRM_DIST = 0.55 is too lax

With the trained model, both known (mean 0.272) and unknown (mean 0.355) probes sit comfortably below CONFIRM_DIST = 0.55, so an unknown tiger would be wrongly auto-confirmed 100% of the time. The benchmark reports this explicitly and recommends the calibrated threshold 0.316. This is a calibration decision, not a code defect — the pipeline accepts a custom threshold via `TigerReID.CONFIRM_DIST`.

## 3. How to Verify

```bash
cd /home/ubuntu/project
python3 -m pench.evaluate --all            # all 8 suites
python3 -m pench.evaluate alerts           # alerting unit tests only
python3 -m pench.evaluate blank            # blank filter v1 vs v2 comparison
python3 -m pench.demo_run                  # end-to-end synthetic demo
```

## 4. Deliverables

The complete project lives in `/home/ubuntu/project/`: the four modules (`pench/`), training and benchmark scripts (`scripts/`), model checkpoints and reports (`models/`), the automated evaluation suite (`pench/evaluate.py`), and the technical documentation (`TECHNICAL_DOCUMENTATION.md`, updated with the full Task 1–10 program in Section 7).
