# Blank filter: checkpoint and threshold decision (fix-spec item 2)

## What the audit asked for

Use `blank_filter_v2.pth` rather than v1, re-derive the quarantine band from the
model's own validation score distribution rather than copying v1's numbers, keep
`false_empty_rate <= 0.02`, and aim for a `review_rate` of 20-30%. If the target
is unreachable, report the measured number instead of inventing a band.

## What the validation data turned out to be

`models/blank_filter_v2_report.json` stores aggregate metrics only, and the CCT
subset it was scored on was not in the archive, so the sweep had no score
distribution to run on. `scripts/fetch_cct_subset.py` rebuilds a subset from the
public Caltech Camera Traps release (1200/1200 train, 300/300 val).

The first rebuild reproduced the original script's sampling: fill an `animal`
bucket and an `empty` bucket by walking locations in sorted order. That fills
each class from a *different* set of locations, so "animal vs empty" is
perfectly predictable from the background alone. A model trained on it learns
the background, and on a location-disjoint val split the shortcut inverts:

| sampling | val ROC-AUC |
| --- | --- |
| shipped v2 checkpoint | 0.509 (chance) |
| v3 trained on location-unmatched data | 0.204 (inverted) |

`choose()` now draws both classes round-robin from the same locations, and only
from locations that contain both. This is the root cause of the "blank filter
has a checkpoint problem" symptom in the audit — it was a data bug, not a
threshold bug.

## Sweep results

Grid sweep over every `(lo, hi)` pair, subject to `false_empty_rate <= 0.02`
(`scripts/calibrate_blank_v2.py`, `scripts/train_blank_v3.py`):

| checkpoint | ROC-AUC | band | false-empty | false-animal | review |
| --- | --- | --- | --- | --- | --- |
| v2 as shipped | 0.509 | 0.20 / 0.75 | 0.7% | 10.3% | 83.0% |
| v2 best achievable | 0.509 | 0.21 / 0.76 | 2.0% | 9.3% | 83.2% |
| **v3 (shipped)** | **0.787** | **0.07 / 0.77** | **2.0%** | **9.3%** | **64.8%** |

## Decision

Ship `blank_filter_v3.pth` (MobileNetV3-small, backbone unfrozen, camera-trap
augmentation on, 10 epochs, band `[0.07, 0.77]`). `BlankFilter` loads the newest
available checkpoint and falls back to v2 then v1, and takes its band from the
checkpoint rather than from a constant.

**The 20-30% review target is not met: the measured review rate is 64.8%.** At
AUC 0.787 a 2% false-empty ceiling forces a wide quarantine band, and no
`(lo, hi)` pair on this val split does better. Reaching 20-30% needs a stronger
detector (MegaDetector-class), not a different band. No band was fabricated to
hit the target.

What the change is worth on the demo set (`demo/ingest`, 6 tiger frames,
4 blanks, 2 borderline): v2 passed **0** animals through and auto-archived
3 real tiger frames as empty, so Re-ID never ran at all. v3 passes 5 of 6 tiger
frames, quarantines 1, and archives all 4 blanks plus both borderlines.
