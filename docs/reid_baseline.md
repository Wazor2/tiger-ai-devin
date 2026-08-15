# Re-ID: honest baseline and augmentation provenance

Fix-spec items 6 and 7. Two questions: what does the shipped Re-ID model actually
score under a leak-free protocol, and was it trained with augmentation or not.

## 1. The old benchmark leaked

`scripts/bench_reid.py` embeds every labelled ATRW image, uses **all** of them as
gallery vectors, and only blanks the exact query on the diagonal. A query is
therefore matched against a gallery that contains its own near-duplicates from
the same burst, and (in the centroid variant) against a centroid it helped
compute. Its Rank-1 76.4% / open-set ROC-AUC 0.820 are upper bounds, not
field expectations.

`scripts/bench_reid_strict.py` replaces it:

* every identity with >= 4 images is split in half — gallery half builds the
  centroid, probe half is only ever a query, so no query touches its own centroid;
* 25% of eligible identities (26 of 107) are held **entirely** out of the
  catalogue and used as unknown probes, which is what the field case actually is;
* the split is seeded (`SEED = 17`) so numbers are reproducible;
* output: `models/reid_benchmark_strict.json` (full threshold sweep included).

```
python scripts/bench_reid_strict.py                  # scores models/tiger_reid.pth
python scripts/bench_reid_strict.py --ckpt models/tiger_reid_v3.pth
```

## 2. Strict numbers for the old checkpoint (epoch-1 weights)

81 catalogue identities, 761 known probes, 398 unknown probes, CPU. Recorded in
`models/reid_benchmark_strict_epoch1.json`; superseded by section 5.

| metric | old (leaky) | strict |
| --- | --- | --- |
| Rank-1 | 0.7642 | **0.6846** |
| Rank-5 | 0.8980 | **0.8752** |
| Rank-10 | — | 0.9251 |
| open-set ROC-AUC | 0.8197 | **0.7310** |

At the calibrated `CONFIRM_DIST = 0.316`: known acceptance **47.8%**, unknown
rejection **87.7%**. Known-probe min distance averages 0.314, unknown-probe min
distance 0.368 — the two populations overlap heavily, which is the real ceiling
here, not the threshold.

The strict sweep's own best (Youden) threshold is **0.3268** — within 0.011 of
the 0.316 already in `pench/model_serving.py`. The P0 recalibration survives an
honest protocol; only the *expected accuracy at* that threshold moves. Read it
as: roughly half of true revisits auto-confirm and the rest fall to human review,
and ~1 in 8 genuinely new tigers is wrongly auto-confirmed as a known one. That
is the direction the safety argument wants (missing a match costs a review click;
a false merge corrupts the catalogue), but it is not the 74%/78% the old report
implied, and the demo script should not claim it.

Section 5 replaces these numbers: they are the pre-fix baseline that the
retrained checkpoint had to beat.

No identity in the catalogue has fewer than 10 images, so weak-identity handling
(fix-spec item 11) is not what is limiting this split. 1 of 81 identities scores
Rank-1 0.0.

## 3. Augmentation provenance

The `augmentation: "none"` in `report/evaluation_report.json` is **blank-filter
v2 metadata**, written by `scripts/restore_blank_ckpt.py`, which deliberately
stripped augmentation "for speed and stability". It says nothing about Re-ID.
That checkpoint is already superseded by `blank_filter_v3.pth`
(camera-trap augmentation, backbone unfrozen — see `docs/blank_filter_decision.md`).

Re-ID augmentation was never missing: `scripts/train_reid_v2.py` trains through
`RandomHorizontalFlip`, `RandomCrop(padding=14)`, `ColorJitter` and
`RandomAffine`. The checkpoint simply did not record that, which is what made it
look suspicious; it now saves `augmentation` and `val_protocol` fields.

**The real defects in that script were the loss and the validation, not the
augmentation.**

### 3a. The triplet loss had no gradient

`SemiHardTripletLoss.forward()` built its masks the wrong way round twice:

```python
pos  = sim.masked_fill(~same, -1.0).min(dim=1)[0]   # non-positives -> -1, then min
neg  = sim.masked_fill(~diff,  1.0)                 # self + positives -> 1.0
semi = neg.masked_fill(neg > pos + margin, -1.0).max(dim=1)[0]
```

The first line fills every non-positive cell with `-1` and then takes the **min**
of the row, so `pos` is `-1.0` for every anchor no matter what the embeddings are.
The second makes the anchor's own cell (similarity 1.0) and its positives the
most attractive "negatives" available. Together they pin
`semi - pos` at a constant, so the loss printed exactly `0.8000` — the margin —
every epoch, with **zero gradient**. Training moved no weights at all: the strict
Rank-1 of 0.6846 in section 2 is what ImageNet-pretrained EfficientNet-B0
features score with an untrained projection head.

Fixed: hardest positive fills non-positives with `+2.0` before `min`; negatives
mask out self *and* positives with `-2.0`; semi-hard picks the hardest negative
still easier than the positive and falls back to the hardest negative otherwise;
loss is `relu(margin - (pos - neg))`. `tests/test_reid_training.py` pins that the
loss ranks a separated batch below a mixed one, that it produces a non-zero
gradient, and that a collapsed batch costs exactly the margin.

With the fix, loss falls (0.78 -> 0.42 over 8 epochs) and strict val rank-1 rises
0.7467 -> 0.8233 instead of standing still.

### 3b. Validation could not select a best epoch

`val_rank1()` averaged every image of an identity into a centroid and then probed
with that same average, so rank-1 was 1.0 by construction. Epoch 1 hit the
ceiling, nothing could ever beat it, early stopping never fired, and
`models/tiger_reid.pth` is consequently **epoch-1 weights** (`val_rank1: 1.0` in
its metadata) — and per 3a those weights never moved anyway. Between them, these
two bugs explain the modest strict Rank-1 far better than any augmentation
setting does.

Also fixed in `scripts/train_reid_v2.py`:

* `val_rank1()` now holds out a probe half per identity, same protocol as the
  strict benchmark (smoke run: 0.7533 -> 0.7700 across two epochs, i.e. the
  metric moves and best-epoch selection works);
* runs write `models/tiger_reid_v3.pth` / `reid_centroids_v3.json`, **never** over
  the serving checkpoint, so a retrain can't break a rehearsed demo;
* `--epochs / --iters / --time-budget` for bounded runs, CUDA-aware device
  (`resolve_device`), repo-relative dataset path (`ATRW_DIR` to override),
  and augmentation/protocol/device recorded in the checkpoint.

## 4. Promotion procedure

A candidate replaces the serving model only if all four hold:

1. `bench_reid_strict.py --ckpt models/tiger_reid_v3.pth` beats Rank-1 0.6846
   **and** ROC-AUC 0.7310;
2. `CONFIRM_DIST` is re-derived from the candidate's own sweep (its best-Youden
   threshold), not inherited;
3. `scripts/rehearse_demo.py` still passes all three beats — known T-172
   auto-confirmed, empty frame filtered, unknown tiger held for review — after
   `scripts/build_demo_cards.py` is re-run against the new catalogue;
4. the copy into `models/tiger_reid.pth` + `models/reid_centroids.json` is a
   separate, reviewable commit.

## 5. What is actually being served now

A 30-epoch run with the fixed loss (`--time-budget 5400`, 899 s on CPU) selected
epoch 28 at strict val rank-1 0.8800, and it clears the gate:

| metric | epoch-1 (was served) | epoch-28 (now served) |
| --- | --- | --- |
| strict Rank-1 | 0.6846 | **0.9120** |
| strict Rank-5 | 0.8752 | **0.9947** |
| strict mAP | 0.7692 | **0.9510** |
| open-set ROC-AUC | 0.7310 | **0.8060** |

Gate steps executed:

1. `bench_reid_strict.py --ckpt models/tiger_reid_v3.pth` beat both gate metrics
   (`models/reid_benchmark_strict.json`; the old result is kept as
   `reid_benchmark_strict_epoch1.json`);
2. **`CONFIRM_DIST` re-derived: 0.316 -> 0.0133.** This matters more than the
   headline metrics. The retrained model packs identities far more tightly, so the
   whole distance scale shrank (known-probe min distance now averages 0.016, not
   0.314) and the old 0.316 auto-confirms *100%* of unknown tigers — exactly the
   failure the P0 recalibration removed. 0.0133 gives 56.8% known-acceptance at
   83.7% unknown-rejection, dominating the old checkpoint's best achievable point
   (55.3% / 82.9%). `AlertConfigV2.new_identity_min_distance` mirrors it, and
   `tests/test_p0.py` now asserts `CONFIRM_DIST` sits on the measured sweep with
   rejection >= 0.80 instead of hardcoding a number that silently rots whenever
   the checkpoint changes;
3. cards rebuilt and the rehearsal re-run against the live API: all three beats
   pass — T-172 auto-confirmed at distance **0.0027** (was 0.2474), empty frame
   filtered at 0.0085, unknown tiger held for review at 0.1746 — and the intended
   T-252 `core_shift` alert still escalates. Pipeline latency 58-97 ms/frame;
4. the checkpoint/catalogue swap is its own commit.

`ENROLL_DIST` stays at 0.95, i.e. auto-enrol remains unreachable by design; see
the comment in `pench/model_serving.py` for why 0.08 was rejected.

One demo-content change came out of this: `pick_unknown()` in
`scripts/build_demo_cards.py` now requires the blank filter to call the candidate
an animal before maximising catalogue distance. The furthest-away test image under
the new model sat in the blank filter's review band (0.57), which muddles the beat
— "unknown tiger" should be unambiguously a tiger the catalogue cannot name. The
selected card now scores 0.9932 animal.

### Remaining known gap

Open-set separation is still the weak axis: ROC-AUC 0.806 means ~16% of genuinely
new tigers are auto-confirmed as known at the operating point. The fix-spec items
aimed at exactly this (ArcFace/CosFace margin loss, hard-negative mining, higher
input resolution) are untouched and are where further accuracy work should go.
Closed-set ranking at 0.912 Rank-1 / 0.995 Rank-5 is no longer the bottleneck.
