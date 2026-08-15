# Training state log (Aug 15, 2026)

## Blank filter v2 (scripts/train_blank_v2.py) — RUNNING
- Config: MobileNetV3-small, frozen backbone, head-only AdamW LR=1e-4, BATCH 256, EPOCHS 20, num_workers=0.
- Epoch 1: f1=0.91 fE=0.007 (great); epoch 2 regressed (0.16/0.30); epoch 3: f1=1.0 fE=0.0; epoch 4: f1=0.47 fE=0.077.
- Oscillation likely from heavy augmentation noise + small batch stats on CPU. Monitor: if it converges to f1>0.8 fE<0.05 accept.
- Log: /tmp/blank_v2.log; checkpoint: models/blank_filter_v2.pth (only saved at end of 20 epochs).
- RISK: checkpoint only written after ALL epochs. If killed mid-way, pipeline suite fails (needs blank_filter_v2.pth).
- DECISION: add early-save (save best model each epoch).

## Re-ID benchmark (scripts/bench_reid.py) — FIXED, DONE
- Fixed broadcast error (min_dist returned 2D matrix; take .min(axis=1)) and mAP formula (divide by n_relevant).
- TASK2: rank1=0.694 rank5=0.860 mAP=0.278.
- TASK3 open-set currently BAD (auc=0.49, false_known=0.99) because tiger_reid.pth was only trained 1 epoch (val rank1 0.49) — need full triplet retraining.
- Output: models/reid_benchmark.json (regenerate after reid retraining).

## Re-ID model (scripts/train_reid.py) — NEEDS RETRAIN
- tiger_reid.pth: epoch 1, val rank1 0.49 (random-ish).
- Plan: run full training (18 epochs x 60 steps, triplet margin 0.6, BATCH 16, p_per_id=4).
- CPU: ~? per epoch; expect 2-4 h total. Background, monitor.
- Validation subset only 15 ids; consider larger probe set for reliable report.
- Log: /tmp/reid_train.log

## Alerts suite — FIXED, 12/12 PASS
## Memory: 3.9GB total, keep single training job + small suites.
