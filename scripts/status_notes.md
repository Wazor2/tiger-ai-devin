# Status notes — Aug 15

Blank v2 training (main-1): epochs 1-5 done. Disk checkpoint = best epoch (epoch 1: f1 0.91, fE 0.0067). Oscillating metrics are due to fixed 0.15/0.85 thresholds during training + heavy augmentation; the saved checkpoint on disk is the best model. Final calibration + full report written at end of epoch 20. If killed mid-run, v2.pth still = best-so-far.

Re-ID triplet training (main-1): epoch1 rank1=0.5228; epoch4 rank1=0.3686 (regressed — triplet loss saturates at 0.01, embeddings collapsing). 18 epochs total. If final rank1 < 0.5, retrain with higher margin (0.8-1.0) and semi-hard mining.

Suites passing so far: alerts 12/12, data OK, home OK, versioning OK.
perf and pipeline FAIL until blank_filter_v2.pth exists (training creates it).
reid suite needs fresh models/reid_benchmark.json after reid retraining.
