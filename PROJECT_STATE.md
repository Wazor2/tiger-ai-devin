# PROJECT STATE (consolidated)

## Environment
- Sandbox: CPU-only Python 3.11, torch CPU installed, shapely/scikit-learn/faiss-cpu/numpy/scipy/matplotlib/pillow OK.
- Working dir: /home/ubuntu/project (root of deliverable).

## Audit (new validation task)
- pench/: model_serving.py, pipeline.py, demo_run.py; modules/occupancy.py, alerting.py; __init__.py.
- models/: blank_filter.pth (BAD per user — 32.6% exact acc; DO NOT overwrite), blank_filter_report.json, tiger_reid.pth (efficientnet_b0 triplet embed 128), reid_centroids.json (107 ids plain lists).
- ATRW at datasets/atrw/: train/ (1887 imgs w/ ids), test/ (1764 imgs, NO id labels — column 1 empty!), reid_keypoints_train.json, reid_keypoints_test.json, reid_list_train.csv (id,filename), reid_list_test.csv (filename only). NOTE: train ids are filenames like '000001.jpg'? NO — reid_list_train.csv has `250,003597.jpg` = id 250, file 003597.jpg. Test list has only filenames.
- CCT subset: cct_subset/train/{animal,empty} (1200/1200?), val/ (300/300), val_local, val_official.

## New task (pasted_content.txt): 10 tasks, deliver final Module|Old|New|Target|Pass/Fail table.
1 Retrain blank filter: balanced + augmentation, GPU when available (sandbox CPU fallback), versioned checkpoints (blank_filter_v2.pth), calibrated 3-way thresholds via validation set, full metrics incl ROC-AUC/PR-AUC/false-empty rate. Primary: minimize false-empty.
2 Re-ID benchmark vs ATRW mapping: Rank-1/5, mAP, CMC, per-id acc, same/diff distance distributions, confusion pairs; validate CONFIRM_DIST=0.55/ENROLL_DIST=0.95 with top1/top2 margin + calibrated confidence.
3 Open-set Re-ID: hold-out identities (e.g. from train), measure known acceptance, unknown rejection, false-known, false-new, open-set ROC/PR.
4 Data validation: ATRW filename->identity mapping, duplicates, corrupt, leakage (train/test id overlap), balance, missing, invalid labels; fail loudly.
5 Home range safety: min detections, coord validation, outlier handling, configurable windows, bandwidth validation, 50% core + 95% ranges, uncertainty metadata; keep existing MCP/KDE.
6 Alerting unit tests (deterministic): core_shift, territorial_overlap, new_identity, unusual movement, activity anomaly; debounce/persistence; full alert fields (tiger_id, timestamp/window, evidence, metric/value, threshold, severity, confidence, explanation).
7 Pipeline safety: animal never silently discarded, ambiguous->review, quarantine preserved w/ metadata.
8 Automated tests + `python -m pench.evaluate --all` -> JSON report + console summary.
9 Model versioning: never overwrite; store version/dataset/date/hyperparams/metrics/thresholds/device; promote only if beats on safety metrics.
10 Perf benchmark: CPU (here)/CUDA (user), batch inference, imgs/sec, avg/p95 latency, VRAM/RAM, RTX3050-4GB compat.

## DEBUG NOTES (degenerate geometry bug — just fixed, VERIFY)
- Root cause of safe_home_range failures: (1) np.trapz removed in numpy 2.x -> replaced with np.trapezoid (done, sed across scripts+pench); (2) MultiPoint(np.array) fails in shapely 2.1.2 -> use MultiPoint([tuple(p) for p in arr]); (3) all-identical-location points (collinear) -> ConvexHull fails; jitter fallback added in _hull_vertices; degenerate cases now return quality=degenerate_geometry with reason. JUST EDITED occupancy_safe.py: dedup hull verts, guard geom_type=='Point'/'Polygon', peel loop fixed. VERIFY with python3 -u scripts/probe_alerts.py then python3 -u -m pench.evaluate alerts (expect 11/11 pass).
- probe scripts: scripts/probe_alerts.py, scripts/probe2.py (debug probes, delete later)

## Current build state (validation task)
- scripts/train_blank_v2.py RUNNING in bg (log /tmp/blank_v2.log): train=5187 (2797 animal/2390 empty — NOTE more animal in cct_subset/train; class weights applied), val=600, 20 epochs, saves models/blank_filter_v2.pth + v2_report.json with calibrated lo/hi. EXPECT: ~15-20 min.
- scripts/bench_reid.py STARTED in bg (log /tmp/reid_bench.log): runs task4 validation + task2 closed-set (Rank-1/5/10, mAP, CMC, per-id acc, confusion pairs, same/diff dists, threshold sweep vs CONFIRM 0.55/ENROLL 0.95) + task3 open-set (holdout ids >=5 samples, acceptance/rejection rates, ROC/PR). Outputs models/reid_benchmark.json. NOTE reid_list_test.csv has NO id labels; closed-set uses train images probe-vs-gallery (excludes self).
- pench/modules/occupancy_safe.py DONE (safe_home_range: min_detections=5, coord validation lat 21-22.5 lon 78.5-80.5, IQR outlier filter, bw clamping 0.5-5km, uncertainty metadata).
- pench/modules/alerting_v2.py DONE with 5 alert types (core_shift, territorial_overlap, new_identity, unusual_movement, activity_anomaly) + apply_debounce + full Alert fields. FIX NEEDED: __main__ smoke test fails when run as -m (relative import from occupancy) — need absolute import fallback; functionality itself OK. Alert config: confirmation_windows=2, overlap 50km2, speed 3km/day, z=2.
- Task 5-6 scripts complete; Task 7 pipeline safety to be verified in evaluate suite; Task 8 pench/evaluate.py next; Task 9 versioning handled by v2 checkpoint metadata; Task 10 perf benchmark pending (write pench_perf_bench.py using model_serving classes).

## Plan status
Phase 1 (audit/infra): DONE
Phase 2: Task 1 blank filter retrain — RUNNING (verify v2 beats v1 on false-empty + F1)
Phase 3: Tasks 2-3 Re-ID — RUNNING (check reid_benchmark.json)
Phase 4: Task 4 data validation — RUNNING (part of bench_reid.py)
Phase 5: Tasks 5-7 — occupancy_safe + alerting_v2 written; pipeline safety check in evaluate
Phase 6: Tasks 8-10 (pench.evaluate module) — NEXT
Phase 7: Final report with Module|Old|New|Target|Pass/Fail table

## Old state (delivered earlier)
- All 4 modules built+trained, demo works (demo/output/{pipeline_report.json,home_range_map.png,detection_history.csv,quarantine}).
- README.md + TECHNICAL_DOCUMENTATION.md written and delivered. Zips delivered: pench_tiger_system.zip, atrw_dataset.zip, cct_training_data_windows.zip.
- Old model metrics: blank v1 val 70.5% (local) / 61.1% (official held-out); Re-ID rank-1 49.4% on 15-id probe (not ATRW test).
