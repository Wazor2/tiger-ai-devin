# Pench Tiger Reserve — Camera-Trap Triage & Tiger Movement Intelligence

An end-to-end pipeline that turns raw camera-trap images into conservation intelligence: blank-frame filtering, individual tiger identification from stripe patterns, home-range estimation, and deviation alerting.

## Quickstart

```bash
# 1. PyTorch first — pick the wheel for your box (see requirements.txt)
pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
# ...or --index-url https://download.pytorch.org/whl/cu121 on a CUDA box

# 2. Everything else, pinned
pip3 install -r requirements.txt          # add -r requirements-dev.txt to run the tests

# 3. Run the full synthetic demo (generates sample frames, runs all 4 modules, draws the map)
python3 -m pench.demo_run

# 4. Run on your own camera-trap images
python3 -m pench.pipeline --ingest /path/to/frames --output ./run_output

# 5. Live console (phone camera via Iriun, or stills replay) — docs/live_demo_runbook.md
python3 -m pench.server --source webcam --device 0 --inference models
```

Inference runs on CUDA automatically when it is available and falls back to CPU;
set `PENCH_DEVICE=cpu` to force CPU on a GPU box.

Demo outputs land in `demo/output/`: `pipeline_report.json`, `home_range_map.png`, `detection_history.csv`, and a `quarantine/` folder of frames needing human review.

## The Four Modules

| # | Module | What it does |
|---|---|---|
| 1 | Blank filtering | MobileNetV3 classifies frames; confident blanks are archived, confident animals pass through, and ambiguous frames go to a quarantine folder for human review (never deleted) |
| 2 | Tiger Re-ID | EfficientNet-B0 produces a 128-d stripe-pattern embedding; cosine search against a 107-identity catalogue (trained on the ATRW dataset) auto-confirms, flags for review, or auto-enrolls new tigers |
| 3 | Home range | Regenerated fresh each run: Minimum Convex Polygon, 50% core area, 95% KDE contour, activity centroid, and pairwise territorial overlap (UTM 44N) |
| 4 | Alerting | Compares the last 30 days against a 60-day baseline; fires range-expansion, core-shift, and absence alerts only after artefact filtering (low confidence, camera downtime, minimum detection counts) |

## Project Structure

```
pench/           pipeline, model serving, demo, and spatial/alert modules
models/          trained checkpoints + identity catalogue
scripts/         training and dataset-preparation scripts
datasets/        ATRW (tiger Re-ID) and Caltech Camera Traps subsets
demo/output/     demo results and map
```

See `TECHNICAL_DOCUMENTATION.md` for full details on datasets, architecture, training results, and production recommendations.
