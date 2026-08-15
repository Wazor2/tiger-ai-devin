# Live demo runbook — Pench forest-department console

Everything the demo needs is in this repo; there is no build step and no
external service. Measured numbers below come from this machine (CPU-only
torch 2.5.1, MobileNetV3 blank filter + ResNet50 Re-ID).

## 1. Boot

```powershell
# phone as webcam: start Iriun Webcam on the phone AND Iriun on the PC first
.venv\Scripts\python.exe -m pench.live_capture --probe      # find the index
.venv\Scripts\python.exe -m pench.server --source webcam --device 1 `
    --inference models --reseed --port 8000
```

Fallbacks, in order of preference if the phone link misbehaves:

```powershell
# 1. replay the rehearsal stills as if they were camera frames
.venv\Scripts\python.exe -m pench.server --source folder --folder demo/rehearsal `
    --inference models --hold 6 --reseed --port 8000
# 2. plumbing only, no models, no camera (never fails)
.venv\Scripts\python.exe -m pench.server --source dummy --inference dummy --port 8000
```

Dashboard: <http://127.0.0.1:8000/> (redirects to `/static/index.html`).

`--reseed` rewrites the synthetic 60-day baseline. **Always boot the real demo
with `--reseed`**: it guarantees the alert feed shows the one story the demo is
about (tiger 252's core shift) instead of leftovers from rehearsal runs.

## 2. Demo script — three beats, in this order

Hold each card steady in front of the phone for ~6 s. Cards are in
`demo/demo_cards/` (`demo/rehearsal/` is the same three, numbered for replay).

| # | Card | What the console must show | Talking point |
|---|------|---------------------------|---------------|
| 1 | `known_tiger_172.jpg` | `ANIMAL` 0.970 → identity **T-172**, distance 0.0027 < 0.0133, logged to history | "known individual, auto-confirmed, home range updates" |
| 2 | `empty_jungle.jpg` | `EMPTY` 0.009, no identity, nothing written to history | "the blank filter is what makes 100k images/season tractable" |
| 3 | `unknown_tiger.jpg` | `ANIMAL` 0.993 → **UNIDENTIFIED**, nearest catalogue distance 0.175, queued for human review | "it refuses to guess; a miscalibrated threshold would have called this a known tiger" |

Distances are small because the retrained Re-ID model packs identities tightly;
what matters on screen is the gap between beat 1 (0.0027, ~65x inside the
confirm threshold) and beat 3 (0.175, ~13x outside it). See
`docs/reid_baseline.md` for the calibration.

Then point at the alert panel: `Core Shift · T-252`, escalated — the seeded
history has 252's recent detections ~8 km east of its 60-day core, which is the
kind of thing a range officer should be told about.

## 3. Rehearsal / smoke check

```powershell
.venv\Scripts\python.exe -m pench.server --source folder --folder demo/rehearsal `
    --inference models --hold 6 --reseed --port 8020
.venv\Scripts\python.exe scripts\rehearse_demo.py --port 8020
```

The harness polls `/api/latest` like the dashboard does and exits non-zero if
any beat classifies differently than the table above. Latest run:

```
PASS 01_known_tiger_172.jpg: animal 0.9699 known_identity/172 dist=0.0027 pipeline=94.3ms
PASS 02_empty_jungle.jpg   : empty  0.0085 pipeline=58.0ms
PASS 03_unknown_tiger.jpg  : animal 0.9932 human_review/154 dist=0.1746 pipeline=96.6ms
alerts: 252/core_shift/escalated        total 23.5s for 3/3 beats
```

After any `scripts/build_demo_cards.py` run, refresh the replay copies
(`demo/rehearsal/01..03`) from `demo/demo_cards/` before rehearsing, or the
harness scores last week's cards.

Timing budget: capture → frame written → both models → history/alerts recomputed
is **50–100 ms**; the dashboard polls every 2 s, so a capture is on screen in
well under the 5 s the spec asks for. The remaining wait is the capture
trigger itself (motion, or the 5 s forced interval).

## 4. Filming a screen (what we actually tested)

`scripts/simulate_screen_capture.py` re-runs the cards through blur, brightness,
contrast and JPEG degradation standing in for a phone filming a monitor:

- tiger 172 stays `known_identity` under clean, typical and harsh conditions — **use 172 for the live beat**
- the empty frame stays `empty` in all three
- the unknown tiger stays `human_review` in all three
- tiger 85 does not confirm reliably, and 252 degrades under harsh conditions — keep both off camera and let the seeded history carry 252's alert

Practical: pause the video on a clean frame, kill glare, hold the phone steady,
fill the frame with the tiger (a small tiger in a large frame loses the match).

## 5. If something breaks on stage

| Symptom | Fix |
|---------|-----|
| no webcam index found | Iriun PC client not running, or phone on a different Wi-Fi; fall back to `--source folder` |
| frames arrive but nothing classifies | check `/api/status` for `blank_model`; models load lazily on the first capture |
| alert feed is noisy | restart with `--reseed` |
| dashboard blank | check `/api/latest` directly; endpoints return 503 only before the capture loop starts |
| everything is on fire | `--source dummy --inference dummy` still shows the full console with synthetic data |

## 6. What is real vs. staged

Real: both models and their calibrated thresholds, the classification, the
Re-ID distances and the accept/reject decision, home-range geometry, the alert
logic and its debounce.

Staged: the 60-day detection history is synthetic (`LiveStore.seed_baseline`) —
there is no historical Pench camera-trap archive on this machine — and every
live frame is attributed to station S01 at 21.6853, 79.2520 because there is
one camera. The movement-speed threshold is raised to 20 km/day for the demo
(`LIVE_MOVEMENT_SPEED_KM_PER_DAY`); the module default of 3 km/day flags normal
tiger movement. Say this if asked; it is a better answer than pretending.
