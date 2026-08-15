# Status notes 3 — Aug 15, 2026 (19:30)

## Suites status
- blank: OK (fE=0.01, f1=0.76, PROMOTE v2)
- pipeline: OK
- perf: OK (blank 31.5 img/s p95 51ms)
- data: OK
- home: OK
- versioning: OK
- alerts: 12/12 PASS
- reid: FAIL/missing (needs fresh reid_benchmark.json after reid v2 training)

## Reid v2 training
- Loss=0.8 from epoch 1 (margin saturated = good separation). rank1=1.0 on 20-id probe (overfitting artifact of probe=train set).
- Let run 5-10 epochs then stop and run bench_reid.py for real metrics.
- bench_reid.py will produce reid_benchmark.json → reid suite passes.

## Next: run bench_reid after reid v2 training, then run --all for final report.
