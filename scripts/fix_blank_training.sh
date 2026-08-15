#!/bin/bash
# Kill all project training processes safely (single pkill invocation)
pkill -f "scripts/train_blank_v2.py"
pkill -f "scripts/bench_reid.py"
