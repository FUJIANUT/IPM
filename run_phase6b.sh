#!/usr/bin/env bash
# Phase 6 v2: degradation gradient robustness. Waits for Phase 3b to free the GPUs first.
set -e
cd ~/cgp
source venv/bin/activate
echo "[p6b] waiting for Phase 3b to finish (free GPUs)..."
for i in $(seq 1 160); do [ -f data/phase3b_DONE ] && break; sleep 15; done
python code/prep_robustness.py --n 700
bash code/run_extract.sh data/robust_examples.jsonl robust
echo "==================== PHASE 6 v2: DEGRADATION GRADIENT ===================="
python code/robustness_eval.py | tee data/results_phase6b.txt
echo "PHASE6B_DONE" > data/phase6b_DONE
