#!/usr/bin/env bash
# Phase 6: retrieval-quality robustness. Build gold/distractor stress set, extract, evaluate.
set -e
cd ~/cgp
source venv/bin/activate
echo "[phase6] building robustness stress set ..."
python code/prep_robustness.py --n 900
bash code/run_extract.sh data/robust_examples.jsonl robust
echo "==================== PHASE 6: ROBUSTNESS ===================="
python code/robustness_eval.py | tee data/results_phase6.txt
echo "PHASE6_DONE" > data/phase6_DONE
