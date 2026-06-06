#!/usr/bin/env bash
# Phase 3b: add RAGBench (natural, labeled) + FaithEval (counterfactual conflict) to the
# cross-dataset transfer matrix.
set -e
cd ~/cgp
source venv/bin/activate
echo "[p3b] RAGBench prep + extract ..."
python code/prep_ragbench.py --per_config 800
bash code/run_extract.sh data/ragbench_examples.jsonl ds_ragbench
echo "[p3b] FaithEval prep + extract ..."
python code/prep_faitheval.py --max 2000
bash code/run_extract.sh data/faitheval_examples.jsonl ds_faitheval
echo "[p3b] assembling + 6-dataset matrix ..."
cp data/features_mech2.jsonl data/features_ds_ragtruth.jsonl
cp data/features_haluqa.jsonl data/features_ds_haluqa.jsonl
echo "==================== EXPANDED CROSS-DATASET MATRIX ===================="
python code/transfer_matrix.py | tee data/results_phase3b.txt
echo "PHASE3B_DONE" > data/phase3b_DONE
