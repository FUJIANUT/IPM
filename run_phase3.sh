#!/usr/bin/env bash
# Phase 3: cross-dataset transfer matrix. Add HaluEval Summarization + Dialogue (QA + RAGTruth
# already extracted), then build the train-X-test-Y AUROC matrix.
set -e
cd ~/cgp
source venv/bin/activate
echo "[phase3] HaluEval Summarization ..."
python code/prep_halueval.py --task summarization --out data/halusum_examples.jsonl --max_rows 4000
bash code/run_extract.sh data/halusum_examples.jsonl ds_halusum
echo "[phase3] HaluEval Dialogue ..."
python code/prep_halueval.py --task dialogue --out data/haludia_examples.jsonl --max_rows 4000
bash code/run_extract.sh data/haludia_examples.jsonl ds_haludia
echo "[phase3] assembling matrix inputs ..."
cp data/features_mech2.jsonl data/features_ds_ragtruth.jsonl
cp data/features_haluqa.jsonl data/features_ds_haluqa.jsonl
echo "==================== CROSS-DATASET TRANSFER MATRIX ===================="
python code/transfer_matrix.py | tee data/results_phase3.txt
echo "PHASE3_DONE" > data/phase3_DONE
