#!/usr/bin/env bash
# E-step: build HaluEval-QA, extract mech2 features (3 GPUs), run zero-shot transfer probe.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ROWS="${ROWS:-4000}"
python code/prep_haluqa.py --max_rows "$ROWS"
echo "[haluqa] extracting features (3 shards) ..."
pids=()
for k in 0 1 2; do
  python code/extract_features_mech2.py --examples data/haluqa_examples.jsonl \
    --shard "$k" --nshards 3 --device "cuda:$k" --model "$MODEL" \
    --out "data/features_haluqa_shard$k.jsonl" > "data/haluqa_shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 1 ]; then echo "[haluqa] shard FAILED"; tail -n 8 data/haluqa_shard*.log; exit 1; fi
cat data/features_haluqa_shard0.jsonl data/features_haluqa_shard1.jsonl data/features_haluqa_shard2.jsonl > data/features_haluqa.jsonl
echo "[haluqa] records: $(wc -l < data/features_haluqa.jsonl)"
echo "================ CROSS-DATASET TRANSFER (RAGTruth -> HaluEval-QA) ================"
python code/train_transfer.py --train data/features_mech2.jsonl --test data/features_haluqa.jsonl --topk 64 | tee data/results_transfer.txt
echo "ALL_DONE" > data/haluqa_DONE
