#!/usr/bin/env bash
# Full RAGTruth run: shard feature extraction across the 3 GPUs, then train probes.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "[full] model=$MODEL  launching 3 shards across cuda:0,1,2 ..."
pids=()
for k in 0 1 2; do
  python code/extract_features.py --shard "$k" --nshards 3 --device "cuda:$k" \
    --model "$MODEL" --out "data/features_shard$k.jsonl" \
    > "data/extract_shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 1 ]; then echo "[full] a shard FAILED, see data/extract_shard*.log"; tail -n 5 data/extract_shard*.log; exit 1; fi
cat data/features_shard0.jsonl data/features_shard1.jsonl data/features_shard2.jsonl > data/features.jsonl
echo "[full] total feature records: $(wc -l < data/features.jsonl)"
echo "============ RESULTS ============"
python code/train_probe.py --features data/features.jsonl | tee data/results.txt
