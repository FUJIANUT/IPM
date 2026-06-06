#!/usr/bin/env bash
# Full run with mechanistic features: shard across 3 GPUs, concat, probe.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TAG="${TAG:-$(basename "$MODEL")}"
MAXLEN="${MAXLEN:-2048}"
DTYPE="${DTYPE:-float16}"
echo "[mech-full] model=$MODEL tag=$TAG maxlen=$MAXLEN dtype=$DTYPE  launching 3 shards ..."
pids=()
for k in 0 1 2; do
  python code/extract_features_mech.py --shard "$k" --nshards 3 --device "cuda:$k" \
    --model "$MODEL" --max_len "$MAXLEN" --dtype "$DTYPE" --out "data/features_mech_${TAG}_shard$k.jsonl" \
    > "data/mech_${TAG}_shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 1 ]; then echo "[mech-full] a shard FAILED"; tail -n 8 data/mech_${TAG}_shard*.log; exit 1; fi
cat data/features_mech_${TAG}_shard0.jsonl data/features_mech_${TAG}_shard1.jsonl data/features_mech_${TAG}_shard2.jsonl > data/features_mech_${TAG}.jsonl
echo "[mech-full] tag=$TAG total feature records: $(wc -l < data/features_mech_${TAG}.jsonl)"
echo "============ RESULTS (logit + mechanistic, $TAG) ============"
python code/train_probe.py --features data/features_mech_${TAG}.jsonl | tee data/results_mech_${TAG}.txt
