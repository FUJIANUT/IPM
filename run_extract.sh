#!/usr/bin/env bash
# Generic sharded mech2 feature extraction. Usage: run_extract.sh <examples.jsonl> <tag>
set -e
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXAMPLES="$1"; TAG="$2"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"; DTYPE="${DTYPE:-float16}"
MAXLEN="${MAXLEN:-2048}"; NGPU="${NGPU:-3}"
pids=()
for k in $(seq 0 $((NGPU-1))); do
  python code/extract_features_mech2.py --examples "$EXAMPLES" --shard "$k" --nshards "$NGPU" \
    --device "cuda:$k" --model "$MODEL" --dtype "$DTYPE" --max_len "$MAXLEN" \
    --out "data/features_${TAG}_shard$k.jsonl" > "data/${TAG}_shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 1 ]; then echo "[extract] $TAG FAILED"; tail -n 6 data/${TAG}_shard*.log; exit 1; fi
cat data/features_${TAG}_shard*.jsonl > data/features_${TAG}.jsonl
echo "[extract] $TAG: $(wc -l < data/features_${TAG}.jsonl) records"
