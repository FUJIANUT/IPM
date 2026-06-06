#!/usr/bin/env bash
# B-step full run: mech2 extraction (3 GPUs) -> standard probe + copy-head probe.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TAG="${TAG:-mech2}"
MAXLEN="${MAXLEN:-2048}"
DTYPE="${DTYPE:-float16}"
NGPU="${NGPU:-3}"
echo "[mech2-full] model=$MODEL tag=$TAG dtype=$DTYPE  launching $NGPU shards ..."
pids=()
for k in $(seq 0 $((NGPU-1))); do
  python code/extract_features_mech2.py --shard "$k" --nshards "$NGPU" --device "cuda:$k" \
    --model "$MODEL" --max_len "$MAXLEN" --dtype "$DTYPE" --out "data/features_${TAG}_shard$k.jsonl" \
    > "data/${TAG}_shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 1 ]; then echo "[mech2-full] a shard FAILED"; tail -n 8 data/${TAG}_shard*.log; exit 1; fi
cat data/features_${TAG}_shard*.jsonl > data/features_${TAG}.jsonl
echo "[mech2-full] total: $(wc -l < data/features_${TAG}.jsonl)"
echo "================ STANDARD PROBE ================"
python code/train_probe.py --features data/features_${TAG}.jsonl | tee data/results_${TAG}.txt
echo "================ COPY-HEAD PROBE ================"
python code/probe_copyheads.py --features data/features_${TAG}.jsonl --topk 24 | tee -a data/results_${TAG}.txt
echo "ALL_DONE" > "data/${TAG}_DONE"
