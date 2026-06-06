#!/usr/bin/env bash
# Span/token-level run: extract per-token features+labels (3 GPUs) -> token probe.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "[tokens] extracting per-token features (3 shards) ..."
pids=()
for k in 0 1 2; do
  python code/extract_tokens.py --shard "$k" --nshards 3 --device "cuda:$k" --model "$MODEL" \
    --out "data/tokens_shard$k.jsonl" > "data/tokens_shard$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" = 1 ]; then echo "[tokens] shard FAILED"; tail -n 8 data/tokens_shard*.log; exit 1; fi
cat data/tokens_shard0.jsonl data/tokens_shard1.jsonl data/tokens_shard2.jsonl > data/tokens.jsonl
echo "[tokens] records: $(wc -l < data/tokens.jsonl)"
echo "================ TOKEN-LEVEL PROBE ================"
python code/probe_tokens.py --tokens data/tokens.jsonl | tee data/results_tokens.txt
echo "ALL_DONE" > data/tokens_DONE
