#!/usr/bin/env bash
# Quick end-to-end validation on a few dozen examples before the full run.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "[smoke] model=$MODEL  extracting 60 examples on cuda:0 ..."
python code/extract_features.py --limit 60 --device cuda:0 \
  --model "$MODEL" --out data/features_smoke.jsonl
echo "[smoke] feature record sample:"
head -n1 data/features_smoke.jsonl | python3 -m json.tool | head -40
echo "[smoke] probe (not meaningful at n=60, just checks the pipeline):"
python code/train_probe.py --features data/features_smoke.jsonl || true
