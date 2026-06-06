#!/usr/bin/env bash
# Phase 2: proxy-family generality. Extract mech2 features + probe for several proxy LMs,
# then summarize. Waits for NLI to free cuda:0 first (LettuceDetect runs on CPU).
set -e
cd ~/cgp
echo "[phase2] waiting for NLI (cuda:0) to finish before grabbing GPUs..."
for i in $(seq 1 120); do [ -f data/nli_DONE ] && break; sleep 15; done
PROXIES="qwen05:Qwen/Qwen2.5-0.5B-Instruct qwen15:Qwen/Qwen2.5-1.5B-Instruct smol17:HuggingFaceTB/SmolLM2-1.7B-Instruct tiny11:TinyLlama/TinyLlama-1.1B-Chat-v1.0"
for spec in $PROXIES; do
  tag="p2_${spec%%:*}"; model="${spec#*:}"
  echo "==================== Phase2 extract: $tag ($model) ===================="
  MODEL="$model" TAG="$tag" DTYPE=bfloat16 MAXLEN=1536 NGPU=3 bash code/run_mech2_full.sh 2>&1 | tail -3
done
echo "==================== PHASE 2 SUMMARY (proxy generality) ===================="
python code/phase2_summary.py | tee data/results_phase2.txt
echo "PHASE2_DONE" > data/phase2_DONE
