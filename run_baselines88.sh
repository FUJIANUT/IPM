#!/usr/bin/env bash
# Run GPU baselines on .88 (A5000), then the unified comparison (needs .234 scores already
# copied into ~/cgp/data/scores/).
set -e
cd ~/cgp
source venv/bin/activate
export TMPDIR="$PWD/tmp"; mkdir -p tmp
echo "[baselines88] installing lettucedetect..."
pip install --no-cache-dir --retries 10 --timeout 180 lettucedetect 2>&1 | tail -1 || echo "(lettucedetect install issue)"
echo "[baselines88] NLI (cuda:0)..."
python code/nli_baseline.py --device cuda:0 || echo "NLI failed"
echo "[baselines88] LettuceDetect..."
python code/lettucedetect_baseline.py || echo "LettuceDetect failed"
echo "================ FULL BASELINE COMPARISON ================"
python code/compare.py --ref ours
echo "BASELINES88_DONE" > data/baselines88_DONE
