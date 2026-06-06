#!/usr/bin/env bash
# Validate mech2: Data2txt context detection + per-head output, sampled across ALL tasks.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "[mech2-smoke] ~590 examples spread across the dataset (nshards 30, shard 7) on cuda:0 ..."
python code/extract_features_mech2.py --nshards 30 --shard 7 --device cuda:0 \
  --model "$MODEL" --out data/features_mech2_smoke.jsonl
python3 -c "
import json
from collections import Counter
recs=[json.loads(l) for l in open('data/features_mech2_smoke.jsonl')]
print('n =', len(recs))
print('by task:', dict(Counter(r['task_type'] for r in recs)))
tasks=set(r['task_type'] for r in recs)
print('ctx_found rate by task:', {t: round(sum(r['ctx_found'] for r in recs if r['task_type']==t)/max(1,sum(1 for r in recs if r['task_type']==t)),3) for t in tasks})
print('head_a2c length:', len(recs[0]['head_a2c']), '(= nlayers*nheads =', recs[0]['nlayers']*recs[0]['nheads'], ')')
print('mech_attn2ctx_mean sample:', round(recs[0]['features']['mech_attn2ctx_mean'],4))
"
