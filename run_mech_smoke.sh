#!/usr/bin/env bash
# Validate the mechanistic extractor (attention + FFN capture) before the full run.
set -euo pipefail
cd ~/cgp
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "[mech-smoke] model=$MODEL  extracting 40 examples on cuda:0 ..."
python code/extract_features_mech.py --limit 40 --device cuda:0 \
  --model "$MODEL" --out data/features_mech_smoke.jsonl
echo "[mech-smoke] feature keys + sample mech values:"
head -n1 data/features_mech_smoke.jsonl | python3 -c "
import sys,json
d=json.loads(sys.stdin.read()); f=d['features']
print('n_resp_tok', d['n_resp_tok'])
print('mech keys:', [k for k in sorted(f) if k.startswith(('mech_','cmech_'))])
print('attn2ctx_mean=%.4f ffnnorm_mean=%.3f ffnratio_mean=%.3f' %
      (f['mech_attn2ctx_mean'], f['mech_ffnnorm_mean'], f['mech_ffnratio_mean']))
"
