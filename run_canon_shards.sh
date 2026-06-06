#!/usr/bin/env bash
cd ~/cgp || exit 1
pkill -f canonical_selfcheck.py 2>/dev/null
sleep 3
rm -f data/canon_s0_DONE data/canon_s1_DONE data/scores/canonical_s0.jsonl data/scores/canonical_s1.jsonl

run_shard () {
  local sh=$1 st=$2 en=$3
  nohup setsid bash -c "source ~/cgp/venv/bin/activate; cd ~/cgp; export HF_HUB_OFFLINE=1; python code/canonical_selfcheck.py --device cuda:${sh} --limit 240 --start ${st} --end ${en} --n 3 --max_new_tokens 110 --out data/scores/canonical_s${sh}.jsonl > data/run_canon_s${sh}.log 2>&1; echo done > data/canon_s${sh}_DONE" </dev/null >/dev/null 2>&1 &
}
run_shard 0 0 120
run_shard 1 120 240
sleep 10
echo "running: $(pgrep -cf canonical_selfcheck.py)"
