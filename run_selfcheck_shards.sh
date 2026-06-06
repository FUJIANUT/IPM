#!/usr/bin/env bash
# Launch SelfCheckGPT in 3 GPU shards over the 2700-example RAGTruth test set, then merge.
cd ~/cgp || exit 1
pkill -f "selfcheck_baseline.py" 2>/dev/null
sleep 2
rm -f data/sc_s0_DONE data/sc_s1_DONE data/sc_s2_DONE data/scores/selfcheck_s0.jsonl data/scores/selfcheck_s1.jsonl data/scores/selfcheck_s2.jsonl

run_shard () {
  local start=$1 end=$2 dev=$3 idx=$4
  nohup setsid bash -c "source ~/cgp/venv/bin/activate; cd ~/cgp; python code/selfcheck_baseline.py --start ${start} --end ${end} --device ${dev} --max_new_tokens 128 --out data/scores/selfcheck_s${idx}.jsonl > data/run_sc_s${idx}.log 2>&1; echo done > data/sc_s${idx}_DONE" </dev/null >/dev/null 2>&1 &
}

run_shard 0    900  cuda:0 0
run_shard 900  1800 cuda:1 1
run_shard 1800 2700 cuda:2 2
sleep 8
echo "running selfcheck procs: $(pgrep -cf 'selfcheck_baseline.py')"
pgrep -af "selfcheck_baseline.py" | grep -v run_selfcheck_shards | head
