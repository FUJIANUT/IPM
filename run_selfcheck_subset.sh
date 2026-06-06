#!/usr/bin/env bash
# SelfCheckGPT on a stratified ~1200-example subset of the RAGTruth test set, round-robin across 3 GPUs.
# Subsampling is standard for SelfCheckGPT given its O(N*k) generation cost; we report a bootstrap CI.
cd ~/cgp || exit 1
pkill -f "selfcheck_baseline.py" 2>/dev/null
sleep 2
rm -f data/sc_s0_DONE data/sc_s1_DONE data/sc_s2_DONE data/scores/selfcheck_s0.jsonl data/scores/selfcheck_s1.jsonl data/scores/selfcheck_s2.jsonl data/selfcheck_subset.jsonl

# Build deterministic stratified subset (proportional by task_type), interleaved so stride-3 balances long/short.
source ~/cgp/venv/bin/activate
python - <<'PY'
import json, random
te=[json.loads(l) for l in open("data/ragtruth_examples.jsonl") if l.strip()]
te=[r for r in te if r["split"]=="test"]
from collections import defaultdict
by=defaultdict(list)
for r in te: by[r.get("task_type","?")].append(r)
rng=random.Random(20260606)
TARGET=1200
sub=[]
for k,rows in by.items():
    rng.shuffle(rows)
    take=max(1,round(TARGET*len(rows)/len(te)))
    sub.append(rows[:take])
# global shuffle of the combined subset so round-robin stride-3 gives each GPU an even mix of long/short
out=[r for g in sub for r in g]
rng.shuffle(out)
with open("data/selfcheck_subset.jsonl","w") as f:
    for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
from collections import Counter
print(f"subset={len(out)} from {len(te)} test; subset per-task:",dict(Counter(r.get('task_type','?') for r in out)))
PY

run_shard () {
  local off=$1 dev=$2 idx=$3
  nohup setsid bash -c "source ~/cgp/venv/bin/activate; cd ~/cgp; python code/selfcheck_baseline.py --examples data/selfcheck_subset.jsonl --offset ${off} --stride 3 --device ${dev} --n 4 --max_new_tokens 100 --out data/scores/selfcheck_s${idx}.jsonl > data/run_sc_s${idx}.log 2>&1; echo done > data/sc_s${idx}_DONE" </dev/null >/dev/null 2>&1 &
}
run_shard 0 cuda:0 0
run_shard 1 cuda:1 1
run_shard 2 cuda:2 2
sleep 8
echo "running selfcheck procs: $(pgrep -cf 'selfcheck_baseline.py')"
