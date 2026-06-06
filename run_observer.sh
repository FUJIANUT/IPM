#!/usr/bin/env bash
# Extract observer-probe hidden states for RAGTruth + 3 transfer sets, 3-GPU sharded, merge per dataset.
cd ~/cgp || exit 1
source venv/bin/activate
rm -f data/observer_DONE
DS="ragtruth haluqa ragbench faitheval"
for d in $DS; do
  EX="data/${d}_examples.jsonl"
  [ -f "$EX" ] || { echo "MISSING $EX, skip"; continue; }
  echo "=== $d ($(wc -l < $EX) ex) ==="
  rm -f data/obs_${d}_s0_DONE data/obs_${d}_s1_DONE data/obs_${d}_s2_DONE
  for s in 0 1 2; do
    nohup setsid bash -c "source ~/cgp/venv/bin/activate; cd ~/cgp; python code/extract_hidden.py --examples $EX --shard $s --nshards 3 --device cuda:$s --out data/hidden_${d}_s${s}.jsonl > data/run_obs_${d}_s${s}.log 2>&1; echo done > data/obs_${d}_s${s}_DONE" </dev/null >/dev/null 2>&1 &
  done
  # wait for this dataset's 3 shards
  while [ ! -f data/obs_${d}_s0_DONE ] || [ ! -f data/obs_${d}_s1_DONE ] || [ ! -f data/obs_${d}_s2_DONE ]; do sleep 10; done
  cat data/hidden_${d}_s0.jsonl data/hidden_${d}_s1.jsonl data/hidden_${d}_s2.jsonl > data/hidden_${d}.jsonl
  echo "$d merged: $(wc -l < data/hidden_${d}.jsonl) rows"
done
echo done > data/observer_DONE
echo "ALL OBSERVER EXTRACTION DONE"
