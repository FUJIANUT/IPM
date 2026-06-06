#!/usr/bin/env bash
# Set up the .88 WSL Ubuntu environment (A5000 x2) for GPU experiments.
set -e
cd ~/cgp
mkdir -p tmp
export TMPDIR="$PWD/tmp"          # WSL /tmp races on pip unpack; use ext4 home instead
[ -d RAGTruth/dataset ] || git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel
# torch wheel ~700MB; WSL2 NAT can drop mid-download -> retry
for i in 1 2 3 4 5; do
  pip install --no-cache-dir --retries 10 --timeout 180 torch --index-url https://download.pytorch.org/whl/cu121 && break
  echo "torch attempt $i failed; retrying..."; sleep 5
done
pip install --no-cache-dir --retries 10 --timeout 180 "transformers==4.46.3" datasets scikit-learn numpy pandas accelerate sentencepiece protobuf
python -c "import torch;print('TORCH', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
python code/data_prep.py
echo "WSL_SETUP_DONE"
