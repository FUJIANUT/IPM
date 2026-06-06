#!/usr/bin/env bash
# Isolated venv for LettuceDetect (ModernBERT needs transformers>=4.48 / newer torch),
# kept separate from the extraction venv (pinned torch2.5+transformers4.46.3).
set -e
cd ~/cgp
mkdir -p tmp; export TMPDIR="$PWD/tmp"
python3 -m venv venv_ld
source venv_ld/bin/activate
pip install -U pip
pip install --no-cache-dir --retries 10 --timeout 180 lettucedetect
python -c "import transformers,torch;print('LD env:',transformers.__version__, torch.__version__, torch.cuda.is_available())"
python code/lettucedetect_baseline.py
echo "LD_DONE" > data/ld_DONE
