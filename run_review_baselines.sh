#!/usr/bin/env bash
# Reviewer-demanded baselines: SelfCheckGPT (GPU), RAGAS faithfulness (API), and GPT-4o-judge on the
# transfer set HaluEval-QA (API) for a fair zero-shot-vs-zero-shot comparison.
cd ~/cgp
rm -f data/selfcheck_DONE data/ragas_DONE data/judgetrans_DONE
nohup setsid bash -c 'source venv/bin/activate; python code/selfcheck_baseline.py --device cuda:0 >data/run_selfcheck.log 2>&1; echo done>data/selfcheck_DONE' </dev/null >/dev/null 2>&1 &
nohup setsid bash -c 'source venv/bin/activate; python code/ragas_baseline.py --model openai/gpt-4o-mini >data/run_ragas.log 2>&1; echo done>data/ragas_DONE' </dev/null >/dev/null 2>&1 &
nohup setsid bash -c 'source venv/bin/activate; mkdir -p data/scores_transfer; python code/llm_judge_baseline.py --examples data/haluqa_examples.jsonl --model openai/gpt-4o --tag judge_gpt4o --out_dir data/scores_transfer --limit 1500 >data/run_judgetrans.log 2>&1; echo done>data/judgetrans_DONE' </dev/null >/dev/null 2>&1 &
sleep 4
echo "launched: selfcheck=$(pgrep -cf selfcheck_baseline) ragas=$(pgrep -cf ragas_baseline) judgetrans=$(pgrep -cf 'llm_judge_baseline.*haluqa')"
