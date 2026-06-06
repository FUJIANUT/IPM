#!/usr/bin/env bash
# Run LLM-as-judge baselines (OpenRouter) on RAGTruth test. No GPU needed.
set -e
cd ~/cgp
source venv/bin/activate
python code/llm_judge_baseline.py --model openai/gpt-4o-mini --tag llmjudge_gpt4omini
python code/llm_judge_baseline.py --model openai/gpt-4o --tag llmjudge_gpt4o
echo "DONE" > data/judges_DONE
