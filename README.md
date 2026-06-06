# Conflict-Gated Proxy Probing for RAG Hallucination Detection

A low-cost, training-free experiment: use a **tiny proxy LM** to detect hallucinations in
**larger models'** RAG outputs, and test whether **conflict-gating** the internal signals
improves detection over plain all-token aggregation.

## Research question / hypothesis
RAG hallucinations concentrate on tokens where the model's **parametric knowledge conflicts
with the retrieved context**. If we isolate those *conflict tokens* and aggregate detection
signals over them only, a simple probe should detect hallucinations **better** than
aggregating over all response tokens (and far better than a perplexity baseline).

Grounded in: ReDeEP (parametric-vs-context decoupling), CK-PLUG (Confidence Gain),
InterpDetect (small-proxy → big-model transfer), evaluated on RAGTruth.

## Method (all inference-only — no LLM training)
For each RAGTruth example, a small proxy LM (default **Qwen2.5-0.5B-Instruct**) reads the
response twice:
- **WITH context**: `[full prompt incl. retrieved context]` + response
- **WITHOUT context**: `[instruction with the context text removed]` + response

Per response token *t*:
| signal | meaning |
|---|---|
| `lp_w(t)`  | log p(token \| context, prefix) — is the token supported? |
| `H_w(t)`   | entropy of next-token dist WITH context |
| `CG(t) = H_wo - H_w` | **Confidence Gain** (>0 ⇒ context reduces uncertainty) |
| `dlp(t) = lp_w - lp_wo` | (>0 ⇒ context makes the actual token MORE likely) |

**Conflict tokens** := `dlp(t) < 0` (context *disagrees* with the generated token).
We build three feature families and compare them with a logistic-regression probe:
- `baseline_ppl` — mean/min logprob only
- `all_token` — mean/max/min/std of {lp_w, H_w, CG, dlp} over **all** tokens
- `conflict_gated` — aggregates over **conflict tokens only** + conflict fraction

Cross-model by construction: RAGTruth responses come from GPT-4, GPT-3.5, Llama-2-7/13/70B,
Mistral-7B; the 0.5B proxy only *reads* them.

## Files
| file | purpose |
|---|---|
| `data_prep.py` | join RAGTruth response+source_info → labeled examples |
| `extract_features.py` | the proxy forward passes + conflict-gated features |
| `train_probe.py` | logistic probe; compares feature sets by AUROC/AUPRC/F1 |
| `run_smoke.sh` | 60-example end-to-end sanity check |
| `run_full.sh` | shard extraction across 3 GPUs → probe → `data/results.txt` |
| `setup_env.sh` | venv + torch(cu121) + transformers==4.46.3 + sklearn |

## How to run (on the workstation, `~/cgp`)
```bash
bash code/setup_env.sh                 # once
python3 code/data_prep.py              # build labeled examples
bash code/run_smoke.sh                 # sanity check
bash code/run_full.sh                  # full run -> data/results.txt
# scale the proxy:  MODEL=Qwen/Qwen2.5-1.5B-Instruct bash code/run_full.sh
```

## Dataset
RAGTruth (ParticleMedia/RAGTruth): 17,790 examples — train 15,090 (44.5% hallucinated),
test 2,700 (34.9%); tasks Summary / Data2txt / QA; 6 generator models.

## Environment note
torch 2.5.1+cu121 requires `transformers==4.46.3` (newer transformers import-references
`torch.float8_e8m0fnu`, absent in 2.5.1). Pinned in `setup_env.sh`.

## Results (Qwen2.5-0.5B proxy, RAGTruth test n=2700)
Full numbers and interpretation in [`RESULTS.md`](./RESULTS.md). Headline:

| feature set | nfeat | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| baseline_ppl | 2 | 0.680 | 0.462 | 0.576 |
| all_token | 16 | 0.819 | 0.718 | 0.667 |
| conflict_gated | 9 | 0.820 | 0.706 | 0.671 |
| logit(all+conf) | 25 | 0.826 | 0.726 | 0.670 |
| logit + mechanistic (v2) | 49 | 0.841 | 0.757 | 0.691 |
| **logit+FFN+copy-head ECS (BEST, CV K=64)** | 184 | **0.864** | **0.786** | **0.715** |

Takeaways: (1) a **0.5B proxy** detects big-model hallucinations at AUROC ≈ 0.86
(vs 0.68 perplexity baseline) — incl. GPT-4 and Llama-2-70B.
(2) **Scaling the proxy to 1.5B does NOT help** (0.841→0.839) — 0.5B is already at ceiling.
(3) Both **ReDeEP legs** contribute: **FFN-write (PKS)** and, once context is detected on every
task and **copy heads** are selected (conflict-gated ECS, K chosen by train-CV), **attention-to-
context (ECS)** — lifting the best to **AUROC 0.864 / F1 0.715**. Copy heads sit in mid-to-late
layers. (4) Next: cross-dataset generalization (HaluEval) — see [`RESULTS.md`](./RESULTS.md).
