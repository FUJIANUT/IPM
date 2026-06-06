"""Extract proxy-model signals for RAG hallucination detection, with conflict gating.

For each RAGTruth example we run a SMALL proxy LM twice over the response tokens:
  WITH context    : proxy reads [full prompt incl. context] then the response
  WITHOUT context : proxy reads [instruction with context removed] then the response

Per response token t we record:
  lp_w(t)  : log p(token_t | context, prefix)         -- "is the token supported?"
  H_w(t)   : entropy of next-token dist WITH context
  CG(t)    : H_wo(t) - H_w(t)  (Confidence Gain; >0 => context reduces uncertainty)
  dlp(t)   : lp_w(t) - lp_wo(t) (>0 => context makes the actual token MORE likely)

CONFLICT tokens := dlp(t) < 0  (context disagrees with the generated token).
Conflict-gated features aggregate signals over conflict tokens only -- the hypothesis
is that gating to these tokens sharpens hallucination detection vs. all-token aggregation.

Output: one JSON record per example with a flat feature dict + label/split/model/task.
"""
import json, os, argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_prefix(tok, instruction):
    msgs = [{"role": "user", "content": instruction}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return list(ids)


@torch.no_grad()
def token_stats(model, tok, prefix_ids, resp_ids, device, max_len):
    """Return per-response-token (logprob_of_actual, entropy). resp_ids assumed pre-capped."""
    full = prefix_ids + resp_ids
    if len(full) > max_len:                      # left-truncate prefix, keep response intact
        keep = max(1, max_len - len(resp_ids))
        prefix_ids = prefix_ids[-keep:]
        full = prefix_ids + resp_ids
    rstart = len(prefix_ids)
    ids = torch.tensor([full], device=device)
    logits = model(ids).logits[0].float()        # [L, V]
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)           # [L]
    out = []
    for t in range(rstart, len(full)):           # token at pos t is predicted by logits[t-1]
        out.append((logp[t - 1, full[t]].item(), ent[t - 1].item()))
    return out


def agg(x):
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(x)), float(np.max(x)), float(np.min(x)), float(np.std(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/features.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_resp", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to(args.device).eval()

    exs = load_jsonl(args.examples)
    if args.nshards > 1:
        exs = [e for i, e in enumerate(exs) if i % args.nshards == args.shard]
    if args.limit:
        exs = exs[:args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0 = time.time()
    n_written = 0
    with open(args.out, "w") as fout:
        for i, e in enumerate(exs):
            def as_text(x):
                if isinstance(x, str):
                    return x
                if x is None:
                    return ""
                return json.dumps(x, ensure_ascii=False)  # Data2txt source_info is a dict
            context = as_text(e.get("context", ""))
            prompt = as_text(e.get("prompt", ""))
            response = as_text(e.get("response", ""))
            if not response.strip():
                continue
            with_instr = prompt
            if context and context in prompt:
                without_instr = prompt.replace(context, "").strip()
            else:
                without_instr = (e.get("task_type", "") or "") + " Respond to the request."
            if not without_instr.strip():
                without_instr = "Respond to the request."

            resp_ids = tok(response, add_special_tokens=False).input_ids[:args.max_resp]
            if len(resp_ids) == 0:
                continue
            sw = token_stats(model, tok, build_prefix(tok, with_instr), resp_ids, args.device, args.max_len)
            so = token_stats(model, tok, build_prefix(tok, without_instr), resp_ids, args.device, args.max_len)
            n = min(len(sw), len(so))
            if n == 0:
                continue
            lp_w = np.array([sw[j][0] for j in range(n)])
            H_w = np.array([sw[j][1] for j in range(n)])
            lp_wo = np.array([so[j][0] for j in range(n)])
            H_wo = np.array([so[j][1] for j in range(n)])
            CG = H_wo - H_w
            dlp = lp_w - lp_wo
            conflict = dlp < 0

            f = {}
            f["base_meanlp"] = float(np.mean(lp_w))
            f["base_minlp"] = float(np.min(lp_w))
            for name, arr in [("lp_w", lp_w), ("H_w", H_w), ("CG", CG), ("dlp", dlp)]:
                m, mx, mn, sd = agg(arr)
                f[f"all_{name}_mean"], f[f"all_{name}_max"] = m, mx
                f[f"all_{name}_min"], f[f"all_{name}_std"] = mn, sd
            f["conf_frac"] = float(np.mean(conflict))
            for name, arr in [("lp_w", lp_w[conflict]), ("H_w", H_w[conflict]),
                              ("dlp", dlp[conflict]), ("CG", CG[conflict])]:
                m, mx, mn, sd = agg(arr)
                f[f"conf_{name}_mean"], f[f"conf_{name}_min"] = m, mn

            rec = {"id": e["id"], "model": e["model"], "task_type": e["task_type"],
                   "split": e["split"], "label": e["label"], "n_resp_tok": int(n), "features": f}
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
            if (i + 1) % 200 == 0:
                print(f"{i+1}/{len(exs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"saved -> {args.out}  ({n_written} records, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
