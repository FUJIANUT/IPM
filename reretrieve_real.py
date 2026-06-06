"""Real generation validation of the re-retrieval premise. For a subset of robustness base queries we
actually GENERATE answers with a real generator (Qwen2.5-3B-Instruct) under (a) a DEGRADED retrieval context
and (b) the GOLD context (= 're-retrieve + regenerate'), then measure each answer's faithfulness with an
INDEPENDENT SummaC-style NLI oracle (answer sentences entailed by the gold evidence). This turns the
counterfactual's premise---that re-retrieval+regeneration recovers faithfulness---from an assumption into a
measured fact. Cached Qwen2.5-3B (fp16, fits 12GB) + DeBERTa-NLI.
"""
import json, os, argparse, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def as_text(x): return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))
def sents(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def chunks(text, tok, maxlen=380):
    ss = sents(text); out = []; cur = ""
    for s in ss:
        if len(tok(cur + " " + s)["input_ids"]) > maxlen and cur:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur: out.append(cur)
    return out[:12] or [text[:1500]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/reretrieve_real.json"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.gen_model)
    gen = AutoModelForCausalLM.from_pretrained(args.gen_model, torch_dtype=torch.float16).to(args.device).eval()
    ntok = AutoTokenizer.from_pretrained(args.nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(args.device).eval()
    ent_idx = [i for i, l in nli.config.id2label.items() if "entail" in l.lower()][0]

    rob = load(os.path.expanduser("~/cgp/data/robust_examples.jsonl"))
    by = {}
    for r in rob:
        base, cond = r["id"].rsplit("_", 1)
        by.setdefault(base, {})[cond] = r
    bases = [b for b, d in by.items() if "gold" in d and any(c in d for c in ("dist", "random", "partial", "inject"))]
    bases = bases[:args.limit]

    def generate(prompt):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(args.device)
        if ids.shape[1] > 3500: ids = ids[:, -3500:]
        g = gen.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=160, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)

    def faithfulness(answer, gold_ctx):
        rs = sents(answer)[:15]
        if not rs: return 1.0
        cks = chunks(gold_ctx, ntok)
        pairs = [(c, s) for s in rs for c in cks]
        ent = np.zeros(len(pairs))
        for b in range(0, len(pairs), 64):
            ba = pairs[b:b + 64]
            enc = ntok([p for p, _ in ba], [h for _, h in ba], return_tensors="pt", truncation=True, max_length=512, padding=True).to(args.device)
            ent[b:b + len(ba)] = nli(**enc).logits.softmax(-1)[:, ent_idx].float().cpu().numpy()
        ent = ent.reshape(len(rs), len(cks))
        return float(ent.max(1).mean())     # mean over answer sentences of max entailment vs gold evidence

    f_deg, f_gold = [], []
    for k, b in enumerate(bases):
        d = by[b]
        gold = d["gold"]; deg = next(d[c] for c in ("dist", "random", "partial", "inject") if c in d)
        gold_ctx = as_text(gold.get("context"))
        try:
            a_deg = generate(as_text(deg["prompt"])); a_gold = generate(as_text(gold["prompt"]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        f_deg.append(faithfulness(a_deg, gold_ctx)); f_gold.append(faithfulness(a_gold, gold_ctx))
        if (k + 1) % 50 == 0:
            print("%d/%d  deg=%.3f gold=%.3f" % (k + 1, len(bases), np.mean(f_deg), np.mean(f_gold)), flush=True)
    fd, fg = np.array(f_deg), np.array(f_gold)
    # bootstrap CI on the gain
    rng = np.random.RandomState(0); gains = [(fg[i] - fd[i]).mean() for i in (rng.randint(0, len(fd), len(fd)) for _ in range(2000))]
    res = {"n": len(fd), "faith_degraded": float(fd.mean()), "faith_gold_regenerated": float(fg.mean()),
           "gain": float(fg.mean() - fd.mean()), "gain_ci": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))]}
    json.dump(res, open(args.out, "w"))
    print("REAL re-retrieve+regenerate (n=%d): degraded faithfulness=%.3f -> gold-regenerated=%.3f (gain %.3f %s)"
          % (res["n"], res["faith_degraded"], res["faith_gold_regenerated"], res["gain"], res["gain_ci"]))


if __name__ == "__main__":
    main()
