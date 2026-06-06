"""SelfCheckGPT-NLI baseline (RAG-adapted). Sample N stochastic responses from the proxy given the
same (context+question) prompt, then score each response sentence by its inconsistency with the samples
via NLI (1 - max entailment over samples). Example score = mean over sentences. Dumps scores/selfcheck.jsonl.

Supports sharding (--start/--end + per-shard --out) so the O(N*k) sampling cost can be split across GPUs.
"""
import json, os, argparse, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))


def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--gen_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/scores/selfcheck.jsonl"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)   # round-robin shard: te[offset::stride]
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.gen_model)
    gen = AutoModelForCausalLM.from_pretrained(args.gen_model, torch_dtype=torch.float16).to(args.device).eval()
    ntok = AutoTokenizer.from_pretrained(args.nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(args.device).eval()
    ent_idx = [i for i, l in nli.config.id2label.items() if "entail" in l.lower()][0]

    te = [r for r in load_jsonl(args.examples) if r["split"] == "test"]
    if args.stride > 1:
        te = te[args.offset::args.stride]   # round-robin shard (balances long/short across GPUs)
    else:
        end = len(te) if args.end < 0 else min(args.end, len(te))
        te = te[args.start:end]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = []
    for k, r in enumerate(te):
        prompt = as_text(r["prompt"]); resp = as_text(r["response"])
        rs = sents(resp)[:20]
        if not rs:
            out.append((r["id"], r["label"], 0.5)); continue
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(args.device)
        if ids.shape[1] > 3500:
            ids = ids[:, -3500:]
        attn = torch.ones_like(ids)
        try:
            gced = gen.generate(ids, attention_mask=attn, do_sample=True, temperature=1.0, top_p=0.9,
                                max_new_tokens=args.max_new_tokens, num_return_sequences=args.n,
                                pad_token_id=tok.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); out.append((r["id"], r["label"], 0.5)); continue
        samples = [tok.decode(g[ids.shape[1]:], skip_special_tokens=True) for g in gced]
        # NLI: per response sentence, max entailment over samples
        pairs = [(s_sample, s_resp) for s_resp in rs for s_sample in samples]
        ent = np.zeros(len(pairs))
        for b in range(0, len(pairs), 64):
            batch = pairs[b:b + 64]
            enc = ntok([p for p, _ in batch], [h for _, h in batch], return_tensors="pt",
                       truncation=True, max_length=512, padding=True).to(args.device)
            ent[b:b + len(batch)] = nli(**enc).logits.softmax(-1)[:, ent_idx].float().cpu().numpy()
        ent = ent.reshape(len(rs), len(samples))
        score = float((1.0 - ent.max(axis=1)).mean())   # higher = more inconsistent = more hallucinated
        out.append((r["id"], r["label"], score))
        if (k + 1) % 100 == 0:
            print(f"[{args.device}] {k+1}/{len(te)}", flush=True)
    with open(args.out, "w") as f:
        for i, lab, s in out:
            f.write(json.dumps({"id": i, "label": int(lab), "score": float(s)}) + "\n")
    print(f"dumped selfcheck shard [{args.start}:{end}] -> {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
