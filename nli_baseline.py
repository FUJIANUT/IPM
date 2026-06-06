"""NLI-faithfulness baseline (SummaC-ZS style).

For each response sentence, take the max entailment from any context chunk; the example
hallucination score = 1 - mean over sentences of that max entailment. Standard RAG faithfulness
baseline. Dumps data/scores/nli.jsonl.
"""
import json, os, argparse, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))


def sents(t):
    s = re.split(r"(?<=[.!?])\s+", t.strip())
    return [x.strip() for x in s if x.strip()]


def chunk_ctx(t, tok, maxtok=380, maxchunks=12):
    ss = sents(t)
    ch, cur = [], ""
    for s in ss:
        cand = (cur + " " + s).strip()
        if cur and len(tok(cand, add_special_tokens=False).input_ids) > maxtok:
            ch.append(cur); cur = s
        else:
            cur = cand
    if cur:
        ch.append(cur)
    return (ch or [t[:1500]])[:maxchunks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/scores/nli.jsonl"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device).eval()
    torch.set_grad_enabled(False)
    ent_idx = [i for i, l in model.config.id2label.items() if "entail" in l.lower()][0]

    te = [r for r in load_jsonl(args.examples) if r["split"] == "test"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = []
    for k, r in enumerate(te):
        ctx = as_text(r["context"]); resp = as_text(r["response"])
        rs = sents(resp)[:25]
        cs = chunk_ctx(ctx, tok)
        if not rs or not cs:
            out.append((r["id"], r["label"], 1.0)); continue
        pairs = [(c, s) for s in rs for c in cs]
        ent = np.zeros(len(pairs))
        for b in range(0, len(pairs), 64):
            batch = pairs[b:b + 64]
            enc = tok([p for p, _ in batch], [h for _, h in batch], return_tensors="pt",
                      truncation=True, max_length=512, padding=True).to(args.device)
            prob = model(**enc).logits.softmax(-1)[:, ent_idx].float().cpu().numpy()
            ent[b:b + len(batch)] = prob
        ent = ent.reshape(len(rs), len(cs))
        sent_max = ent.max(axis=1)              # best entailment per response sentence
        score = 1.0 - float(sent_max.mean())    # higher = less entailed = more hallucinated
        out.append((r["id"], r["label"], score))
        if (k + 1) % 500 == 0:
            print(f"{k+1}/{len(te)}", flush=True)
    with open(args.out, "w") as f:
        for i, lab, s in out:
            f.write(json.dumps({"id": i, "label": int(lab), "score": float(s)}) + "\n")
    print(f"dumped NLI: {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
