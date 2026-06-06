"""Canonical (generator-sampled) SelfCheckGPT on the RAGTruth Mistral-7B subset, where we CAN run the
actual generator. Unlike our proxy-sampled adaptation, here we sample N responses from Mistral-7B-Instruct
itself (the model that produced the responses) and score each response sentence by NLI inconsistency with
those samples. Compares canonical (generator-sampled) vs proxy-sampled SelfCheckGPT against human labels.
4-bit to fit a 12GB GPU.
"""
import json, os, argparse, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score


def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def as_text(x): return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))
def sents(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--generator_tag", default="mistral-7B-instruct")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/scores/canonical_selfcheck_mistral.jsonl"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tok = AutoTokenizer.from_pretrained(args.gen_model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    gen = AutoModelForCausalLM.from_pretrained(args.gen_model, quantization_config=bnb, device_map={"": args.device}).eval()
    ntok = AutoTokenizer.from_pretrained(args.nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(args.device).eval()
    ent_idx = [i for i, l in nli.config.id2label.items() if "entail" in l.lower()][0]

    ex = [r for r in load(os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
          if r.get("split") == "test" and r.get("model") == args.generator_tag]
    ex = ex[:args.limit]
    end = len(ex) if args.end < 0 else args.end
    ex = ex[args.start:end]
    out = []
    for k, r in enumerate(ex):
        prompt = as_text(r["prompt"]); resp = as_text(r["response"]); rs = sents(resp)[:20]
        if not rs:
            out.append((r["id"], r["label"], 0.5)); continue
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(args.device)
        if ids.shape[1] > 3500:
            ids = ids[:, -3500:]
        try:
            g = gen.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0,
                             top_p=0.9, max_new_tokens=args.max_new_tokens, num_return_sequences=args.n,
                             pad_token_id=tok.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); out.append((r["id"], r["label"], 0.5)); continue
        samples = [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in g]
        pairs = [(smp, sr) for sr in rs for smp in samples]
        ent = np.zeros(len(pairs))
        for b in range(0, len(pairs), 64):
            ba = pairs[b:b + 64]
            enc = ntok([p for p, _ in ba], [h for _, h in ba], return_tensors="pt", truncation=True, max_length=512, padding=True).to(args.device)
            ent[b:b + len(ba)] = nli(**enc).logits.softmax(-1)[:, ent_idx].float().cpu().numpy()
        ent = ent.reshape(len(rs), len(samples))
        out.append((r["id"], r["label"], float((1.0 - ent.max(1)).mean())))
        if (k + 1) % 50 == 0:
            print("%d/%d" % (k + 1, len(ex)), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, lab, s in out:
            f.write(json.dumps({"id": i, "label": int(lab), "score": float(s)}) + "\n")
    y = [l for _, l, _ in out]; s = [v for _, _, v in out]
    print("canonical (Mistral-sampled) SelfCheckGPT: n=%d AUROC=%.3f -> %s" % (len(out), roc_auc_score(y, s), args.out))


if __name__ == "__main__":
    main()
