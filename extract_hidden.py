"""Faithful 'observer probe' baseline (O'Neill et al. / InterpDetect style): dump the proxy's mean
RESIDUAL-STREAM hidden state over the response tokens (with context in scope), at the last and a middle
layer. A linear probe on these hidden states is the real static-observer baseline we compare against
(no conflict-gating, no contrastive two-pass). Reuses the exact context/response handling of the main
extractor so the comparison is apples-to-apples on the same examples/splits.
"""
import json, os, argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def as_text(x):
    if isinstance(x, str):
        return x
    return "" if x is None else json.dumps(x, ensure_ascii=False)


def context_candidates(raw):
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if raw is None:
        return []
    out = []
    if isinstance(raw, dict):
        for key in ("passages", "passage", "context", "text", "document", "content"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v)
        for v in raw.values():
            if isinstance(v, str) and v.strip() and v not in out:
                out.append(v)
    if isinstance(raw, list):
        joined = "\n".join(x for x in raw if isinstance(x, str))
        if joined.strip():
            out.append(joined)
    for f in (lambda: str(raw), lambda: json.dumps(raw, ensure_ascii=False)):
        try:
            out.append(f())
        except Exception:
            pass
    return [c for c in out if c and c.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/hidden_ragtruth.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_resp", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, attn_implementation="eager").to(args.device).eval()
    torch.set_grad_enabled(False)
    nL = model.config.num_hidden_layers
    mid = nL // 2

    exs = load_jsonl(args.examples)
    if args.nshards > 1:
        exs = [e for i, e in enumerate(exs) if i % args.nshards == args.shard]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0, n = time.time(), 0
    with open(args.out, "w") as fout:
        for e in exs:
            prompt = as_text(e.get("prompt", "")); response = as_text(e.get("response", ""))
            if not response.strip():
                continue
            rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, tokenize=False)
            pre_ids = tok(rendered, add_special_tokens=False)["input_ids"]
            resp_ids = tok(response, add_special_tokens=False)["input_ids"][:args.max_resp]
            if not resp_ids:
                continue
            full = pre_ids + resp_ids
            if len(full) > args.max_len:
                keep = max(1, args.max_len - len(resp_ids))
                pre_ids = pre_ids[-keep:]; full = pre_ids + resp_ids
            rstart = len(pre_ids); L = len(full)
            try:
                out = model(torch.tensor([full], device=args.device),
                            output_hidden_states=True, use_cache=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            hs = out.hidden_states                       # tuple len nL+1, each [1,L,d]
            h_last = hs[-1][0][rstart:L].float().mean(0).cpu().numpy()
            h_mid = hs[mid][0][rstart:L].float().mean(0).cpu().numpy()
            fout.write(json.dumps({"id": e.get("id"), "label": int(e["label"]),
                                   "split": e.get("split", ""), "model": e.get("model", ""),
                                   "task_type": e.get("task_type", ""),
                                   "h_last": [round(float(x), 5) for x in h_last],
                                   "h_mid": [round(float(x), 5) for x in h_mid]}) + "\n")
            n += 1
            if n % 500 == 0:
                print("[%s] %d (%.0fs)" % (args.device, n, time.time() - t0), flush=True)
    print("dumped %d -> %s (%.0fs)" % (n, args.out, time.time() - t0))


if __name__ == "__main__":
    main()
