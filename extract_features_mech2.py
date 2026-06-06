"""B-step extractor: fixes Data2txt context detection (dict is embedded as Python str(dict),
not json) AND emits PER-HEAD attention-to-context so we can select real copy heads downstream.

Superset of extract_features_mech.py outputs, plus:
  head_a2c       : per-(layer,head) mean attention-to-context over response tokens   (len = nlayers*nheads)
  head_a2c_conf  : same but averaged over conflict tokens only
  nlayers, nheads: for reshaping
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
    if x is None:
        return ""
    return json.dumps(x, ensure_ascii=False)


def context_candidates(raw):
    """Strings to try matching against the prompt to locate the retrieved context.
    Handles: plain str; QA dict {question, passages}; Data2txt dict (embedded as str(dict))."""
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
        for v in raw.values():                     # any other string fields
            if isinstance(v, str) and v.strip() and v not in out:
                out.append(v)
    if isinstance(raw, list):
        joined = "\n".join(x for x in raw if isinstance(x, str))
        if joined.strip():
            out.append(joined)
    try:
        out.append(str(raw))                       # Python repr -> matches Data2txt prompts
    except Exception:
        pass
    try:
        out.append(json.dumps(raw, ensure_ascii=False))
    except Exception:
        pass
    return [c for c in out if c and c.strip()]


def agg(x):
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(x)), float(np.max(x)), float(np.min(x)), float(np.std(x))


@torch.no_grad()
def logit_stats(model, prefix_ids, resp_ids, device, max_len):
    full = prefix_ids + resp_ids
    if len(full) > max_len:
        keep = max(1, max_len - len(resp_ids))
        prefix_ids = prefix_ids[-keep:]
        full = prefix_ids + resp_ids
    rstart = len(prefix_ids)
    logits = model(torch.tensor([full], device=device)).logits[0]
    rows = logits[rstart - 1:len(full) - 1].float()
    logp = torch.log_softmax(rows, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    resp = torch.tensor(full[rstart:len(full)], device=device)
    lp = logp[torch.arange(len(resp), device=device), resp].float().cpu().numpy()
    H = ent.float().cpu().numpy()
    return lp, H


def build_with_mask(tok, prompt, ctx_str):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    mask = [False] * len(ids)
    if ctx_str:
        s = rendered.find(ctx_str)
        if s >= 0:
            e = s + len(ctx_str)
            for i, (a, b) in enumerate(offs):
                if b > a and a >= s and b <= e:
                    mask[i] = True
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/features_mech2.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
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
    nlayers = model.config.num_hidden_layers
    nheads = model.config.num_attention_heads

    mlp_store = {}
    def mk(i):
        def hook(m, inp, out):
            mlp_store[i] = out.detach()
        return hook
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(mk(i))

    exs = load_jsonl(args.examples)
    if args.nshards > 1:
        exs = [e for i, e in enumerate(exs) if i % args.nshards == args.shard]
    if args.limit:
        exs = exs[:args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0, n_written, n_trunc, n_ctxfound = time.time(), 0, 0, 0
    with open(args.out, "w") as fout:
        for idx, e in enumerate(exs):
            prompt = as_text(e.get("prompt", ""))
            response = as_text(e.get("response", ""))
            if not response.strip():
                continue
            cands = context_candidates(e.get("context"))
            matches = [c for c in cands if c and c in prompt]
            ctx_str = max(matches, key=len) if matches else ""    # longest match = the real context
            if ctx_str:
                n_ctxfound += 1

            resp_ids = tok(response, add_special_tokens=False)["input_ids"][:args.max_resp]
            if len(resp_ids) == 0:
                continue
            pre_ids, ctx_mask_pre = build_with_mask(tok, prompt, ctx_str)
            full = pre_ids + resp_ids
            ctx_mask = ctx_mask_pre + [False] * len(resp_ids)
            if len(full) > args.max_len:
                n_trunc += 1
                keep = max(1, args.max_len - len(resp_ids))
                pre_ids = pre_ids[-keep:]
                ctx_mask = ctx_mask_pre[-keep:] + [False] * len(resp_ids)
                full = pre_ids + resp_ids
            rstart = len(pre_ids)
            L = len(full)
            R = L - rstart
            ctx_idx = torch.tensor([i for i, m in enumerate(ctx_mask) if m],
                                   device=args.device, dtype=torch.long)

            mlp_store.clear()
            try:
                out = model(torch.tensor([full], device=args.device),
                            output_attentions=True, output_hidden_states=True, use_cache=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            rows = out.logits[0][rstart - 1:L - 1].float()
            logp = torch.log_softmax(rows, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            resp_t = torch.tensor(full[rstart:L], device=args.device)
            lp_w = logp[torch.arange(R, device=args.device), resp_t].float().cpu().numpy()
            H_w = ent.float().cpu().numpy()

            # per-(layer,head) attention-to-context per response token -> [nlayers, nheads, R]
            A2C = np.zeros((nlayers, nheads, R), dtype=np.float32)
            if ctx_idx.numel() > 0:
                for li, la in enumerate(out.attentions):
                    cm = la[0][:, rstart:L, :][:, :, ctx_idx].sum(-1)   # [H,R]
                    A2C[li] = cm.float().cpu().numpy()

            # FFN write magnitude per response token
            ffn_norm = np.zeros(R, dtype=np.float32)
            ffn_ratio = np.zeros(R, dtype=np.float32)
            for i in range(nlayers):
                mo = mlp_store[i][0][rstart:L]
                hs = out.hidden_states[i + 1][0][rstart:L]
                nmo = mo.norm(dim=-1).float()
                nhs = hs.norm(dim=-1).float().clamp_min(1e-6)
                ffn_norm += nmo.cpu().numpy()
                ffn_ratio += (nmo / nhs).cpu().numpy()
            ffn_norm /= nlayers
            ffn_ratio /= nlayers
            del out
            mlp_store.clear()

            # WITHOUT-context pass
            if ctx_str and ctx_str in prompt:
                without_instr = prompt.replace(ctx_str, "").strip()
            else:
                without_instr = (e.get("task_type", "") or "") + " Respond to the request."
            if not without_instr.strip():
                without_instr = "Respond to the request."
            pre_wo = list(tok.apply_chat_template(
                [{"role": "user", "content": without_instr}], add_generation_prompt=True, tokenize=True))
            lp_wo, H_wo = logit_stats(model, pre_wo, resp_ids, args.device, args.max_len)

            n = min(len(lp_w), len(lp_wo), R)
            if n == 0:
                continue
            lp_w, H_w = lp_w[:n], H_w[:n]
            lp_wo, H_wo = lp_wo[:n], H_wo[:n]
            A2C = A2C[:, :, :n]
            ffn_norm, ffn_ratio = ffn_norm[:n], ffn_ratio[:n]
            CG = H_wo - H_w
            dlp = lp_w - lp_wo
            conflict = dlp < 0

            # aggregate (all-head) mech features for backward-compat
            attn2ctx = A2C.mean(axis=(0, 1))
            attn2ctx_max = A2C.max(axis=1).max(axis=0)

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
            for name, arr in [("attn2ctx", attn2ctx), ("attn2ctxmax", attn2ctx_max),
                              ("ffnnorm", ffn_norm), ("ffnratio", ffn_ratio)]:
                m, mx, mn, sd = agg(arr)
                f[f"mech_{name}_mean"], f[f"mech_{name}_max"] = m, mx
                f[f"mech_{name}_min"], f[f"mech_{name}_std"] = mn, sd
            for name, arr in [("attn2ctx", attn2ctx[conflict]), ("attn2ctxmax", attn2ctx_max[conflict]),
                              ("ffnnorm", ffn_norm[conflict]), ("ffnratio", ffn_ratio[conflict])]:
                m, mx, mn, sd = agg(arr)
                f[f"cmech_{name}_mean"], f[f"cmech_{name}_min"] = m, mn

            head_a2c = A2C.mean(axis=2)                                   # [nlayers, nheads]
            if conflict.any():
                head_a2c_conf = A2C[:, :, conflict].mean(axis=2)
            else:
                head_a2c_conf = np.zeros((nlayers, nheads), dtype=np.float32)

            rec = {"id": e["id"], "model": e["model"], "task_type": e["task_type"],
                   "split": e["split"], "label": e["label"], "n_resp_tok": int(n),
                   "nlayers": nlayers, "nheads": nheads,
                   "ctx_found": bool(ctx_str),
                   "features": f,
                   "head_a2c": [round(v, 5) for v in head_a2c.flatten().tolist()],
                   "head_a2c_conf": [round(v, 5) for v in head_a2c_conf.flatten().tolist()]}
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
            if (idx + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"{idx+1}/{len(exs)} ({dt:.0f}s, {dt/max(n_written,1):.2f}s/ex, "
                      f"trunc={n_trunc}, ctxfound={n_ctxfound})", flush=True)
    print(f"saved -> {args.out}  ({n_written} recs, {time.time()-t0:.0f}s, "
          f"trunc={n_trunc}, ctxfound={n_ctxfound})")


if __name__ == "__main__":
    main()
