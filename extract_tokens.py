"""Span/token-level extractor: per response-token features + per-token hallucination labels
(from RAGTruth char-offset spans). Enables token-level detection comparable to LettuceDetect.

Per token features (fixed order in FEAT_ORDER): lp_w, H_w, CG, dlp, conflict, ffn_norm, ffn_ratio,
attn2ctx (mean over heads/layers), attn2ctx_max (strongest head). Output one record per example with
a [T x F] feature array + length-T 0/1 token labels.
"""
import json, os, argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FEAT_ORDER = ["lp_w", "H_w", "CG", "dlp", "conflict", "ffn_norm", "ffn_ratio", "attn2ctx", "attn2ctx_max"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    return json.dumps(x, ensure_ascii=False)


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
    try:
        out.append(str(raw))
    except Exception:
        pass
    return [c for c in out if c and c.strip()]


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


@torch.no_grad()
def logit_lp_H(model, prefix_ids, resp_ids, device, max_len):
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
    return lp, ent.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragtruth_dir", default=os.path.expanduser("~/cgp/RAGTruth/dataset"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/tokens.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
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

    mlp_store = {}
    def mk(i):
        def hook(m, inp, out):
            mlp_store[i] = out.detach()
        return hook
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(mk(i))

    resp = load_jsonl(os.path.join(args.ragtruth_dir, "response.jsonl"))
    src = {s["source_id"]: s for s in load_jsonl(os.path.join(args.ragtruth_dir, "source_info.jsonl"))}
    if args.nshards > 1:
        resp = [r for i, r in enumerate(resp) if i % args.nshards == args.shard]
    if args.limit:
        resp = resp[:args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0, nw = time.time(), 0
    with open(args.out, "w") as fout:
        for idx, r in enumerate(resp):
            s = src.get(r["source_id"])
            if s is None:
                continue
            response = as_text(r.get("response", ""))
            prompt = as_text(s.get("prompt", ""))
            if not response.strip():
                continue
            labels = r.get("labels", [])
            if isinstance(labels, str):
                labels = json.loads(labels)
            spans = [(int(x["start"]), int(x["end"])) for x in labels if "start" in x]

            enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
            resp_ids = enc["input_ids"][:args.max_resp]
            offs = enc["offset_mapping"][:args.max_resp]
            if len(resp_ids) == 0:
                continue
            tok_lab = []
            for (a, b) in offs:
                hit = any(a < e and b > st for (st, e) in spans)
                tok_lab.append(1 if hit else 0)

            cands = context_candidates(s.get("source_info"))
            matches = [c for c in cands if c and c in prompt]
            ctx_str = max(matches, key=len) if matches else ""
            pre_ids, ctx_mask_pre = build_with_mask(tok, prompt, ctx_str)
            full = pre_ids + resp_ids
            ctx_mask = ctx_mask_pre + [False] * len(resp_ids)
            if len(full) > args.max_len:
                keep = max(1, args.max_len - len(resp_ids))
                pre_ids = pre_ids[-keep:]
                ctx_mask = ctx_mask_pre[-keep:] + [False] * len(resp_ids)
                full = pre_ids + resp_ids
            rstart = len(pre_ids)
            L = len(full)
            R = L - rstart
            ctx_idx = torch.tensor([i for i, m in enumerate(ctx_mask) if m], device=args.device, dtype=torch.long)

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

            if ctx_idx.numel() > 0:
                a2c = torch.zeros(R, device=args.device)
                a2cmax = torch.zeros(R, device=args.device)
                for la in out.attentions:
                    cm = la[0][:, rstart:L, :][:, :, ctx_idx].sum(-1)   # [H,R]
                    a2c += cm.mean(0)
                    a2cmax = torch.maximum(a2cmax, cm.max(0).values)
                attn2ctx = (a2c / len(out.attentions)).float().cpu().numpy()
                attn2ctx_max = a2cmax.float().cpu().numpy()
            else:
                attn2ctx = np.zeros(R); attn2ctx_max = np.zeros(R)

            ffn_norm = np.zeros(R); ffn_ratio = np.zeros(R)
            for i in range(nlayers):
                mo = mlp_store[i][0][rstart:L]
                hs = out.hidden_states[i + 1][0][rstart:L]
                nmo = mo.norm(dim=-1).float()
                ffn_norm += nmo.cpu().numpy()
                ffn_ratio += (nmo / hs.norm(dim=-1).float().clamp_min(1e-6)).cpu().numpy()
            ffn_norm /= nlayers; ffn_ratio /= nlayers
            del out; mlp_store.clear()

            if ctx_str and ctx_str in prompt:
                wo = prompt.replace(ctx_str, "").strip()
            else:
                wo = "Respond to the request."
            pre_wo = list(tok.apply_chat_template([{"role": "user", "content": wo or "Respond."}],
                          add_generation_prompt=True, tokenize=True))
            lp_wo, H_wo = logit_lp_H(model, pre_wo, resp_ids, args.device, args.max_len)

            n = min(R, len(lp_wo), len(tok_lab))
            if n == 0:
                continue
            CG = H_wo[:n] - H_w[:n]
            dlp = lp_w[:n] - lp_wo[:n]
            conflict = (dlp < 0).astype(float)
            feats = np.column_stack([lp_w[:n], H_w[:n], CG, dlp, conflict,
                                     ffn_norm[:n], ffn_ratio[:n], attn2ctx[:n], attn2ctx_max[:n]])
            rec = {"id": r["id"], "split": r.get("split", "train"), "model": r.get("model", ""),
                   "task_type": s.get("task_type", ""), "ex_label": 1 if spans else 0,
                   "feats": [[round(v, 5) for v in row] for row in feats.tolist()],
                   "tok_labels": tok_lab[:n]}
            fout.write(json.dumps(rec) + "\n")
            nw += 1
            if (idx + 1) % 200 == 0:
                print(f"{idx+1}/{len(resp)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"saved -> {args.out}  ({nw} recs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
