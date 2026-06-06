"""Richer feature extractor: logit-level signals (as before) + ReDeEP-style MECHANISTIC
signals (FFN write magnitude = parametric-injection proxy; attention-to-context = copy-head
proxy), each in all-token and conflict-gated forms.

WITH-context pass runs with eager attention + output_attentions + output_hidden_states and
forward hooks on each layer's MLP, so we get per-response-token:
  attn2ctx  : mean over layers/heads of attention mass on the retrieved-context tokens (ECS)
  attn2ctx_maxhead : strongest single copy-head's attention-to-context (running max over layers)
  ffn_norm  : mean over layers of ||MLP_output|| at that token            (PKS magnitude)
  ffn_ratio : mean over layers of ||MLP_output|| / ||residual||           (PKS relative)
WITHOUT-context pass is logits-only (to compute Confidence Gain + Δlogprob -> conflict mask).

Conflict tokens := dlp<0 (context disagrees with the generated token), same as the logit script.
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
    rows = logits[rstart - 1:len(full) - 1].float()          # only response-predicting rows
    logp = torch.log_softmax(rows, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    resp = torch.tensor(full[rstart:len(full)], device=device)
    lp = logp[torch.arange(len(resp), device=device), resp].float().cpu().numpy()
    H = ent.float().cpu().numpy()
    return lp, H


def build_with_mask(tok, prompt, context):
    """Render WITH-context prefix and a boolean mask marking context tokens."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    mask = [False] * len(ids)
    if context:
        s = rendered.find(context)
        if s >= 0:
            e = s + len(context)
            for i, (a, b) in enumerate(offs):
                if b > a and a >= s and b <= e:
                    mask[i] = True
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/features_mech.jsonl"))
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
            context = as_text(e.get("context", ""))
            prompt = as_text(e.get("prompt", ""))
            response = as_text(e.get("response", ""))
            if not response.strip():
                continue
            resp_ids = tok(response, add_special_tokens=False)["input_ids"][:args.max_resp]
            if len(resp_ids) == 0:
                continue
            pre_ids, ctx_mask_pre = build_with_mask(tok, prompt, context)
            if any(ctx_mask_pre):
                n_ctxfound += 1

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
            ctx_idx = torch.tensor([i for i, m in enumerate(ctx_mask) if m],
                                   device=args.device, dtype=torch.long)

            mlp_store.clear()
            try:
                out = model(torch.tensor([full], device=args.device),
                            output_attentions=True, output_hidden_states=True, use_cache=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            R = L - rstart
            rows = out.logits[0][rstart - 1:L - 1].float()    # only response-predicting rows
            logp = torch.log_softmax(rows, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            resp_t = torch.tensor(full[rstart:L], device=args.device)
            lp_w = logp[torch.arange(R, device=args.device), resp_t].float().cpu().numpy()
            H_w = ent.float().cpu().numpy()

            # attention-to-context per response token
            if ctx_idx.numel() > 0:
                a2c = torch.zeros(R, device=args.device)
                a2c_max = torch.zeros(R, device=args.device)
                for la in out.attentions:                     # [1,H,L,L]
                    sub = la[0][:, rstart:L, :]                # [H,R,L]
                    cm = sub[:, :, ctx_idx].sum(-1)            # [H,R]
                    a2c += cm.mean(0)
                    a2c_max = torch.maximum(a2c_max, cm.max(0).values)
                attn2ctx = (a2c / len(out.attentions)).float().cpu().numpy()
                attn2ctx_max = a2c_max.float().cpu().numpy()
            else:
                attn2ctx = np.zeros(R)
                attn2ctx_max = np.zeros(R)

            # FFN write magnitude per response token
            ffn_norm = np.zeros(R)
            ffn_ratio = np.zeros(R)
            for i in range(nlayers):
                mo = mlp_store[i][0][rstart:L]                 # [R,d]
                hs = out.hidden_states[i + 1][0][rstart:L]     # [R,d]
                nmo = mo.norm(dim=-1).float()
                nhs = hs.norm(dim=-1).float().clamp_min(1e-6)
                ffn_norm += nmo.cpu().numpy()
                ffn_ratio += (nmo / nhs).cpu().numpy()
            ffn_norm /= nlayers
            ffn_ratio /= nlayers
            del out
            mlp_store.clear()

            # WITHOUT-context pass (logits only)
            if context and context in prompt:
                without_instr = prompt.replace(context, "").strip()
            else:
                without_instr = (e.get("task_type", "") or "") + " Respond to the request."
            if not without_instr.strip():
                without_instr = "Respond to the request."
            pre_wo = list(tok.apply_chat_template(
                [{"role": "user", "content": without_instr}], add_generation_prompt=True, tokenize=True))
            lp_wo, H_wo = logit_stats(model, pre_wo, resp_ids, args.device, args.max_len)

            n = min(len(lp_w), len(lp_wo), len(attn2ctx), len(ffn_norm))
            if n == 0:
                continue
            lp_w, H_w = lp_w[:n], H_w[:n]
            lp_wo, H_wo = lp_wo[:n], H_wo[:n]
            attn2ctx, attn2ctx_max = attn2ctx[:n], attn2ctx_max[:n]
            ffn_norm, ffn_ratio = ffn_norm[:n], ffn_ratio[:n]
            CG = H_wo - H_w
            dlp = lp_w - lp_wo
            conflict = dlp < 0

            f = {}
            f["base_meanlp"] = float(np.mean(lp_w))
            f["base_minlp"] = float(np.min(lp_w))
            # logit all-token
            for name, arr in [("lp_w", lp_w), ("H_w", H_w), ("CG", CG), ("dlp", dlp)]:
                m, mx, mn, sd = agg(arr)
                f[f"all_{name}_mean"], f[f"all_{name}_max"] = m, mx
                f[f"all_{name}_min"], f[f"all_{name}_std"] = mn, sd
            # logit conflict-gated
            f["conf_frac"] = float(np.mean(conflict))
            for name, arr in [("lp_w", lp_w[conflict]), ("H_w", H_w[conflict]),
                              ("dlp", dlp[conflict]), ("CG", CG[conflict])]:
                m, mx, mn, sd = agg(arr)
                f[f"conf_{name}_mean"], f[f"conf_{name}_min"] = m, mn
            # mechanistic all-token
            for name, arr in [("attn2ctx", attn2ctx), ("attn2ctxmax", attn2ctx_max),
                              ("ffnnorm", ffn_norm), ("ffnratio", ffn_ratio)]:
                m, mx, mn, sd = agg(arr)
                f[f"mech_{name}_mean"], f[f"mech_{name}_max"] = m, mx
                f[f"mech_{name}_min"], f[f"mech_{name}_std"] = mn, sd
            # mechanistic conflict-gated
            for name, arr in [("attn2ctx", attn2ctx[conflict]), ("attn2ctxmax", attn2ctx_max[conflict]),
                              ("ffnnorm", ffn_norm[conflict]), ("ffnratio", ffn_ratio[conflict])]:
                m, mx, mn, sd = agg(arr)
                f[f"cmech_{name}_mean"], f[f"cmech_{name}_min"] = m, mn

            rec = {"id": e["id"], "model": e["model"], "task_type": e["task_type"],
                   "split": e["split"], "label": e["label"], "n_resp_tok": int(n), "features": f}
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
