"""Dump per-example TEST scores for our method(s) and feature-based baselines into data/scores/.
Each output: data/scores/<name>.jsonl with {id,label,score} (higher score = more hallucinated).
"""
import json, os, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.expanduser("~/cgp/data/features_mech2.jsonl"))
    ap.add_argument("--outdir", default=os.path.expanduser("~/cgp/data/scores"))
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    recs = load_jsonl(args.features)
    fn = sorted(recs[0]["features"].keys())
    base = ["base_meanlp", "base_minlp"]
    logit = [n for n in fn if n.startswith("all_") or n.startswith("conf_")]
    mech_ffn = [n for n in fn if "ffn" in n and (n.startswith("mech_") or n.startswith("cmech_"))]
    mech_attn = [n for n in fn if "attn" in n and (n.startswith("mech_") or n.startswith("cmech_"))]
    nheads = recs[0]["nheads"]
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    ytr = np.array([r["label"] for r in tr])

    def feats(rs, names):
        return np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in rs], dtype=float))

    Hall_tr = np.array([r["head_a2c"] for r in tr]); Hall_te = np.array([r["head_a2c"] for r in te])
    Hcf_tr = np.array([r["head_a2c_conf"] for r in tr]); Hcf_te = np.array([r["head_a2c_conf"] for r in te])
    sc_head = np.zeros(Hall_tr.shape[1])
    for j in range(Hall_tr.shape[1]):
        if np.std(Hall_tr[:, j]) > 1e-9:
            sc_head[j] = abs(roc_auc_score(ytr, Hall_tr[:, j]) - 0.5)
    sel = np.argsort(sc_head)[::-1][:args.topk]

    def ecs(Hall, Hcf):
        ms, cs = Hall[:, sel], Hcf[:, sel]
        return np.column_stack([ms.mean(1), ms.max(1), ms.min(1), cs.mean(1), cs.min(1), ms, cs])

    def dump(name, scores):
        with open(os.path.join(args.outdir, name + ".jsonl"), "w") as f:
            for r, s in zip(te, scores):
                f.write(json.dumps({"id": r["id"], "label": int(r["label"]), "score": float(s)}) + "\n")
        print(f"dumped {name}: {len(te)} test scores")

    def supervised(Xtr, Xte):
        scl = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(scl.transform(Xtr), ytr)
        return clf.predict_proba(scl.transform(Xte))[:, 1]

    # our best: logit + ffn + attn + copyhead ECS (K)
    Xtr = np.column_stack([feats(tr, base + logit + mech_ffn + mech_attn), ecs(Hall_tr, Hcf_tr)])
    Xte = np.column_stack([feats(te, base + logit + mech_ffn + mech_attn), ecs(Hall_te, Hcf_te)])
    dump("ours", supervised(Xtr, Xte))
    # our simpler (transfer-robust): standard logit+mech (51)
    dump("ours_std", supervised(feats(tr, fn), feats(te, fn)))
    # ReDeEP-style ablation (proxy framework): mechanistic ONLY (FFN-PKS + copy-head-ECS),
    # NO conflict-gated logit features -> isolates what our logit features add over ReDeEP's idea.
    Xtr_rd = np.column_stack([feats(tr, mech_ffn + mech_attn), ecs(Hall_tr, Hcf_tr)])
    Xte_rd = np.column_stack([feats(te, mech_ffn + mech_attn), ecs(Hall_te, Hcf_te)])
    dump("redeep_style", supervised(Xtr_rd, Xte_rd))
    # unsupervised feature baselines
    dump("ppl", -feats(te, ["base_meanlp"])[:, 0])          # low logprob -> hallucinated
    dump("entropy", feats(te, ["all_H_w_mean"])[:, 0])      # high entropy -> hallucinated
    dump("conflict_frac", feats(te, ["conf_frac"])[:, 0])   # more conflict tokens -> hallucinated


if __name__ == "__main__":
    main()
