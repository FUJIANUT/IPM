"""Phase 2 summary: best probe (logit+ffn+attn+copyhead ECS, K=64) per proxy model,
to show the approach generalizes across proxy families (not Qwen-specific).
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def best_metrics(recs, topk=64):
    fn = sorted(recs[0]["features"].keys())
    base = ["base_meanlp", "base_minlp"]
    logit = [n for n in fn if n.startswith("all_") or n.startswith("conf_")]
    ffn = [n for n in fn if "ffn" in n and (n.startswith("mech_") or n.startswith("cmech_"))]
    attn = [n for n in fn if "attn" in n and (n.startswith("mech_") or n.startswith("cmech_"))]
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    ytr = np.array([r["label"] for r in tr]); yte = np.array([r["label"] for r in te])

    def feats(rs, names):
        return np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in rs], float))

    Htr = np.array([r["head_a2c"] for r in tr]); Hte = np.array([r["head_a2c"] for r in te])
    Ctr = np.array([r["head_a2c_conf"] for r in tr]); Cte = np.array([r["head_a2c_conf"] for r in te])
    sc = np.zeros(Htr.shape[1])
    for j in range(Htr.shape[1]):
        if np.std(Htr[:, j]) > 1e-9:
            sc[j] = abs(roc_auc_score(ytr, Htr[:, j]) - 0.5)
    sel = np.argsort(sc)[::-1][:topk]

    def ecs(H, C):
        ms, cs = H[:, sel], C[:, sel]
        return np.column_stack([ms.mean(1), ms.max(1), ms.min(1), cs.mean(1), cs.min(1), ms, cs])

    Xtr = np.column_stack([feats(tr, base + logit + ffn + attn), ecs(Htr, Ctr)])
    Xte = np.column_stack([feats(te, base + logit + ffn + attn), ecs(Hte, Cte)])
    s = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(s.transform(Xtr), ytr)
    p = clf.predict_proba(s.transform(Xte))[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p), f1_score(yte, (p > 0.5).astype(int), zero_division=0)


PROXIES = [
    ("Qwen2.5-0.5B (Qwen)", "data/features_p2_qwen05.jsonl"),
    ("Qwen2.5-1.5B (Qwen)", "data/features_p2_qwen15.jsonl"),
    ("SmolLM2-1.7B (HF)", "data/features_p2_smol17.jsonl"),
    ("TinyLlama-1.1B (Llama)", "data/features_p2_tiny11.jsonl"),
]


def main():
    os.chdir(os.path.expanduser("~/cgp"))
    print(f"{'proxy (family)':<26}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}")
    print("-" * 49)
    for name, path in PROXIES:
        if not os.path.exists(path):
            print(f"{name:<26}  (missing)")
            continue
        au, pr, f1 = best_metrics(load(path))
        print(f"{name:<26}{au:>8.3f}{pr:>8.3f}{f1:>7.3f}")


if __name__ == "__main__":
    main()
