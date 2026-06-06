"""Copy-head selection probe (D-step): conflict-gated ECS + K sweep.

Selects copy heads on TRAIN by |AUROC-0.5|, then builds an ECS signal from the selected heads
using BOTH all-token and conflict-gated per-head attention-to-context. Sweeps K and reports the
best combination.
"""
import json, os, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.expanduser("~/cgp/data/features_mech2.jsonl"))
    ap.add_argument("--ks", default="8,16,24,48")
    args = ap.parse_args()
    recs = load_jsonl(args.features)
    fn = sorted(recs[0]["features"].keys())
    base = ["base_meanlp", "base_minlp"]
    logit = [n for n in fn if n.startswith("all_") or n.startswith("conf_")]
    mech_ffn = [n for n in fn if "ffn" in n and (n.startswith("mech_") or n.startswith("cmech_"))]
    mech_attn = [n for n in fn if "attn" in n and (n.startswith("mech_") or n.startswith("cmech_"))]
    nheads = recs[0]["nheads"]

    print(f"context-found rate: {np.mean([r.get('ctx_found', False) for r in recs]):.3f}")
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    ytr = np.array([r["label"] for r in tr])
    yte = np.array([r["label"] for r in te])
    Hall_tr = np.array([r["head_a2c"] for r in tr]); Hall_te = np.array([r["head_a2c"] for r in te])
    Hcf_tr = np.array([r["head_a2c_conf"] for r in tr]); Hcf_te = np.array([r["head_a2c_conf"] for r in te])

    scores = np.zeros(Hall_tr.shape[1])
    for j in range(Hall_tr.shape[1]):
        if np.std(Hall_tr[:, j]) > 1e-9:
            scores[j] = abs(roc_auc_score(ytr, Hall_tr[:, j]) - 0.5)
    order = np.argsort(scores)[::-1]
    print(f"\nTop copy heads by train |AUROC-0.5|:")
    for j in order[:10]:
        print(f"  L{j // nheads:>2} H{j % nheads:<2}  score={scores[j]:.3f}")

    def feats(rs, names):
        return np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in rs], dtype=float))

    def ecs(sel, Hall, Hcf):
        ms, cs = Hall[:, sel], Hcf[:, sel]
        ag = np.column_stack([ms.mean(1), ms.max(1), ms.min(1), cs.mean(1), cs.min(1)])
        return np.column_stack([ag, ms, cs])     # aggregates + per-head all-token + per-head conflict

    def ev(name, Xtr, Xte):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        au = roc_auc_score(yte, p)
        print(f"{name:<30}{Xtr.shape[1]:>4}{au:>8.3f}{average_precision_score(yte,p):>8.3f}"
              f"{f1_score(yte,(p>0.5).astype(int),zero_division=0):>7.3f}")
        return au

    lf_tr, lf_te = feats(tr, base + logit + mech_ffn), feats(te, base + logit + mech_ffn)
    print(f"\n{'feature_set':<30}{'nf':>4}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}")
    print("-" * 57)
    ev("logit+ffn (no attn)", lf_tr, lf_te)
    ev("+ attn_v1 (all-head avg)",
       np.column_stack([lf_tr, feats(tr, mech_attn)]), np.column_stack([lf_te, feats(te, mech_attn)]))
    from sklearn.model_selection import StratifiedKFold
    Ks = [int(x) for x in args.ks.split(",")]
    # test-set trend (inspection only — NOT used to pick K)
    for K in Ks:
        sel = order[:K]
        ev(f"+ ECS cg (K={K}) [test trend]",
           np.column_stack([lf_tr, ecs(sel, Hall_tr, Hcf_tr)]),
           np.column_stack([lf_te, ecs(sel, Hall_te, Hcf_te)]))

    def build_all(sel, rs, Hall, Hcf):
        return np.column_stack([feats(rs, base + logit + mech_ffn + mech_attn), ecs(sel, Hall, Hcf)])

    def cv_au(X, y):
        skf = StratifiedKFold(3, shuffle=True, random_state=0)
        a = []
        for tri, vai in skf.split(X, y):
            sc = StandardScaler().fit(X[tri])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tri]), y[tri])
            a.append(roc_auc_score(y[vai], clf.predict_proba(sc.transform(X[vai]))[:, 1]))
        return float(np.mean(a))

    print("\nRigorous: choose K by 3-fold CV on TRAIN, then evaluate test once.")
    cvs = {}
    for K in Ks:
        cvs[K] = cv_au(build_all(order[:K], tr, Hall_tr, Hcf_tr), ytr)
        print(f"  K={K}: CV-AUROC={cvs[K]:.4f}")
    bestK = max(cvs, key=cvs.get)
    sel = order[:bestK]
    print(f"\n>>> CV-selected K={bestK} — TEST:")
    print(f"{'feature_set':<30}{'nf':>4}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}")
    ev(f"ALL (CV-selected K={bestK})", build_all(sel, tr, Hall_tr, Hcf_tr), build_all(sel, te, Hall_te, Hcf_te))


if __name__ == "__main__":
    main()
