"""Phase 6 (v2) eval: detector hallucination score across a gradient of context degradation
(gold < injected-misinfo < partial < random-distractor). Reports mean score per condition (should
rise monotonically) and gold-vs-condition AUROC.
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def cond_of(i):
    for c in ["gold", "inject", "partial", "dist"]:
        if str(i).endswith("_" + c):
            return c
    return "?"


def main():
    os.chdir(os.path.expanduser("~/cgp"))
    RT = load("data/features_mech2.jsonl")
    RB = load("data/features_robust.jsonl")
    names = sorted(set(RT[0]["features"]) & set(RB[0]["features"]))

    def XY(recs):
        X = np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in recs], float))
        return X, np.array([r["label"] for r in recs])

    Xtr, ytr = XY([r for r in RT if r["split"] == "train"])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    Xr, _ = XY(RB)
    pr = clf.predict_proba(sc.transform(Xr))[:, 1]
    cond = np.array([cond_of(r["id"]) for r in RB])

    print("=== Phase 6 (v2): retrieval-degradation gradient ===")
    order = ["gold", "inject", "partial", "dist"]
    desc = {"gold": "gold context (faithful)", "inject": "1 sentence replaced (misinfo injected)",
            "partial": "context truncated to 40%", "dist": "random distractor context"}
    print(f"{'condition':<40}{'mean score':>12}{'flag rate':>12}")
    gold = pr[cond == "gold"]
    thr = np.median(pr[cond == "gold"]) if (cond == "gold").any() else 0.5
    for c in order:
        s = pr[cond == c]
        if len(s) == 0:
            continue
        print(f"{desc[c]:<40}{s.mean():>12.3f}{np.mean(s > thr):>12.2f}")
    print("\ngold-vs-condition AUROC (higher = detector separates degraded context):")
    for c in order[1:]:
        s = pr[cond == c]
        if len(s) == 0:
            continue
        y = np.concatenate([np.zeros(len(gold)), np.ones(len(s))])
        p = np.concatenate([gold, s])
        print(f"  gold vs {c:<8} AUROC = {roc_auc_score(y, p):.3f}")
    print("\n(monotone rising score + high AUROC => detector tracks grounding across degradation levels)")


if __name__ == "__main__":
    main()
