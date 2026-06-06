"""Phase 4 calibration: the RAGTruth-trained probe is miscalibrated on a transfer set (high AUROC,
poor F1@0.5). Platt-scale on a small target calibration split -> report ECE + F1 before/after, plus a
risk-coverage (selective-prediction) curve.
"""
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def main():
    os.chdir(os.path.expanduser("~/cgp"))
    RT = load("data/features_mech2.jsonl")
    HQ = load("data/features_ds_haluqa.jsonl")
    names = sorted(set(RT[0]["features"]) & set(HQ[0]["features"]))

    def XY(recs):
        X = np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in recs], float))
        y = np.array([r["label"] for r in recs])
        return X, y

    Xtr, ytr = XY([r for r in RT if r["split"] == "train"])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)

    # in-domain calibration reference (RAGTruth test)
    Xte, yte = XY([r for r in RT if r["split"] == "test"])
    pin = clf.predict_proba(sc.transform(Xte))[:, 1]
    print("=== In-domain (RAGTruth test) ===")
    print(f"AUROC={roc_auc_score(yte,pin):.3f}  ECE={ece(yte,pin):.3f}  F1@0.5={f1_score(yte,(pin>0.5).astype(int)):.3f}")

    # transfer to HaluEval-QA + calibration
    Xh, yh = XY(HQ)
    ph = clf.predict_proba(sc.transform(Xh))[:, 1]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(yh))
    k = int(0.3 * len(yh))
    ci, ei = idx[:k], idx[k:]
    print("\n=== Transfer RAGTruth -> HaluEval-QA ===")
    print(f"uncalibrated:      AUROC={roc_auc_score(yh[ei],ph[ei]):.3f}  ECE={ece(yh[ei],ph[ei]):.3f}  "
          f"F1@0.5={f1_score(yh[ei],(ph[ei]>0.5).astype(int)):.3f}")
    platt = LogisticRegression().fit(ph[ci].reshape(-1, 1), yh[ci])
    pc = platt.predict_proba(ph[ei].reshape(-1, 1))[:, 1]
    print(f"Platt-calibrated:  AUROC={roc_auc_score(yh[ei],pc):.3f}  ECE={ece(yh[ei],pc):.3f}  "
          f"F1@0.5={f1_score(yh[ei],(pc>0.5).astype(int)):.3f}")

    # risk-coverage (selective prediction): abstain on least-confident
    conf = np.abs(pc - 0.5)
    order = np.argsort(-conf)
    yeval = yh[ei]
    print("\nrisk-coverage (calibrated, abstain on low-confidence):")
    for cov in [1.0, 0.9, 0.75, 0.5]:
        m = order[:max(1, int(cov * len(order)))]
        acc = ((pc[m] > 0.5).astype(int) == yeval[m]).mean()
        print(f"  coverage={cov:>4.0%}  accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
