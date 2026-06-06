"""Train the real 'observer probe' (linear probe on the proxy's mean residual-stream hidden state) on
RAGTruth and evaluate in-domain + zero-shot transfer, to compare against our conflict-gated detector and
the static feature-ablation. This is the faithful reproduction of the observer/InterpDetect-style baseline
the reviewer asked for (same examples, same transfer setting).
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

os.chdir(os.path.expanduser("~/cgp"))


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot(y, s, seed=0, B=2000):
    y = np.asarray(y); s = np.asarray(s); rng = np.random.RandomState(seed); o = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) > 1:
            o.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def mat(rows, key):
    return np.array([r[key] for r in rows], float), np.array([r["label"] for r in rows])


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(Xtr), ytr)
    return sc, clf


rt = load("data/hidden_ragtruth.jsonl")
tr = [r for r in rt if r.get("split") == "train"]
te = [r for r in rt if r.get("split") == "test"]
TRANSFER = {"HaluEval-QA": "data/hidden_haluqa.jsonl", "FaithEval": "data/hidden_faitheval.jsonl",
            "RAGBench": "data/hidden_ragbench.jsonl"}

print("OBSERVER PROBE (linear probe on proxy mean residual-stream hidden state)")
for key in ["h_last", "h_mid"]:
    print("\n=== layer feature: %s ===" % key)
    Xtr, ytr = mat(tr, key); Xte, yte = mat(te, key)
    sc, clf = fit(Xtr, ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    lo, hi = boot(yte, p)
    print("  RAGTruth in-domain AUROC = %.3f [%.3f,%.3f]   (ours full 0.864; static-ablation 0.812)" %
          (roc_auc_score(yte, p), lo, hi))
    for name, path in TRANSFER.items():
        if not os.path.exists(path):
            print("  %s: (pending)" % name); continue
        d = load(path); Xd, yd = mat(d, key)
        pd = clf.predict_proba(sc.transform(Xd))[:, 1]
        lo, hi = boot(yd, pd)
        print("  transfer -> %-12s AUROC = %.3f [%.3f,%.3f]" % (name, roc_auc_score(yd, pd), lo, hi))
print("\n(compare transfer to: ours+CG HaluEval-QA 0.889 / FaithEval 0.672 / RAGBench 0.626;"
      " static-ablation HaluEval-QA 0.727)")
