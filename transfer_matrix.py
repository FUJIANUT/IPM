"""Cross-dataset transfer matrix: train probe on dataset X, test on dataset Y, for all pairs.
Uses the standard logit+mech features (transfer-robust; copy-head ECS overfits the source).
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATASETS = {
    "RAGTruth": "data/features_ds_ragtruth.jsonl",
    "RAGBench": "data/features_ds_ragbench.jsonl",
    "HaluEval-QA": "data/features_ds_haluqa.jsonl",
    "HaluEval-Sum": "data/features_ds_halusum.jsonl",
    "HaluEval-Dia": "data/features_ds_haludia.jsonl",
    "FaithEval": "data/features_ds_faitheval.jsonl",
}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def split(recs):
    has = set(r.get("split", "") for r in recs)
    if "train" in has and "test" in has:
        return [r for r in recs if r["split"] == "train"], [r for r in recs if r["split"] == "test"]
    recs = sorted(recs, key=lambda r: str(r["id"]))
    k = int(0.7 * len(recs))
    return recs[:k], recs[k:]


def XY(recs, names):
    X = np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in recs], float))
    y = np.array([r["label"] for r in recs])
    return X, y


def main():
    os.chdir(os.path.expanduser("~/cgp"))
    loaded = {n: load(p) for n, p in DATASETS.items() if os.path.exists(p)}
    if not loaded:
        print("no dataset feature files found"); return
    names = sorted(set.intersection(*[set(recs[0]["features"].keys()) for recs in loaded.values()]))
    spl = {n: split(recs) for n, recs in loaded.items()}
    cols = list(loaded)
    print(f"Cross-dataset transfer AUROC (rows=train, cols=test); standard logit+mech, {len(names)} feats")
    hdr = "train\\test"
    print(hdr.ljust(16) + "".join(f"{c[:12]:>14}" for c in cols))
    for tr_n in cols:
        Xtr, ytr = XY(spl[tr_n][0], names)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        row = []
        for te_n in cols:
            Xte, yte = XY(spl[te_n][1], names)
            row.append(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
        print(f"{tr_n:<16}" + "".join(f"{a:>14.3f}" for a in row))
    print("\n(diagonal = in-domain; off-diagonal = zero-shot cross-dataset transfer)")


if __name__ == "__main__":
    main()
