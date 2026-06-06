"""Cross-dataset transfer probe (E-step).

Train on RAGTruth (features_mech2), evaluate ZERO-SHOT on another dataset's features
(e.g. HaluEval-QA). Copy heads are selected on RAGTruth-train and applied to the test set.
Also reports an in-domain reference on the test dataset.
"""
import json, os, argparse, random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)   # RAGTruth features_mech2.jsonl
    ap.add_argument("--test", required=True)    # cross-dataset features (HaluEval)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    TR = load_jsonl(args.train)
    TE = load_jsonl(args.test)
    tr = [r for r in TR if r["split"] == "train"]
    te = TE
    std = sorted(tr[0]["features"].keys())
    nheads = tr[0]["nheads"]
    ytr = np.array([r["label"] for r in tr])
    yte = np.array([r["label"] for r in te])

    def feats(rs, names):
        return np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in rs], dtype=float))

    Hall_tr = np.array([r["head_a2c"] for r in tr]); Hall_te = np.array([r["head_a2c"] for r in te])
    Hcf_tr = np.array([r["head_a2c_conf"] for r in tr]); Hcf_te = np.array([r["head_a2c_conf"] for r in te])
    scores = np.zeros(Hall_tr.shape[1])
    for j in range(Hall_tr.shape[1]):
        if np.std(Hall_tr[:, j]) > 1e-9:
            scores[j] = abs(roc_auc_score(ytr, Hall_tr[:, j]) - 0.5)
    sel = np.argsort(scores)[::-1][:args.topk]

    def ecs(Hall, Hcf):
        ms, cs = Hall[:, sel], Hcf[:, sel]
        return np.column_stack([ms.mean(1), ms.max(1), ms.min(1), cs.mean(1), cs.min(1), ms, cs])

    def ev(name, Xtr, ytr_, Xte, yte_):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr_)
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        print(f"{name:<34}{roc_auc_score(yte_,p):>8.3f}{average_precision_score(yte_,p):>8.3f}"
              f"{f1_score(yte_,(p>0.5).astype(int),zero_division=0):>7.3f}")

    print(f"TRANSFER  train=RAGTruth({len(tr)})  ->  test=cross-dataset({len(te)}, "
          f"hallu_rate={yte.mean():.2f}, ctx_found={np.mean([r.get('ctx_found',False) for r in te]):.2f})")
    print(f"{'feature_set':<34}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}")
    print("-" * 57)
    ev("standard (logit+mech)", feats(tr, std), ytr, feats(te, std), yte)
    ev(f"ALL + copyhead ECS (K={args.topk})",
       np.column_stack([feats(tr, std), ecs(Hall_tr, Hcf_tr)]), ytr,
       np.column_stack([feats(te, std), ecs(Hall_te, Hcf_te)]), yte)

    # in-domain reference on the test dataset (70/30 split)
    idx = list(range(len(te)))
    random.seed(0)
    random.shuffle(idx)
    k = int(0.7 * len(te))
    itr = [te[i] for i in idx[:k]]
    ite = [te[i] for i in idx[k:]]
    yitr = np.array([r["label"] for r in itr]); yite = np.array([r["label"] for r in ite])
    print("\n[reference] in-domain on test dataset (70/30 split):")
    print(f"{'feature_set':<34}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}")
    ev("standard (logit+mech)", feats(itr, std), yitr, feats(ite, std), yite)


if __name__ == "__main__":
    main()
