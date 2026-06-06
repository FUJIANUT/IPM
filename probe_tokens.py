"""Token-level hallucination probe. Trains a per-token logistic classifier and reports
token-level metrics (comparable to LettuceDetect) plus example-level via max-token aggregation.
"""
import json, os, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=os.path.expanduser("~/cgp/data/tokens.jsonl"))
    args = ap.parse_args()
    recs = load_jsonl(args.tokens)
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]

    def flat(rs):
        X, y = [], []
        for r in rs:
            for row, lab in zip(r["feats"], r["tok_labels"]):
                X.append(row); y.append(lab)
        return np.nan_to_num(np.array(X, float)), np.array(y)

    Xtr, ytr = flat(tr)
    Xte, yte = flat(te)
    print(f"train tokens={len(ytr)} ({ytr.mean():.3f} hallucinated)  "
          f"test tokens={len(yte)} ({yte.mean():.3f} hallucinated)")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    ptr = clf.predict_proba(sc.transform(Xtr))[:, 1]
    pte = clf.predict_proba(sc.transform(Xte))[:, 1]

    print("\n== TOKEN-LEVEL ==")
    print(f"AUROC={roc_auc_score(yte,pte):.3f}  AUPRC={average_precision_score(yte,pte):.3f}  "
          f"F1@0.5={f1_score(yte,(pte>0.5).astype(int),zero_division=0):.3f}")
    prec, rec, th = precision_recall_curve(ytr, ptr)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    bt = th[int(np.argmax(f1s[:-1]))]
    print(f"F1@best (train-tuned thr={bt:.2f}) on test = "
          f"{f1_score(yte,(pte>bt).astype(int),zero_division=0):.3f}")

    # example-level: max token prob per example
    pe, ye = [], []
    i = 0
    for r in te:
        T = len(r["tok_labels"])
        probs = pte[i:i + T]; i += T
        pe.append(float(probs.max()) if T > 0 else 0.0)
        ye.append(r["ex_label"])
    pe, ye = np.array(pe), np.array(ye)
    print("\n== EXAMPLE-LEVEL (max-token aggregation) ==")
    print(f"AUROC={roc_auc_score(ye,pe):.3f}  AUPRC={average_precision_score(ye,pe):.3f}  "
          f"F1@0.5={f1_score(ye,(pe>0.5).astype(int),zero_division=0):.3f}")

    # feature weights (interpretability)
    names = ["lp_w", "H_w", "CG", "dlp", "conflict", "ffn_norm", "ffn_ratio", "attn2ctx", "attn2ctx_max"]
    w = clf.coef_[0]
    print("\ntoken-probe weights:", {n: round(float(wi), 2) for n, wi in zip(names, w)})


if __name__ == "__main__":
    main()
