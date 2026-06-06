"""Train logistic-regression probes on extracted features and compare feature sets.

Key comparison: does CONFLICT-GATED aggregation beat plain ALL-TOKEN aggregation
(and a perplexity baseline) for detecting RAGTruth hallucinations?
Uses RAGTruth's official train/test split. Also breaks results down by task and model.
"""
import json, os, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def Xy(recs, names):
    X = np.array([[r["features"].get(n, 0.0) for n in names] for r in recs], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([r["label"] for r in recs], dtype=int)
    return X, y


def fit_eval(tr, te, names):
    Xtr, ytr = Xy(tr, names)
    Xte, yte = Xy(te, names)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    prob = clf.predict_proba(sc.transform(Xte))[:, 1]
    return {
        "AUROC": roc_auc_score(yte, prob) if len(set(yte)) > 1 else float("nan"),
        "AUPRC": average_precision_score(yte, prob) if len(set(yte)) > 1 else float("nan"),
        "F1": f1_score(yte, (prob > 0.5).astype(int), zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.expanduser("~/cgp/data/features.jsonl"))
    args = ap.parse_args()
    recs = load_jsonl(args.features)
    allnames = sorted(recs[0]["features"].keys())
    all_tok = [n for n in allnames if n.startswith("all_")]
    conf = [n for n in allnames if n.startswith("conf_")]
    mech = [n for n in allnames if n.startswith("mech_")]
    cmech = [n for n in allnames if n.startswith("cmech_")]
    logit = all_tok + conf
    sets = {
        "baseline_ppl": ["base_meanlp", "base_minlp"],
        "all_token": all_tok,
        "conflict_gated": conf,
        "logit(all+conf)": logit,
    }
    if mech:
        sets["mech_only"] = mech + cmech
        sets["mech_attn"] = [n for n in mech + cmech if "attn" in n]
        sets["mech_ffn"] = [n for n in mech + cmech if "ffn" in n]
        sets["logit+mech"] = logit + mech + cmech
    sets["everything"] = allnames
    best = sets.get("logit+mech", logit)

    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    if len(te) == 0 or len(tr) == 0:
        import random
        random.seed(0); random.shuffle(recs)
        k = int(0.8 * len(recs)); tr, te = recs[:k], recs[k:]
        print("[warn] official split unavailable, using random 80/20")

    print(f"train={len(tr)} test={len(te)} "
          f"test_hallu_rate={np.mean([r['label'] for r in te]):.3f}\n")
    print(f"{'feature_set':<18}{'nfeat':>6}{'AUROC':>8}{'AUPRC':>8}{'F1':>7}")
    print("-" * 47)
    for sname, names in sets.items():
        m = fit_eval(tr, te, names)
        print(f"{sname:<18}{len(names):>6}{m['AUROC']:>8.3f}{m['AUPRC']:>8.3f}{m['F1']:>7.3f}")

    # breakdown by task_type: logit-only vs best (with mechanistic if present)
    print("\nAUROC by task_type (logit(all+conf) vs best):")
    for task in sorted(set(r["task_type"] for r in te)):
        te_t = [r for r in te if r["task_type"] == task]
        if len(set(r["label"] for r in te_t)) < 2:
            continue
        l = fit_eval(tr, te_t, logit)["AUROC"]
        b = fit_eval(tr, te_t, best)["AUROC"]
        print(f"  {task:<12} logit={l:.3f}  best={b:.3f}  (n={len(te_t)})")

    # breakdown by generator model: can the 0.5B proxy flag a big model's hallucinations?
    print("\nAUROC by generator model (best probe):")
    for mdl in sorted(set(r["model"] for r in te)):
        te_m = [r for r in te if r["model"] == mdl]
        if len(set(r["label"] for r in te_m)) < 2:
            continue
        m = fit_eval(tr, te_m, best)
        rate = np.mean([r["label"] for r in te_m])
        print(f"  {mdl:<22} AUROC={m['AUROC']:.3f}  F1={m['F1']:.3f}  "
              f"hallu_rate={rate:.2f}  (n={len(te_m)})")


if __name__ == "__main__":
    main()
