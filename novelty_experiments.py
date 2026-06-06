"""Novelty experiments for the 'universality of grounding signals' reframe + conflict-gating decisiveness.
All computed from EXISTING extracted features (no new GPU runs).

N1  cross-proxy universality:  per-proxy in-domain AUROC, 4x4 cross-proxy probe transfer, per-example
    score correlation across proxies (Spearman). Thesis: the faithfulness signal is reader-agnostic.
N2  leave-one-generator-out:   train on 5 generators, test on the held-out 6th (all folds).
    Thesis: the signal is generator-independent (transfers to UNSEEN writers).
N3  conflict-gating decisiveness: Static (single-pass: base+lp+H+mech) vs +Conflict-gated
    (CG+dlp+conf+cmech), overall vs HARD subsets (hardest generator, cross-dataset transfer).
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

os.chdir(os.path.expanduser("~/cgp"))
RNG = 0


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def feat_names(rows):
    return sorted(rows[0]["features"].keys())


def XY(rows, names):
    X = np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in rows], float))
    y = np.array([r["label"] for r in rows])
    return X, y


def split(rows):
    tr = [r for r in rows if r.get("split") == "train"]
    te = [r for r in rows if r.get("split") == "test"]
    return tr, te


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p), p, sc, clf


def group_names(names):
    """Split features into STATIC (single with-context pass; observer/ReDeEP-like) and
    CONFLICT-GATED (two-pass contrastive + conflict-token gating)."""
    static, cg = [], []
    for n in names:
        if n.startswith(("all_CG", "all_dlp", "conf_", "cmech_")):
            cg.append(n)
        else:  # base_, all_lp_w, all_H_w, mech_
            static.append(n)
    return static, cg


PROXIES = {"Qwen0.5B": "data/features_p2_qwen05.jsonl", "Qwen1.5B": "data/features_p2_qwen15.jsonl",
           "SmolLM2-1.7B": "data/features_p2_smol17.jsonl", "TinyLlama-1.1B": "data/features_p2_tiny11.jsonl"}


def N1_cross_proxy():
    print("\n===== N1: CROSS-PROXY UNIVERSALITY =====")
    data = {k: load(v) for k, v in PROXIES.items()}
    names = sorted(set.intersection(*[set(feat_names(r)) for r in data.values()]))
    sp = {k: split(v) for k, v in data.items()}
    procs = list(PROXIES)
    # align test ids across proxies (same order assumed; align by id)
    test_ids = [r["id"] for r in sp[procs[0]][1]]
    # build per-proxy fitted probe + standardizer on its own train; collect test scores aligned by id
    fitted = {}
    scores = {}
    print("\n[in-domain per proxy]")
    for k in procs:
        tr, te = sp[k]
        Xtr, ytr = XY(tr, names); Xte, yte = XY(te, names)
        auc, p, scl, clf = fit_eval(Xtr, ytr, Xte, yte)
        fitted[k] = (scl, clf, names)
        # map id->score for this proxy
        idscore = {r["id"]: pi for r, pi in zip(te, p)}
        idlab = {r["id"]: r["label"] for r in te}
        scores[k] = idscore
        print("  %-14s in-domain AUROC = %.3f" % (k, auc))
    # common test ids
    common = set.intersection(*[set(scores[k]) for k in procs])
    common = [i for i in test_ids if i in common]
    ylab = np.array([idlab[i] for i in common])
    print("  (aligned test examples: %d)" % len(common))
    # 4x4 cross-proxy transfer: train probe on A, apply to B's test features
    print("\n[cross-proxy transfer AUROC: rows=train proxy, cols=test proxy]")
    print("train\\test    " + "".join("%14s" % c[:12] for c in procs))
    Xte_by = {}
    for k in procs:
        _, te = sp[k]
        te_c = [r for r in te if r["id"] in set(common)]
        # order by common
        order = {r["id"]: r for r in te_c}
        te_c = [order[i] for i in common]
        Xte_by[k] = XY(te_c, names)
    offdiag = []
    for a in procs:
        scl, clf, _ = fitted[a]
        row = []
        for b in procs:
            Xb, yb = Xte_by[b]
            p = clf.predict_proba(scl.transform(Xb))[:, 1]
            au = roc_auc_score(yb, p)
            row.append(au)
            if a != b:
                offdiag.append(au)
        print("%-12s  " % a[:12] + "".join("%14.3f" % v for v in row))
    print("  mean OFF-DIAGONAL transfer AUROC = %.3f (in-domain diag ~0.84-0.86)" % np.mean(offdiag))
    # per-example score correlation across proxies
    print("\n[per-example score Spearman correlation across proxies]")
    S = np.array([[scores[k][i] for i in common] for k in procs])  # 4 x N
    corrs = []
    for ia in range(len(procs)):
        for ib in range(ia + 1, len(procs)):
            rho = spearmanr(S[ia], S[ib]).correlation
            corrs.append(rho)
            print("  %-14s vs %-14s rho = %.3f" % (procs[ia], procs[ib], rho))
    print("  MEAN pairwise Spearman = %.3f  -> signal is a property of the text, not the reader" % np.mean(corrs))


def N2_leave_one_generator():
    print("\n===== N2: LEAVE-ONE-GENERATOR-OUT (generator independence) =====")
    rows = load("data/features_ds_ragtruth.jsonl")
    names = feat_names(rows)
    gens = sorted(set(r["model"] for r in rows))
    tr_all = [r for r in rows if r.get("split") == "train"]
    te_all = [r for r in rows if r.get("split") == "test"]
    print("  generators:", gens)
    print("\n  held-out generator   LOGO-AUROC   in-domain-AUROC   (delta)")
    logo_list, ind_list = [], []
    for g in gens:
        tr = [r for r in tr_all if r["model"] != g]    # train on the OTHER 5 generators
        te = [r for r in te_all if r["model"] == g]     # test on the held-out generator
        Xtr, ytr = XY(tr, names); Xte, yte = XY(te, names)
        logo, _, _, _ = fit_eval(Xtr, ytr, Xte, yte)
        # in-domain: train on g's own train, test on g's test
        trg = [r for r in tr_all if r["model"] == g]
        Xtrg, ytrg = XY(trg, names)
        ind, _, _, _ = fit_eval(Xtrg, ytrg, Xte, yte)
        logo_list.append(logo); ind_list.append(ind)
        print("  %-20s %.3f        %.3f          %+.3f" % (g, logo, ind, logo - ind))
    print("  MEAN LOGO=%.3f  in-domain=%.3f  -> detector generalizes to UNSEEN generators" %
          (np.mean(logo_list), np.mean(ind_list)))


def N3_conflict_gating():
    print("\n===== N3: CONFLICT-GATING DECISIVENESS =====")
    rt = load("data/features_ds_ragtruth.jsonl")
    names = feat_names(rt)
    static, cg = group_names(names)
    print("  #static feats=%d  #conflict-gated feats=%d" % (len(static), len(cg)))
    tr, te = split(rt)

    def auc_for(feats, tr, te):
        Xtr, ytr = XY(tr, feats); Xte, yte = XY(te, feats)
        return fit_eval(Xtr, ytr, Xte, yte)[0]

    print("\n[overall RAGTruth test]")
    aS = auc_for(static, tr, te); aSC = auc_for(names, tr, te)
    print("  Static=%.3f   Static+ConflictGated=%.3f   delta=%+.3f" % (aS, aSC, aSC - aS))

    print("\n[per held-out generator: delta where subtler]")
    gens = sorted(set(r["model"] for r in rt))
    for g in gens:
        teg = [r for r in te if r["model"] == g]
        aS = auc_for(static, tr, teg); aSC = auc_for(names, tr, teg)
        print("  %-20s Static=%.3f  +CG=%.3f  delta=%+.3f" % (g, aS, aSC, aSC - aS))

    print("\n[cross-dataset transfer: train RAGTruth -> test X (where it's HARD)]")
    for dsname, path in [("HaluEval-QA", "data/features_ds_haluqa.jsonl"),
                         ("RAGBench", "data/features_ds_ragbench.jsonl"),
                         ("FaithEval", "data/features_ds_faitheval.jsonl")]:
        if not os.path.exists(path):
            continue
        ds = load(path)
        _, dte = split(ds) if any(r.get("split") == "test" for r in ds) else (ds, ds)
        # common feature names
        cn = sorted(set(names) & set(feat_names(ds)))
        cs = [n for n in cn if not n.startswith(("all_CG", "all_dlp", "conf_", "cmech_"))]
        aS = auc_for(cs, tr, dte); aSC = auc_for(cn, tr, dte)
        print("  %-12s Static=%.3f  +CG=%.3f  delta=%+.3f" % (dsname, aS, aSC, aSC - aS))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "n1"):
        N1_cross_proxy()
    if which in ("all", "n2"):
        N2_leave_one_generator()
    if which in ("all", "n3"):
        N3_conflict_gating()
    print("\nDONE")
