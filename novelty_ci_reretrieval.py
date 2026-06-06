"""(1) Bootstrap CIs for N1 (cross-proxy) and N2 (leave-one-generator-out).
(2) Real re-retrieval closed loop on the robustness data: each base example has a GOLD (faithful) and
    DEGRADED (hallucinated) version with REAL per-condition labels + detector features. We serve a
    stressed retrieval stream, let the detector decide which answers to re-retrieve (degraded->gold)
    under a budget, and measure end-to-end faithfulness vs budget against random and oracle policies.
All from existing features; no GPU.
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

os.chdir(os.path.expanduser("~/cgp"))
B = 2000


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def names_of(rows):
    return sorted(rows[0]["features"].keys())


def XY(rows, names):
    X = np.nan_to_num(np.array([[r["features"].get(n, 0.0) for n in names] for r in rows], float))
    return X, np.array([r["label"] for r in rows])


def fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return sc, clf


def score(sc, clf, X):
    return clf.predict_proba(sc.transform(X))[:, 1]


def boot_auc_ci(y, s, b=B, seed=0):
    y = np.asarray(y); s = np.asarray(s); rng = np.random.RandomState(seed); out = []
    for _ in range(b):
        idx = rng.randint(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


PROX = {"Qwen0.5B": "data/features_p2_qwen05.jsonl", "Qwen1.5B": "data/features_p2_qwen15.jsonl",
        "SmolLM2-1.7B": "data/features_p2_smol17.jsonl", "TinyLlama-1.1B": "data/features_p2_tiny11.jsonl"}


def ci_n1():
    print("\n===== N1 CIs: cross-proxy =====")
    data = {k: load(v) for k, v in PROX.items()}
    names = sorted(set.intersection(*[set(names_of(v)) for v in data.values()]))
    procs = list(PROX)
    sc_test = {}
    for k in procs:
        tr = [r for r in data[k] if r.get("split") == "train"]
        te = [r for r in data[k] if r.get("split") == "test"]
        Xtr, ytr = XY(tr, names); Xte, yte = XY(te, names)
        s, c = fit(Xtr, ytr); p = score(s, c, Xte)
        lo, hi = boot_auc_ci(yte, p)
        print("  %-14s in-domain AUROC=%.3f [%.3f,%.3f]" % (k, roc_auc_score(yte, p), lo, hi))
        sc_test[k] = {r["id"]: pi for r, pi in zip(te, p)}
        sc_test[k + "_lab"] = {r["id"]: r["label"] for r in te}
    common = sorted(set.intersection(*[set(sc_test[k]) for k in procs]))
    S = np.array([[sc_test[k][i] for i in common] for k in procs])
    # bootstrap mean pairwise Spearman
    rng = np.random.RandomState(0); means = []
    for _ in range(B):
        idx = rng.randint(0, S.shape[1], S.shape[1])
        rr = [spearmanr(S[a][idx], S[b][idx]).correlation
              for a in range(len(procs)) for b in range(a + 1, len(procs))]
        means.append(np.mean(rr))
    print("  mean pairwise Spearman rho=%.3f [%.3f,%.3f]" %
          (np.mean([spearmanr(S[a], S[b]).correlation for a in range(len(procs)) for b in range(a + 1, len(procs))]),
           np.percentile(means, 2.5), np.percentile(means, 97.5)))


def ci_n2():
    print("\n===== N2 CIs: leave-one-generator-out =====")
    rows = load("data/features_ds_ragtruth.jsonl"); names = names_of(rows)
    gens = sorted(set(r["model"] for r in rows))
    tr_all = [r for r in rows if r.get("split") == "train"]
    te_all = [r for r in rows if r.get("split") == "test"]
    logos = []
    for g in gens:
        tr = [r for r in tr_all if r["model"] != g]; te = [r for r in te_all if r["model"] == g]
        Xtr, ytr = XY(tr, names); Xte, yte = XY(te, names)
        s, c = fit(Xtr, ytr); p = score(s, c, Xte)
        lo, hi = boot_auc_ci(yte, p)
        print("  held-out %-20s LOGO AUROC=%.3f [%.3f,%.3f]" % (g, roc_auc_score(yte, p), lo, hi))
        logos.append((g, yte, p))
    # mean LOGO CI (bootstrap over generators' pooled examples, stratified by gen)
    rng = np.random.RandomState(0); ms = []
    for _ in range(B):
        accs = []
        for g, y, p in logos:
            idx = rng.randint(0, len(y), len(y))
            if len(set(y[idx])) < 2:
                continue
            accs.append(roc_auc_score(y[idx], p[idx]))
        ms.append(np.mean(accs))
    print("  MEAN LOGO=%.3f [%.3f,%.3f]  (in-domain mean 0.846)" %
          (np.mean([roc_auc_score(y, p) for _, y, p in logos]), np.percentile(ms, 2.5), np.percentile(ms, 97.5)))


def reretrieval(p_fail=0.30):
    print("\n===== N7: REAL RE-RETRIEVAL CLOSED LOOP (robustness data) =====")
    rob = load("data/features_robust.jsonl")
    rnames = names_of(rob)
    rt = load("data/features_ds_ragtruth.jsonl")
    cn = sorted(set(names_of(rt)) & set(rnames))
    tr = [r for r in rt if r.get("split") == "train"]
    Xtr, ytr = XY(tr, cn); s, c = fit(Xtr, ytr)
    # score every robustness example; split by condition via id suffix
    for r in rob:
        r["_cond"] = r["id"].split("_")[-1]
        r["_base"] = r["id"].rsplit("_", 1)[0]
    Xr, yr0 = XY(rob, cn); pr0 = score(s, c, Xr)
    for r, sv in zip(rob, pr0):
        r["_score"] = float(sv)
    auc = roc_auc_score(yr0, pr0)
    conds = sorted(set(r["_cond"] for r in rob))
    print("  detector AUROC (gold vs degraded) = %.3f ; conditions=%s" % (auc, conds))
    gold = {r["_base"]: r for r in rob if r["_cond"] == "gold"}
    degr = [r for r in rob if r["_cond"] != "gold"]
    # Build a realistic served stream: each base query has prob p_fail of bad retrieval (a random degraded
    # version) else its gold version. Re-retrieving a served answer -> its gold version (REAL faithfulness flip).
    rng = np.random.RandomState(0)
    by_base = {}
    for r in degr:
        by_base.setdefault(r["_base"], []).append(r)
    stream = []
    for b, g in gold.items():
        if rng.random() < p_fail and b in by_base:
            served = by_base[b][rng.randint(0, len(by_base[b]))]   # a degraded answer (hallucinated)
        else:
            served = g                                             # gold answer (faithful)
        stream.append(served)
    y = np.array([r["label"] for r in stream])          # served-answer faithfulness label
    sc = np.array([r["_score"] for r in stream])         # detector score of the served answer
    N = len(stream); base = float((y == 0).mean())
    print("  served stream: N=%d  p_fail=%.2f  base-faithful=%.3f" % (N, p_fail, base))
    order_det = np.argsort(-sc); order_rand = rng.permutation(N); order_oracle = np.argsort(-y)

    def curve(order):
        base_f = int((y == 0).sum()); ks = np.linspace(0, N, 51).astype(int); cov = []; fa = []
        for k in ks:
            sel = order[:k]
            cov.append(k / N); fa.append((base_f + int((y[sel] == 1).sum())) / N)  # re-retrieved hallu -> faithful
        return np.array(cov), np.array(fa)
    cd, fd = curve(order_det); cr, fr = curve(order_rand); co, fo = curve(order_oracle)

    def bud(cov, fa, t):
        ok = np.where(fa >= t)[0]; return float(cov[ok[0]]) if len(ok) else None
    for t in (0.90, 0.95, 0.99):
        bd, br, bo = bud(cd, fd, t), bud(cr, fr, t), bud(co, fo, t)
        save = (br - bd) / br * 100 if (bd and br) else float("nan")
        print("  reach %.0f%% faithful: detector=%s  random=%s  oracle=%s  (detector saves %.0f%% vs random)" %
              (t * 100, "%.2f" % bd if bd else "n/a", "%.2f" % br if br else "n/a",
               "%.2f" % bo if bo else "n/a", save))
    json.dump({"cov": cd.tolist(), "det": fd.tolist(), "rand": fr.tolist(), "oracle": fo.tolist(),
               "base": base, "auc": auc, "p_fail": p_fail}, open("data/reretrieval_points.json", "w"))
    print("  saved data/reretrieval_points.json")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ci"):
        ci_n1(); ci_n2()
    if which in ("all", "rr"):
        reretrieval(0.30)
    print("\nDONE")
