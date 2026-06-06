"""Unified comparison harness with statistical rigor.

Reads per-method score files (data/scores/<method>.jsonl, each line {id,label,score} on the test
set; score = hallucination score, higher = more hallucinated), aligns by id, and reports
AUROC/AUPRC/F1 with bootstrap 95% CIs and a paired-bootstrap p-value vs a reference method.
"""
import json, os, glob, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_scores(path):
    d = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            d[str(r["id"])] = (int(r["label"]), float(r["score"]))
    return d


def boot_metric(y, s, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def paired_boot_pvalue(y, s_ref, s_other, fn=roc_auc_score, B=2000, seed=0):
    """One-sided-ish: p that ref is NOT better than other (then x2 for two-sided)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], s_ref[idx]) - fn(y[idx], s_other[idx]))
    diffs = np.array(diffs)
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(min(p, 1.0)), float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", default=os.path.expanduser("~/cgp/data/scores"))
    ap.add_argument("--ref", default="ours")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.scores_dir, "*.jsonl")))
    methods = {os.path.splitext(os.path.basename(f))[0]: load_scores(f) for f in files}
    if not methods:
        print("no score files in", args.scores_dir); return
    common = set.intersection(*[set(d.keys()) for d in methods.values()])
    ids = sorted(common)
    print(f"methods: {list(methods)}\naligned test examples: {len(ids)}")
    y = np.array([methods[list(methods)[0]][i][0] for i in ids])

    S = {m: np.array([methods[m][i][1] for i in ids]) for m in methods}
    ref = args.ref if args.ref in S else list(S)[0]
    print(f"reference = {ref}  (hallu rate {y.mean():.3f})\n")
    print(f"{'method':<24}{'AUROC [95% CI]':<26}{'AUPRC':<18}{'F1':<8}{'p_vs_ref':<8}")
    print("-" * 84)
    rows = []
    for m in methods:
        s = S[m]
        au, al, ah = boot_metric(y, s, roc_auc_score, args.B)
        pr, pl, ph = boot_metric(y, s, average_precision_score, args.B)
        # F1 at best train-free 0.5 on minmax-normalized score
        sn = (s - s.min()) / (s.max() - s.min() + 1e-9)
        f1 = f1_score(y, (sn > 0.5).astype(int), zero_division=0)
        if m == ref:
            pv = "-"
        else:
            p, _ = paired_boot_pvalue(y, S[ref], s, roc_auc_score, args.B)
            pv = f"{p:.3f}"
        rows.append((au, m, f"{au:.3f} [{al:.3f},{ah:.3f}]", f"{pr:.3f} [{pl:.3f},{ph:.3f}]", f"{f1:.3f}", pv))
    for au, m, auc_s, prc_s, f1_s, pv in sorted(rows, reverse=True):
        print(f"{m:<24}{auc_s:<26}{prc_s:<18}{f1_s:<8}{pv:<8}")


if __name__ == "__main__":
    main()
