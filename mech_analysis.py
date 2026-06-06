"""Phase 5 mechanistic interpretability: where are the copy heads, and do the two ReDeEP legs
behave as predicted? (copy heads attend LESS to context on hallucinations; FFN writes MORE.)
"""
import os, json
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    os.chdir(os.path.expanduser("~/cgp"))
    recs = load("data/features_mech2.jsonl")
    nlayers, nheads = recs[0]["nlayers"], recs[0]["nheads"]
    tr = [r for r in recs if r["split"] == "train"]
    y = np.array([r["label"] for r in tr])
    H = np.array([r["head_a2c"] for r in tr])
    auroc = np.array([roc_auc_score(y, H[:, j]) if np.std(H[:, j]) > 1e-9 else 0.5 for j in range(H.shape[1])])
    score = np.abs(auroc - 0.5)
    layers = np.arange(H.shape[1]) // nheads
    top = np.argsort(score)[::-1][:32]

    print(f"proxy: {nlayers} layers x {nheads} heads")
    print("\nTop-32 copy heads by |AUROC-0.5| — layer histogram:")
    c = Counter(layers[top].tolist())
    for l in sorted(c):
        print(f"  layer {l:>2}: {'#' * c[l]} ({c[l]})")
    frac_late = np.mean(layers[top] >= nlayers / 2)
    print(f"  -> {frac_late:.0%} of top copy heads are in the latter half of layers")

    print("\nTop-6 copy heads: mean attention-to-context (faithful vs hallucinated):")
    for j in top[:6]:
        m0, m1 = H[y == 0, j].mean(), H[y == 1, j].mean()
        print(f"  L{j // nheads:>2} H{j % nheads:<2}: faithful={m0:.4f}  hallucinated={m1:.4f}  (Δ={m1 - m0:+.4f})")

    fn0 = np.array([r["features"]["mech_ffnnorm_mean"] for r in tr if r["label"] == 0])
    fn1 = np.array([r["features"]["mech_ffnnorm_mean"] for r in tr if r["label"] == 1])
    a0 = np.array([r["features"]["mech_attn2ctx_mean"] for r in tr if r["label"] == 0])
    a1 = np.array([r["features"]["mech_attn2ctx_mean"] for r in tr if r["label"] == 1])
    print("\nReDeEP legs (means, faithful vs hallucinated):")
    print(f"  FFN-write (PKS)        faithful={fn0.mean():.3f}  hallucinated={fn1.mean():.3f}  "
          f"({'HIGHER on hallu' if fn1.mean() > fn0.mean() else 'lower on hallu'})")
    print(f"  attn-to-context (ECS)  faithful={a0.mean():.4f}  hallucinated={a1.mean():.4f}  "
          f"({'LOWER on hallu' if a1.mean() < a0.mean() else 'higher on hallu'})")


if __name__ == "__main__":
    main()
