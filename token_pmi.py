"""Token-level PMI distributions for supported vs unsupported tokens (validates Proposition 1).
tokens.jsonl has per-token features in FEAT_ORDER (dlp = Delta-ell = context PMI at index 3) and per-token
0/1 hallucination labels. We split dlp by label, compute the margin, the empirical token-AUROC of dlp, and
the Gaussian-predicted AUROC Phi((mu+ - mu-)/(sigma*sqrt2)) from Proposition 1. Saves histogram data.
"""
import json, os, math
import numpy as np
from sklearn.metrics import roc_auc_score

os.chdir(os.path.expanduser("~/cgp"))
FEAT = ["lp_w", "H_w", "CG", "dlp", "conflict", "ffn_norm", "ffn_ratio", "attn2ctx", "attn2ctx_max"]
DLP = FEAT.index("dlp")

pos, neg = [], []   # dlp for unsupported (label 1) and supported (label 0)
keys_printed = False
n_rec = 0
for line in open("data/tokens.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    if not keys_printed:
        print("record keys:", list(r.keys())); keys_printed = True
    if r.get("split") not in (None, "test"):
        continue
    feats = np.asarray(r.get("feats") or r.get("features"))
    labs = np.asarray(r.get("tok_labels"))
    if feats.ndim != 2 or feats.shape[0] != len(labs) or feats.shape[0] == 0:
        continue
    dlp = feats[:, DLP]
    neg.append(dlp[labs == 0]); pos.append(dlp[labs == 1])
    n_rec += 1

neg = np.concatenate(neg); pos = np.concatenate(pos)
# clip extreme tails for stable stats/plot
def clip(x):
    lo, hi = np.percentile(x, [0.5, 99.5]); return np.clip(x, lo, hi)
negc, posc = clip(neg), clip(pos)
mu_p, mu_n = pos.mean(), neg.mean()           # mu_+ = supported? NO: pos=unsupported. define below
# supported = label 0 (neg list), unsupported = label 1 (pos list)
mu_sup, mu_uns = neg.mean(), pos.mean()
sd = math.sqrt((neg.var() + pos.var()) / 2)
y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
score = np.concatenate([neg, pos])            # higher dlp -> more supported; AUROC for unsupported uses -dlp
auc = roc_auc_score(y, -score)                # detect unsupported by LOW dlp
gauss_auc = 0.5 * (1 + math.erf((mu_sup - mu_uns) / (sd * 2)))  # Phi(gap/(sd*sqrt2))
print("records=%d  supported tokens=%d  unsupported tokens=%d" % (n_rec, len(neg), len(pos)))
print("mu_supported=%.3f  mu_unsupported=%.3f  margin=%.3f  pooled_sd=%.3f" %
      (mu_sup, mu_uns, mu_sup - mu_uns, sd))
print("empirical token-AUROC(dlp)=%.3f   Gaussian-predicted Phi(gap/sd/sqrt2)=%.3f" % (auc, gauss_auc))
# histogram data (shared bins)
lo = float(min(negc.min(), posc.min())); hi = float(max(negc.max(), posc.max()))
bins = np.linspace(lo, hi, 61)
hsup, _ = np.histogram(negc, bins=bins, density=True)
huns, _ = np.histogram(posc, bins=bins, density=True)
json.dump({"bins": bins.tolist(), "sup": hsup.tolist(), "uns": huns.tolist(),
           "mu_sup": float(mu_sup), "mu_uns": float(mu_uns), "sd": float(sd),
           "auc": float(auc), "gauss_auc": float(gauss_auc),
           "n_sup": int(len(neg)), "n_uns": int(len(pos))},
          open("data/token_pmi_hist.json", "w"))
print("saved data/token_pmi_hist.json")
