"""Phase 5 span localization with a sequence head: a small BiLSTM over the per-token mechanistic
features (vs the per-token logistic baseline at token AUROC 0.718). Token + example level.
"""
import json, os, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


class Tagger(nn.Module):
    def __init__(self, d=9, h=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lstm = nn.LSTM(d, h, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * h, 1)

    def forward(self, x):
        return self.fc(self.lstm(self.norm(x))[0]).squeeze(-1)


def batches(recs, mean, std, bs, device, shuffle=True):
    idx = np.arange(len(recs))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [recs[j] for j in idx[i:i + bs] if recs[j]["feats"]]
        if not chunk:
            continue
        T = max(len(r["feats"]) for r in chunk)
        X = np.zeros((len(chunk), T, 9), np.float32)
        Y = np.zeros((len(chunk), T), np.float32)
        M = np.zeros((len(chunk), T), np.float32)
        for b, r in enumerate(chunk):
            t = len(r["feats"])
            X[b, :t] = (np.array(r["feats"]) - mean) / std
            Y[b, :t] = r["tok_labels"]
            M[b, :t] = 1
        yield (torch.tensor(X, device=device), torch.tensor(Y, device=device),
               torch.tensor(M, device=device), chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=os.path.expanduser("~/cgp/data/tokens.jsonl"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    recs = load(args.tokens)
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    allf = np.concatenate([np.array(r["feats"]) for r in tr if r["feats"]])
    mean, std = allf.mean(0), allf.std(0) + 1e-6
    pos = sum(sum(r["tok_labels"]) for r in tr); tot = sum(len(r["tok_labels"]) for r in tr)
    pw = torch.tensor((tot - pos) / pos, device=args.device)

    model = Tagger().to(args.device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    for ep in range(args.epochs):
        model.train(); tl = 0.0
        for X, Y, M, _ in batches(tr, mean, std, 64, args.device):
            opt.zero_grad()
            l = (lossf(model(X), Y) * M).sum() / M.sum()
            l.backward(); opt.step(); tl += l.item()
        print(f"epoch {ep+1} loss {tl:.1f}", flush=True)

    model.eval()
    tok_p, tok_y, ex_p, ex_y = [], [], [], []
    with torch.no_grad():
        for X, Y, M, chunk in batches(te, mean, std, 64, args.device, shuffle=False):
            prob = torch.sigmoid(model(X)).cpu().numpy()
            for b, r in enumerate(chunk):
                t = len(r["feats"]); p = prob[b, :t]
                tok_p += p.tolist(); tok_y += r["tok_labels"]
                ex_p.append(float(p.max()) if t > 0 else 0.0); ex_y.append(r["ex_label"])
    tok_p, tok_y = np.array(tok_p), np.array(tok_y)
    ex_p, ex_y = np.array(ex_p), np.array(ex_y)
    print("\n== TOKEN-LEVEL (BiLSTM) ==")
    print(f"AUROC={roc_auc_score(tok_y,tok_p):.3f}  F1@0.5={f1_score(tok_y,(tok_p>0.5).astype(int),zero_division=0):.3f}")
    print("== EXAMPLE-LEVEL (max-token agg) ==")
    print(f"AUROC={roc_auc_score(ex_y,ex_p):.3f}  F1@0.5={f1_score(ex_y,(ex_p>0.5).astype(int),zero_division=0):.3f}")
    print("(per-token logistic baseline was token AUROC 0.718 / example 0.699)")


if __name__ == "__main__":
    main()
