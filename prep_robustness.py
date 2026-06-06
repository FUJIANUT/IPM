"""Phase 6 (v2) robustness gradient: for each faithful RAGTruth response, build several context
conditions of increasing degradation (same response text). A grounding detector should raise its
score monotonically: gold < injected-misinfo < partial-context < random-distractor.
Condition is encoded in the id suffix (_gold/_inject/_partial/_dist) since the extractor drops extra fields.
"""
import json, os, argparse, random, re


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))


def sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/robust_examples.jsonl"))
    ap.add_argument("--n", type=int, default=700)
    args = ap.parse_args()
    te = [r for r in load(args.examples) if r["split"] == "test"]
    faithful = [r for r in te if r["label"] == 0][:args.n]
    pool_ctx = [(as_text(r["context"]), r.get("task_type", "")) for r in te]
    sent_pool = []
    for c, _ in pool_ctx[:400]:
        sent_pool += sents(c)[:5]
    random.seed(0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def rec(r, suf, ctx, prompt, lab):
        return json.dumps({"id": f"{r['id']}_{suf}", "model": r.get("model", ""), "task_type": r.get("task_type", ""),
                           "split": "test", "context": ctx, "prompt": prompt, "response": as_text(r["response"]),
                           "label": lab}, ensure_ascii=False)

    n = 0
    with open(args.out, "w") as f:
        for r in faithful:
            g = as_text(r["context"]); pr = as_text(r["prompt"]); tt = r.get("task_type", "")
            def sub(ctx):
                return pr.replace(g, ctx) if g and g in pr else (tt + " " + ctx)
            # gold
            f.write(rec(r, "gold", g, pr, 0) + "\n")
            # inject: replace one sentence with a random foreign sentence
            ss = sents(g)
            if len(ss) >= 2 and sent_pool:
                j = random.randrange(len(ss)); ss2 = ss[:]; ss2[j] = random.choice(sent_pool)
                ictx = " ".join(ss2)
                f.write(rec(r, "inject", ictx, sub(ictx), 1) + "\n"); n += 1
            # partial: keep only first ~40% of the context
            pctx = g[:max(1, int(0.4 * len(g)))]
            f.write(rec(r, "partial", pctx, sub(pctx), 1) + "\n")
            # distractor: a random different context of the same task type
            cand = [c for c, t in pool_ctx if t == tt and c and c != g] or [c for c, _ in pool_ctx if c and c != g]
            dctx = random.choice(cand)
            f.write(rec(r, "dist", dctx, sub(dctx), 1) + "\n")
            n += 3
    print(f"wrote {n} robustness examples ({len(faithful)} faithful x gold/inject/partial/dist) -> {args.out}")


if __name__ == "__main__":
    main()
