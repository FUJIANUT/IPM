"""Prepare RAGBench (rungalileo/ragbench) — naturally-constructed RAG responses with adherence
labels — into the RAGTruth example schema. label = 0 if adherence_score (faithful) else 1.
Combines several domain configs for diversity.
"""
import json, os, argparse
import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="hotpotqa,covidqa,pubmedqa,finqa,expertqa,msmarco")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/ragbench_examples.jsonl"))
    ap.add_argument("--per_config", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for cfg in args.configs.split(","):
            try:
                d = datasets.load_dataset("rungalileo/ragbench", cfg, split="test")
            except Exception as e:
                print(f"  {cfg}: load err {str(e)[:100]}"); continue
            c = 0
            for ex in d:
                adh = ex.get("adherence_score")
                if adh is None:
                    continue
                docs = ex.get("documents") or []
                ctx = "\n".join(docs) if isinstance(docs, list) else str(docs)
                q = ex.get("question", "") or ""
                resp = ex.get("response", "") or ""
                if not ctx.strip() or not resp.strip():
                    continue
                prompt = f"Answer the question based only on the given context.\nContext: {ctx}\nQuestion: {q}"
                f.write(json.dumps({"id": f"rb_{cfg}_{c}", "model": "ragbench", "task_type": cfg,
                                    "split": "test", "context": ctx, "prompt": prompt,
                                    "response": resp, "label": 0 if adh else 1}, ensure_ascii=False) + "\n")
                n += 1; c += 1
                if c >= args.per_config:
                    break
            print(f"  {cfg}: {c} examples")
    print(f"wrote {n} RAGBench examples -> {args.out}")


if __name__ == "__main__":
    main()
