"""Adapt FaithEval-counterfactual (MC-QA with counterfactual context) into (context, response, label)
detection examples: the answerKey choice is FAITHFUL to the (counterfactual) context (label 0); a
different choice is UNFAITHFUL to the given context (label 1). Tests grounding under subtle conflict.
"""
import json, os, argparse, random
import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Salesforce/FaithEval-counterfactual-v1.0")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/faitheval_examples.jsonl"))
    ap.add_argument("--max", type=int, default=2000)
    args = ap.parse_args()
    d = datasets.load_dataset(args.name, split="test")
    random.seed(0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for i, ex in enumerate(d):
            if n >= 2 * args.max:
                break
            ctx = ex.get("context", "") or ""
            q = ex.get("question", "") or ""
            ch = ex.get("choices") or {}
            labels, texts = ch.get("label", []), ch.get("text", [])
            ak = ex.get("answerKey")
            if not ctx.strip() or ak not in labels or len(texts) < 2:
                continue
            aki = labels.index(ak)
            others = [j for j in range(len(texts)) if j != aki]
            faith, unf = texts[aki], texts[random.choice(others)]
            prompt = f"Answer the question based only on the given context.\nContext: {ctx}\nQuestion: {q}"
            for resp, lab in [(faith, 0), (unf, 1)]:
                f.write(json.dumps({"id": f"fe_{i}_{lab}", "model": "faitheval", "task_type": "FaithEval-CF",
                                    "split": "test", "context": ctx, "prompt": prompt,
                                    "response": resp, "label": lab}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} FaithEval examples -> {args.out}")


if __name__ == "__main__":
    main()
