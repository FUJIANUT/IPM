"""Prepare HaluEval (QA / Summarization / Dialogue) into the RAGTruth example schema.
Each row -> a right (label 0) and a hallucinated (label 1) example grounded in the same context.
"""
import json, os, urllib.request, argparse, random

BASE = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/{}_data.json"
FIELDS = {
    "qa": ("knowledge", "question", "right_answer", "hallucinated_answer"),
    "summarization": ("document", None, "right_summary", "hallucinated_summary"),
    "dialogue": ("knowledge", "dialogue_history", "right_response", "hallucinated_response"),
}
INSTR = {
    "qa": "Answer the question based only on the given context.",
    "summarization": "Summarize the document.",
    "dialogue": "Respond to the dialogue based only on the given context.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(FIELDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=4000)
    args = ap.parse_args()
    raw = os.path.expanduser(f"~/cgp/data/{args.task}_data.json")
    if not os.path.exists(raw):
        print(f"downloading HaluEval {args.task} ...")
        urllib.request.urlretrieve(BASE.format(args.task), raw)
    rows = [json.loads(l) for l in open(raw) if l.strip()]
    random.seed(0); random.shuffle(rows)
    rows = rows[:args.max_rows]
    cf, qf, rf, hf = FIELDS[args.task]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            ctx = r.get(cf, "")
            q = r.get(qf, "") if qf else ""
            prompt = f"{INSTR[args.task]}\nContext: {ctx}" + (f"\nQuestion: {q}" if q else "")
            for resp, lab in [(r.get(rf, ""), 0), (r.get(hf, ""), 1)]:
                ex = {"id": f"{args.task}{i}_{lab}", "model": "halueval", "task_type": args.task.title(),
                      "split": "test", "context": ctx, "prompt": prompt, "response": resp, "label": lab}
                f.write(json.dumps(ex, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n} {args.task} examples -> {args.out}")


if __name__ == "__main__":
    main()
