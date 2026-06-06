"""Prepare HaluEval-QA as a cross-dataset test in the RAGTruth example schema.

HaluEval QA rows: {knowledge, question, right_answer, hallucinated_answer}.
Each row -> two examples: the right answer (label 0) and the hallucinated answer (label 1),
both grounded in the same knowledge (context). Used to test whether RAGTruth-trained probes
transfer zero-shot.
"""
import json, os, urllib.request, argparse, random

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/haluqa_examples.jsonl"))
    ap.add_argument("--raw", default=os.path.expanduser("~/cgp/data/qa_data.json"))
    ap.add_argument("--max_rows", type=int, default=4000)
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        print("downloading HaluEval qa_data.json ...")
        urllib.request.urlretrieve(URL, args.raw)
    rows = [json.loads(l) for l in open(args.raw) if l.strip()]
    random.seed(0)
    random.shuffle(rows)
    rows = rows[:args.max_rows]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            know, q = r["knowledge"], r["question"]
            prompt = f"Answer the question based only on the given context.\nContext: {know}\nQuestion: {q}"
            for resp, lab in [(r["right_answer"], 0), (r["hallucinated_answer"], 1)]:
                ex = {"id": f"halu{i}_{lab}", "model": "halueval", "task_type": "QA",
                      "split": "test", "context": know, "prompt": prompt,
                      "response": resp, "label": lab}
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} examples ({len(rows)} rows x2) -> {args.out}")


if __name__ == "__main__":
    main()
