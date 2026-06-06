"""Prepare RAGTruth into clean example records for conflict-gated probing.

Joins response.jsonl + source_info.jsonl on source_id, derives a binary
example-level hallucination label (1 if any annotated span, else 0).
"""
import json, os, argparse
from collections import Counter


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragtruth_dir", default=os.path.expanduser("~/cgp/RAGTruth/dataset"))
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    args = ap.parse_args()

    resp = load_jsonl(os.path.join(args.ragtruth_dir, "response.jsonl"))
    src = load_jsonl(os.path.join(args.ragtruth_dir, "source_info.jsonl"))
    src_by_id = {s["source_id"]: s for s in src}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    examples = []
    for r in resp:
        s = src_by_id.get(r["source_id"])
        if s is None:
            continue
        labels = r.get("labels", [])
        if isinstance(labels, str):
            labels = json.loads(labels)
        label = 1 if (labels and len(labels) > 0) else 0
        examples.append({
            "id": r["id"],
            "source_id": r["source_id"],
            "model": r.get("model", ""),
            "task_type": s.get("task_type", ""),
            "split": r.get("split", "train"),
            "context": s.get("source_info", ""),
            "prompt": s.get("prompt", ""),
            "response": r.get("response", ""),
            "n_spans": len(labels),
            "label": label,
        })

    with open(args.out, "w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_split = Counter(e["split"] for e in examples)
    pos = Counter((e["split"], e["label"]) for e in examples)
    print(f"total examples: {len(examples)}")
    for sp in by_split:
        n = by_split[sp]
        p = pos[(sp, 1)]
        print(f"  split={sp}: n={n} hallucinated={p} ({100*p/max(n,1):.1f}%)")
    print("by task_type:", dict(Counter(e["task_type"] for e in examples)))
    print("by model:", dict(Counter(e["model"] for e in examples)))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
