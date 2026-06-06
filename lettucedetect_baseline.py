"""LettuceDetect baseline (published RAGTruth detector, ModernBERT token classifier).
Example-level score = max span confidence (0 if no hallucinated span). Dumps data/scores/lettucedetect.jsonl.
Requires: pip install lettucedetect
"""
import json, os, argparse


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))


def context_question(e):
    """Split RAGTruth prompt into (context_list, question) per LettuceDetect's API."""
    ctx = as_text(e.get("context", ""))
    prompt = as_text(e.get("prompt", ""))
    # QA: source_info dict has passages+question; else use instruction = prompt minus context
    if ctx and ctx in prompt:
        question = prompt.replace(ctx, "").strip() or "Respond to the request."
    else:
        question = prompt[:400] or "Respond to the request."
    return [ctx], question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/scores/lettucedetect.jsonl"))
    args = ap.parse_args()
    from lettucedetect.models.inference import HallucinationDetector
    detector = HallucinationDetector(method="transformer", model_path=args.model)

    te = [r for r in load_jsonl(args.examples) if r["split"] == "test"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = []
    for k, e in enumerate(te):
        ctx, q = context_question(e)
        ans = as_text(e.get("response", ""))
        try:
            spans = detector.predict(context=ctx, question=q, answer=ans, output_format="spans")
            score = max([float(s.get("confidence", 1.0)) for s in spans], default=0.0)
        except Exception:
            score = 0.0
        out.append((e["id"], e["label"], score))
        if (k + 1) % 500 == 0:
            print(f"{k+1}/{len(te)}", flush=True)
    with open(args.out, "w") as f:
        for i, lab, s in out:
            f.write(json.dumps({"id": i, "label": int(lab), "score": float(s)}) + "\n")
    print(f"dumped lettucedetect: {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
