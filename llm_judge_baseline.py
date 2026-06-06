"""LLM-as-judge faithfulness baseline via OpenRouter (OpenAI-compatible).

For each RAGTruth test example, asks a judge model to rate hallucination risk 0-100 given
(context, question, response). Dumps data/scores/<tag>.jsonl. Concurrency + retries.
Reads OPENROUTER_API_KEY from env or ~/cgp/.env.
"""
import json, os, argparse, time, re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))


def get_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    envp = os.path.expanduser("~/cgp/.env")
    if os.path.exists(envp):
        for l in open(envp):
            if l.startswith("OPENROUTER_API_KEY="):
                return l.split("=", 1)[1].strip()
    raise SystemExit("no OPENROUTER_API_KEY")


PROMPT = ("You evaluate whether an AI response is fully supported by the given context.\n"
          "Context:\n{ctx}\n\nQuestion/Task:\n{q}\n\nResponse:\n{resp}\n\n"
          "Rate the HALLUCINATION risk of the Response from 0 (every claim supported by the context) "
          "to 100 (clearly contains unsupported/contradicted content). Output ONLY the integer.")


def call(key, model, ctx, q, resp, max_ctx=4000, retries=4):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(ctx=ctx[:max_ctx], q=q[:1000], resp=resp[:2000])}],
        "temperature": 0, "max_tokens": 6,
    }).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            txt = out["choices"][0]["message"]["content"]
            m = re.search(r"\d+", txt)
            return (min(100, max(0, int(m.group()))) / 100.0) if m else 0.5
        except Exception:
            time.sleep(2 * (a + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--tag", default="llmjudge_gpt4omini")
    ap.add_argument("--out_dir", default=os.path.expanduser("~/cgp/data/scores"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = get_key()
    te = [r for r in load_jsonl(args.examples) if r["split"] == "test"]
    if args.limit:
        te = te[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    t0 = time.time()
    def work(r):
        s = call(key, args.model, as_text(r["context"]), as_text(r["prompt"]), as_text(r["response"]))
        return r["id"], r["label"], s
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, r) for r in te]
        for f in as_completed(futs):
            i, lab, s = f.result()
            results[i] = (lab, s)
            done += 1
            if done % 300 == 0:
                print(f"{done}/{len(te)} ({time.time()-t0:.0f}s)", flush=True)
    miss = sum(1 for _, (_, s) in results.items() if s is None)
    out = os.path.join(args.out_dir, args.tag + ".jsonl")
    with open(out, "w") as f:
        for i, (lab, s) in results.items():
            f.write(json.dumps({"id": i, "label": int(lab), "score": float(0.5 if s is None else s)}) + "\n")
    print(f"dumped {args.tag}: {len(results)} ({miss} failed) -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
