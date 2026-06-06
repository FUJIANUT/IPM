"""RAGAS-style faithfulness baseline via OpenRouter. For each example an LLM decomposes the response into
atomic claims and marks each SUPPORTED/UNSUPPORTED by the context; faithfulness = supported/total,
hallucination score = 1 - faithfulness. Dumps scores/ragas.jsonl.
"""
import json, os, argparse, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def as_text(x):
    return x if isinstance(x, str) else ("" if x is None else json.dumps(x, ensure_ascii=False))


def get_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    p = os.path.expanduser("~/cgp/.env")
    if os.path.exists(p):
        for l in open(p):
            if l.startswith("OPENROUTER_API_KEY="):
                return l.split("=", 1)[1].strip()
    raise SystemExit("no OPENROUTER_API_KEY")


PROMPT = ("Decompose the Response into atomic factual claims, then judge each claim as SUPPORTED (its "
          "information is present in or directly inferable from the Context) or UNSUPPORTED.\n\n"
          "Context:\n{ctx}\n\nQuestion:\n{q}\n\nResponse:\n{resp}\n\n"
          "Output ONLY two integers as <num_supported>/<num_total_claims> (e.g. 3/5).")


def call(key, model, ctx, q, resp, retries=4):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 12,
                       "messages": [{"role": "user", "content": PROMPT.format(ctx=ctx[:4000], q=q[:800], resp=resp[:2000])}]}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            m = re.search(r"(\d+)\s*/\s*(\d+)", txt)
            if not m:
                return 0.5
            sup, tot = int(m.group(1)), int(m.group(2))
            faith = sup / tot if tot > 0 else 1.0
            return max(0.0, min(1.0, 1.0 - faith))    # higher = less faithful = more hallucinated
        except Exception:
            time.sleep(2 * (a + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=os.path.expanduser("~/cgp/data/ragtruth_examples.jsonl"))
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--out", default=os.path.expanduser("~/cgp/data/scores/ragas.jsonl"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    key = get_key()
    te = [r for r in load_jsonl(args.examples) if r["split"] == "test"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = {}
    t0 = time.time(); done = 0
    def work(r):
        return r["id"], r["label"], call(key, args.model, as_text(r["context"]), as_text(r["prompt"]), as_text(r["response"]))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, r) for r in te]):
            i, lab, s = f.result(); res[i] = (lab, s); done += 1
            if done % 300 == 0:
                print(f"{done}/{len(te)} ({time.time()-t0:.0f}s)", flush=True)
    miss = sum(1 for _, (_, s) in res.items() if s is None)
    with open(args.out, "w") as f:
        for i, (lab, s) in res.items():
            f.write(json.dumps({"id": i, "label": int(lab), "score": float(0.5 if s is None else s)}) + "\n")
    print(f"dumped ragas: {len(res)} ({miss} failed) -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
