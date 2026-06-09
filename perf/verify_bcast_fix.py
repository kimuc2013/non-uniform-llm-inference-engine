"""Bit-exact output verification + perf measurement across many prompts.

Runs N requests against the localhost server, dumps each request's
text + token IDs to a file. Run twice (once with flag off, once on),
then diff the two output files.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    "What is the capital of France?",
    "Write three rhymes for the word cat.",
    "List the planets in our solar system.",
    "Explain photosynthesis in one paragraph.",
    "What is the square root of 144?",
    "Translate 'Hello, world' to Spanish.",
    "Name five primary colors.",
    "What year did World War II end?",
    "Describe a sunrise in detail.",
    "What is the largest mammal?",
    "Write a haiku about autumn.",
    "List the first ten prime numbers.",
    "Who painted the Mona Lisa?",
    "Explain the Pythagorean theorem.",
    "What is the chemical formula for water?",
    "Name three Shakespeare plays.",
    "What is the speed of light?",
    "Describe the taste of a lemon.",
    "What is the boiling point of water in Celsius?",
    "Write a short poem about the ocean.",
    "Who wrote 'Pride and Prejudice'?",
    "What is the largest desert in the world?",
    "List the days of the week.",
    "What is the smallest country in Europe?",
    "Explain what a noun is.",
    "Name the four seasons.",
    "What is the currency of Japan?",
    "Who discovered penicillin?",
    "What is the longest river in the world?",
    "Explain gravity in simple terms.",
    "List five vegetables.",
    "What is the meaning of 'serendipity'?",
]


def req_one(model: str, base_url: str, idx: int, prompt: str, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
    }
    t = time.perf_counter()
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    dur = time.perf_counter() - t
    r.raise_for_status()
    d = r.json()
    ch = d["choices"][0]
    return {
        "idx": idx,
        "prompt": prompt,
        "text": ch["message"]["content"],
        "finish": ch.get("finish_reason"),
        "n_prompt": d["usage"]["prompt_tokens"],
        "n_out": d["usage"]["completion_tokens"],
        "wall_s": dur,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--base-url", default="http://127.0.0.1:28100")
    args = p.parse_args()

    prompts = PROMPTS
    t0 = time.perf_counter()
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(req_one, args.model, args.base_url, i, p, args.max_tokens): i
            for i, p in enumerate(prompts)
        }
        for f in futs:
            r = f.result()
            results[r["idx"]] = r
    wall = time.perf_counter() - t0

    total_out = sum(r["n_out"] for r in results)
    print(f"wall: {wall:.2f}s  total_out: {total_out}  tok/s: {total_out/wall:.1f}",
          file=sys.stderr)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
