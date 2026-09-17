#!/usr/bin/env python3
"""Batcher - continuous batching load harness.

Sends real, escalating concurrent traffic to a vLLM serving endpoint and
records per-request wall time, tokens, and derived tok/s at each concurrency
level. Output: a JSON results file (benchmarks/ dated) + stdout tables.

The point: find the concurrency knee where aggregate throughput stops
scaling and per-stream latency degrades. That knee is the serving stack's
honest capacity number.

READ-ONLY w.r.t. infrastructure: chat completions only. No config changes,
no restarts, no serving-knob edits. Same traffic class as devs using VS Code
against the cluster.

Usage:
    VLLM_API_KEY=<key> python batcher.py --levels 1 2 4 --requests-per-level 8 --max-tokens 150

Env:
    VLLM_BASE_URL  default http://192.168.0.182:8000
    VLLM_API_KEY   required bearer token
"""
import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.request

# Mixed realistic workload: varied lengths and task types.
PROMPTS = [
    ("short", "What is 2+2? Answer in one word."),
    ("short", "Name three primary colors."),
    ("medium", "Explain continuous batching in one paragraph to a bank auditor."),
    ("medium", "Summarize the trade-offs between MIG and time-slicing GPUs in two sentences."),
    ("medium", "What does p95 latency mean and why do SREs prefer it to the mean?"),
    ("code", "Write a Python function that checks whether a string is a palindrome, with type hints and a docstring."),
    ("code", "Write a JSON object with keys model, tokens_in, tokens_out, cost_usd for an inference bill of $0.42 at 1M tokens."),
    ("code", "One-liner in bash: find all files over 100MB in /var and print their sizes."),
    ("long", "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry: type hints, docstrings, and a short design rationale."),
    ("long", "Draft a professional email to a client explaining that a scheduled maintenance window will move to Saturday night, and why."),
]

WEIGHTS = {  # probability of each class per request
    "short": 0.25, "medium": 0.35, "code": 0.25, "long": 0.15,
}

import random


def pick_prompt(rng: random.Random):
    r = rng.random()
    acc = 0.0
    for cls, w in WEIGHTS.items():
        acc += w
        if r <= acc:
            pool = [p for c, p in PROMPTS if c == cls]
            return cls, pool[rng.randrange(len(pool))]
    return "medium", PROMPTS[2][1]


def post_chat(base, key, model, prompt, max_tokens, timeout=240):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        wall = time.perf_counter() - t0
        usage = body.get("usage", {})
        return {
            "ok": True,
            "wall_s": wall,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "wall_s": time.perf_counter() - t0}


def run_level(base, key, model, level, n_requests, max_tokens, seed):
    """Fire n_requests at `level` concurrency; return summary dict."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=level) as ex:
        futures = [
            ex.submit(post_chat, base, key, model, pick_prompt(rng)[1], max_tokens)
            for _ in range(n_requests)
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    walls = sorted(r["wall_s"] for r in ok)
    ctok = sum(r["completion_tokens"] for r in ok)
    ptok = sum(r["prompt_tokens"] for r in ok)

    def pct(p):
        if not walls:
            return None
        idx = min(len(walls) - 1, int(p * len(walls)))
        return round(walls[idx], 3)

    return {
        "concurrency": level,
        "requests": n_requests,
        "ok": len(ok),
        "failed": len(failed),
        "wall_s": round(wall, 2),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "aggregate_tok_s": round((ctok + ptok) / wall, 1) if wall else None,
        "completion_tok_s": round(ctok / wall, 1) if wall else None,
        "per_stream_tok_s": round(ctok / wall / level, 1) if wall and level else None,
        "wall_p50_s": pct(0.50),
        "wall_p95_s": pct(0.95),
        "wall_max_s": round(walls[-1], 2) if walls else None,
        "fail_samples": [f.get("error") for f in failed[:3]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4],
                    help="concurrency levels to test in order")
    ap.add_argument("--requests-per-level", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output JSON path (default benchmarks/<ts>.json)")
    args = ap.parse_args()

    base = os.environ.get("VLLM_BASE_URL", "http://192.168.0.182:8000")
    key = os.environ.get("VLLM_API_KEY")
    model = os.environ.get("VLLM_MODEL", "glm-5.3-flash")
    if not key:
        sys.exit("set VLLM_API_KEY (bearer token); never committed to the repo")

    print(f"Batcher: levels {args.levels}, {args.requests_per_level} req/level, "
          f"max_tokens {args.max_tokens} against {base} ({model})\n")

    all_results = []
    for i, level in enumerate(args.levels):
        seed = args.seed + i * 1000
        r = run_level(base, key, model, level, args.requests_per_level, args.max_tokens, seed)
        all_results.append(r)
        print(f"c={level:>2}  ok={r['ok']}/{r['requests']}  "
              f"agg={r['aggregate_tok_s']} tok/s  per-stream={r['per_stream_tok_s']}  "
              f"wall p50={r['wall_p50_s']}s p95={r['wall_p95_s']}s max={r['wall_max_s']}s"
              + (f"  FAILS={len(r['failed'])}" if r["failed"] else ""))
        time.sleep(2)  # settle between levels

    # naive knee detection: first level where aggregate scaling drops below 1.3x
    knee = None
    for prev, cur in zip(all_results, all_results[1:]):
        if prev["aggregate_tok_s"] and cur["aggregate_tok_s"]:
            scaling = cur["aggregate_tok_s"] / prev["aggregate_tok_s"]
            expected = cur["concurrency"] / prev["concurrency"]
            if scaling < expected * 0.65:
                knee = cur["concurrency"]
                break

    doc = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "endpoint": base,
        "model": model,
        "max_tokens": args.max_tokens,
        "requests_per_level": args.requests_per_level,
        "levels": args.levels,
        "results": all_results,
        "knee_concurrency": knee,
        "notes": "wall-clock includes full request lifecycle; aggregate_tok_s includes prompt+completion tokens",
    }
    out = args.out or os.path.join("benchmarks", f"batcher_{doc['timestamp']}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nknee detection: {'c=' + str(knee) if knee else 'no clear knee in tested range'}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
