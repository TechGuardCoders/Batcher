# Batcher

**Continuous batching load test for a self-hosted inference cluster.**

Escalating concurrent traffic against a live vLLM endpoint, measuring where
aggregate throughput stops scaling and per-stream latency degrades. Because
you don't know your serving stack until it breaks under load.

![CI](https://github.com/TechGuardCoders/04-batcher/actions/workflows/ci.yml/badge.svg)

## Why

vLLM's continuous batching is the reason one GPU can serve many users at
once: requests join an in-flight decode batch instead of waiting for the GPU.
But batching has a knee - a concurrency level where aggregate throughput
stops climbing and everyone's latency degrades together. Capacity planning
without finding that knee is guessing.

## What it does

1. Fires escalating concurrency levels (1 -> 2 -> 4 -> ...) at the endpoint
2. Mixed realistic workload: short chat, medium explanation, code generation,
   long-form - weighted like real traffic, not a synthetic hammer
3. Records per-level: aggregate tok/s, per-stream tok/s, wall-clock p50/p95/max
4. Detects the batching knee (first level where scaling drops below 65% of
   perfect linear)
5. Writes dated JSON to `benchmarks/` + a markdown report

## Results (2026-09-16, GLM-5.3-Flash on 2x DGX Spark, TP=2)

| Concurrency | Aggregate tok/s | Per-stream tok/s | Wall p50 |
|---:|---:|---:|---:|
| 1 | 43.8 | 35.6 | 2.9s |
| 2 | 45.2 | 18.7 | 6.2s |
| 4 | **67.9** | 14.2 | 9.7s |

Full findings in `benchmarks/2026-09-16-light-escalation.md`. Headlines:
sub-linear scaling (1.55x at c=4, bandwidth-bound MoE), a non-monotonic knee
at c=2 caused by speculative-decoding overhead, zero failures, and the
production `MAX_SEQS=4` cap confirmed as the concurrency ceiling.

## Run it

```bash
VLLM_API_KEY=<key> python batcher.py --levels 1 2 4 --requests-per-level 8 --max-tokens 150
```

Env: `VLLM_BASE_URL` (default `http://192.168.0.182:8000`), `VLLM_MODEL`
(default `glm-5.3-flash`). The key is never committed. Escalate levels at
your own capacity risk: `--levels 1 2 4 8 12`.

## Architecture

```
batcher.py  --escalating concurrent POSTs-->  vLLM /v1/chat/completions
     |                                              |
     +-- per-request wall time, tokens              +-- untouched serving
     +-- knee detection, JSON + markdown report     stack (read-only side
                                                    effects: real tokens)
```

- Read-only w.r.t. infrastructure: chat completions only, no config changes.
- Results committed under `benchmarks/` - honest evidence, dated, with caveats.

## Resume-grade summary

Load-tested a self-hosted 320B MoE inference cluster with an escalating
concurrency harness; identified the continuous-batching saturation knee
(sub-linear 1.55x aggregate scaling at 4 streams on a bandwidth-bound MoE)
and quantified the latency/throughput trade-off that informs capacity
planning and autoscaler thresholds.
