# Benchmark tables (raw)

All GB10 numbers: single DGX Spark, vLLM 0.27.1+aeon.sm121a.dspark, Qwen3.8-27B-AEON-NVFP4,
256 exact tokens/request (ignore_eos), prose (LRU-cache) workload, 2026-08-29.

## GB10 — spec n=7 (flat) vs dense vs adaptive schedules (aggregate tok/s)

| C | flat spec n=7 | pure dense (no spec) | abrupt cut [[1,2,7],[3,∞,0]] | gentle taper [[1,3,7],[4,6,3],[7,∞,1]] |
|---|---|---|---|---|
| 1 | 22   | 11.5 | 20.8 | 21.3 |
| 2 | 42   | 23.4 | 42.0 | 40.2 |
| 4 | 40   | 45.4 | 33.6 | 69.6 |
| 8 | 41   | 85.7 | 59.1 | 92.9 |

Per-stream decode (flat spec): ~22 tok/s across C1–C8 (flat).
TTFT (flat spec): 250 ms @C1 → 24–38 s @C8 (pure queueing past C2).
TTFT (dense): ~400 ms flat through C8.
Spec acceptance (gentle taper): rises 23% → 65% as batch fills.

Live deployed endpoint re-measure (gentle taper, seqs=8): C1 20.9 / C4 66.7 / C8 90.7.
Direct 3-concurrent proof: running=3 sustained, 3×256-tok in 11.8 s wall.

## Mac — oMLX GLM-5.3-Flash oQ4 (aggregate tok/s)

| C | deployed DFlashEngine (no batching) | batched (no spec) | paged continuous-batched spec |
|---|---|---|---|
| 1 | 39.4 | 27.6 | 31.1 |
| 2 | 33.5 | —    | 37.8 |
| 4 | 33.5 | 54.6 | 52.6 |
| 8 | 32.4 | 69.1 | 70.3 |

Continuous-batched spec: 2.26× (c1→c8). Ceiling is the MoE per-token compute
(verify = 49 ms fixed + 6.3 ms/token); c8=200 not physically reachable.

## Hermes prefix caching (13.5k-token prompt, warm TTFT)

| run | prefix caching OFF | prefix caching ON |
|---|---|---|
| 1 (cold) | 6.87 s | 6.66 s |
| 2 (warm) | 7.09 s | 0.47 s |
| 3 (warm) | 7.13 s | 0.59 s |

14× warm; live cache-hit rate ~64%.
