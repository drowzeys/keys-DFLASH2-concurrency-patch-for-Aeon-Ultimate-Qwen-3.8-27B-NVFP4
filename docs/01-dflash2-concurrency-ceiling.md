# 01 — The DFlash2 concurrency ceiling, and the spec-width taper fix

**Stack:** single DGX Spark (GB10, sm_121a), vLLM `0.27.1+aeon.sm121a.dspark`.
**Model:** `Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4` (dense — 64 layers, hidden 5120,
**no experts**), NVFP4 W4A4, fp8 KV, 1M YaRN context.
**Draft:** `z-lab/Qwen3.8-27B-DFlash2` (block-diffusion drafter), `num_speculative_tokens=7`.

## Symptom

DFlash2 gives a strong single-stream number but aggregate throughput refuses to
scale past ~2 concurrent streams. Measured, 256 exact tokens/request (`ignore_eos`),
prose workload:

| Concurrency | Aggregate tok/s (spec n=7) | Per-stream decode | TTFT |
|---|---|---|---|
| C1 | 22 | 22 | 250 ms |
| C2 | **42** | 22 | 340 ms |
| C4 | 40 | 22 | 11.7 s |
| C8 | 41 | 21 | 24–38 s |

It ceilings at ~42 (= 2× single-stream) and **stays there** while TTFT explodes.
Raising `--max-num-seqs` from 2 → 8 changed *nothing* — so the wall is spec decode
itself, not the sequence cap (the runbook's `max-num-seqs 2` was only there for 1M-KV
headroom).

Output stays byte-clean under load throughout — this is a **throughput** tradeoff,
not a correctness bug. (Contrast the older DSpark batch>1 *corruption* bug, which
was a different problem needing a KV-slot fix.)

## Diagnosis: spec-width inflation, on a bandwidth-bound model

Two facts settle it.

1. **The model is dense.** So per-token decode is **memory-bandwidth bound**
   (~20 GB NVFP4 ÷ 273 GB/s ≈ 13–14 forward passes/s). That is exactly why the
   *dense* (spec-off) curve scales cleanly — batching amortizes one weight-read
   across the whole batch:

   | C | dense (no spec) tok/s |
   |---|---|
   | C1 | 11.5 |
   | C2 | 23.4 |
   | C4 | 45.4 |
   | C8 | **85.7** (TTFT still ~400 ms) |

2. **vLLM's `DFlash2Speculator` already batches** the draft across requests
   (`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` — the draft forward runs
   over all requests at once, and the Triton selector/cache kernels grid over
   `num_reqs`). So the flatline is **not** a missing-batching bug.

The real cause: with `n=7`, each verify step processes `C×(1+7)=C×8` candidate
positions instead of `C×1`. For a model that is only bandwidth-bound at low token
counts, that **8× position inflation** tips the verify into *compute*-bound at high
concurrency, and cancels the amortization → flat 42.

The general principle: batching amortizes only the *fixed* part of the verify; the
per-token compute is the hard wall. Here the per-token cost is the spec-inflated
attention/matmul, so narrowing the spec width at high batch is what unblocks it.

## The fix: taper the speculative width by running batch size

vLLM `SpeculativeConfig` exposes
`num_speculative_tokens_per_batch_size: list[(range_start, range_end, num_spec_tokens)]`
(inclusive batch ranges) — it varies the draft width by the running batch size.

Keep full spec at low batch (where speculation wins on latency) and taper to near-zero
as the batch fills (where plain batching wins on throughput). A **gentle 3-step** taper
avoids thrash:

```json
{"method": "dflash", "model": "/draft", "num_speculative_tokens": 7,
 "num_speculative_tokens_per_batch_size": [[1, 3, 7], [4, 6, 3], [7, 1000, 1]]}
```

Measured (seqs=8, live endpoint):

| C | flat n=7 | pure dense | abrupt cut `[[1,2,7],[3,∞,0]]` | **gentle taper (deployed)** |
|---|---|---|---|---|
| C1 | 22 | 11.5 | 20.8 | **21.3** ✓ latency kept |
| C2 | 42 | 23.4 | 42.0 | **40.2** ✓ latency kept |
| C4 | 40 | 45.4 | 33.6 ✗ thrash | **69.6** |
| C8 | 41 | 85.7 | 59.1 | **92.9** |

The gentle taper is **2.3× the flat-spec ceiling at C8**, monotonic (no thrash),
output byte-clean under 8-way load, and C8 even edges past pure dense because the
`n=1` tail still speculates on top of the batching. Acceptance rises with batch
(23% → 65%) exactly as intended — spec narrows as the batch fills.

### Why *gentle*, not a hard cut

An abrupt one-step cut (`[[1,2,7],[3,1000,0]]`) **thrashes**: at C4 the running batch
oscillates across the single cutover, and the mode-flip overhead makes C4 (33.6)
*worse than both* pure modes. Spreading the transition over three steps gives clean
monotonic scaling.

## Deploying it

- Interactive / single-user (e.g. an agent gateway at C1): plain `n=7`, `seqs=2` is
  ideal — best latency, matches 1M-KV headroom.
- Concurrent serving: the taper + `seqs=8`.
- Both live in `launchers/aeon-ultimate-serve.sh` (taper is the default;
  `MAX_NUM_SEQS` overridable) and `launchers/aeon-serve-concurrent.sh`.

Assert `speculative_config` appears in the boot args ~25 s in — a silently-dropped
spec flag is a classic no-op trap.

## Bottom line

> Speculative decoding is a **low-concurrency latency** optimization. On a dense,
> bandwidth-bound model you get the best of both by tapering the spec width with the
> batch: speculation for latency at C1–C2, plain batching for throughput at C4+.
