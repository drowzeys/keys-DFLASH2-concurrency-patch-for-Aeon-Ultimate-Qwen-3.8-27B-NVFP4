# 02 — Mac oMLX GLM-5.3-Flash oQ4: continuous-batched DFlash2

**Stack:** Mac Studio (M3 Ultra, 256 GB), oMLX 0.6.3 (built from source) + MLX,
vendored `glm5_next` runtime.
**Model:** `GLM-5.3-Flash` oQ4-MTP (abliterated), **MoE** (288 experts, NoPE-DSA +
gated-delta linear attention — a hybrid recurrent/sparse architecture).
**Draft:** `GLM-5.3-Flash-DFlash2` (block-diffusion, block 8, taps [5,14,24,33,42]).

This is the sister lane to [01](01-dflash2-concurrency-ceiling.md): the same DFlash2
concurrency ceiling, on a completely different runtime and a *MoE* model, where the
fix had to be **built from scratch** because the engine had no continuous batching at
all.

## Symptom (identical shape to the GB10 lane)

oMLX's `DFlashEngine` **deliberately bypasses the scheduler** — it runs one
`SpeculativeSession` per request on a single MLX executor thread. So the deployed
DFlash2 serve is flat under concurrency:

```
deployed DFlash2 serve (warm, :11501):  c1 39.4 | c2 33.5 | c4 33.5 | c8 32.4 tok/s
```

Single-stream is great (39.4, vs 1.8 tok/s garbage before the correctness fix — see
below), but there is literally no batching: concurrency is flat/declining.

For contrast, the plain batched (non-spec) engine *does* scale:
`c1 27.6 | c4 54.6 | c8 69.1`. So — exactly as on GB10 — **DFlash2 = latency lane,
batched = throughput lane**, and you couldn't have both from the stock engines.

## Prerequisite: the glm5_next-MLX correctness fix

Before any of this, GLM-5.3-Flash produced **garbage @ 1.8 tok/s** on Mac MLX for
weeks ("MLX lane parked, panicked"). Root cause was a **runtime mismatch**, not the
checkpoint: the oQ4 weights were converted for a *vendor* `glm5_next` runtime
(oMLX 0.6.3rc3 + the 975-line vendor overlay with an absorbed NoPE-MLA prefill path),
but stock `mlx_vlm 0.6.17` shipped a *different* 776-line absorbed-MLA implementation
→ silent wrong output. Overlaying the vendor `glm5_next` (+ deepseek_v32 / mla /
gated_delta / switch_layers deps) into the venv gave **correct + uncensored** output.
DFlash2 then loaded via an `mlx_lm` text adapter and a new
`dflash_mlx/engine/target_glm5_next.py` target backend → **31.6 tok/s** single-stream.

## Building continuous-batched spec (the actual concurrency work)

The path, in milestones (standalone decoders in `~/omlx-glm53/zz_*.py`):

1. **Path-1 de-risk — batched verify is nearly free.** A batched target *verify*
   forward (seq=9 over 256 ctx) costs `B1=1.69s, B2=0.99×, B4=1.05×, B8=1.22×` for 8×
   the work. So verify batches essentially for free → the concept is justified.

2. **Batched spec, identical rows** (`zz_batchspec.py`): prefill native B=N, draft
   once (draft is cheap), broadcast to `(B, BLOCK)` verify → **4.06× at B4**, correct.

3. **Heterogeneous rows** (`zz_hetbatch.py` → `zz_perrow.py`): the crux was **per-row
   accept**. Lockstep min-accept collapses (any row rejecting early caps all). Fix =
   **per-row sparse-KV offset rollback** — and two hard-won facts:
   - The **linear (gated-delta) recurrent state does NOT need rollback**: it tolerates
     recurrent pollution from rejected draft tokens (the gate decays them; sparse KV
     dominates). Shared-batched, no per-row work.
   - Only the **sparse KV** needs per-row rollback: a shared KVCache has one offset and
     cannot represent per-row committed lengths; masking fails because the DSA
     indexer's own cache still accumulates masked tokens. → **per-row KVCaches,
     left-padded + causal/pad-masked into the batched verify** (NoPE ⇒ positions don't
     matter, only causal).
   - `draft_context` must be **all** committed features `cf[:, :1+acc]` (3D), not
     last-only — wrong context → off-manifold drafts → recurrent corruption.

4. **Persistence + warmup** (`zz_fast.py`, `zz_paged*.py`): the standalone decoder was
   1900 ms/forward vs the serve's ~40 ms. Two causes, both fixed:
   - **Growing-shape mask recompiles every step.** A validity mask over `mx.arange(Ktot)`
     with `Ktot` growing changes graph shape → MLX recompiles, never warms. Fix =
     **fixed-size paged buffer** (attend a constant `FIX`-length buffer with a
     fixed-shape validity+causal mask; values change, shape doesn't).
   - **Unwired memory.** Without `mx.set_wired_limit`, big allocs spill to swap-backed
     memory → ~1900 ms/forward. `mx.set_wired_limit(48*1024**3)` at process start →
     **1900 ms → 87 ms**. *(Rule: always call `mx.set_wired_limit` at the top of any
     standalone MLX decode script on Mac; the serve is fast because it keeps weights
     wired.)*

5. **Engine integration:** `omlx/engine/batched_dflash.py` (`BatchedDFlashEngine`) —
   a new engine alongside the working `DFlashEngine` (production-brain safety), fusing
   `BatchedEngine`'s async scheduler + admission with the per-row batched-spec step.
   Validated end-to-end: 4 concurrent requests → correct distinct outputs.

## Result and the hard wall

Paged continuous-batched DFlash2, clean fresh-memory sweep
(`FIX=160` buffer + fixed-shape mask + per-row batched spec + `set_wired_limit(48GB)`):

```
c1 31.1 | c2 37.8 (1.22×) | c4 52.6 (1.69×) | c8 70.3 tok/s (2.26×)
```

c8 aggregate = **1.8× the 39.4 single-stream serve** — and **this is the ceiling**.
`c8=200` is **not physically reachable on Mac Studio**:

> verify = **49 ms fixed** + **6.3 ms/token MoE compute**. Eight rows need ≥8 tokens/step
> ⇒ verify floor ~450 ms at c8. Batching amortizes only the 49 ms fixed part. The
> **MoE per-token compute** (not attention, not warmup, not KV) is the hard wall — the
> same per-token cost the serve already pays (~8–12 ms/tok).

The only remaining lever would be **NVFP4 text-only MoE weights** (cut per-token
bytes) — not available on this stack today.

## Contrast with the GB10 lane

| | GB10 / vLLM / Qwen (dense) | Mac / oMLX / GLM-5.3 (MoE) |
|---|---|---|
| Concurrency already in framework? | Yes (vLLM continuous batching) | No (had to build `BatchedDFlashEngine`) |
| Per-token wall | Memory **bandwidth** | MoE **expert compute** |
| Fix | Config: spec-width taper | Code: paged continuous-batched spec |
| Ceiling reached | C8 **93 tok/s** (past dense) | C8 **70 tok/s** (2.26×, hard wall) |

Same finding, two proofs: **spec is a latency lever; concurrency throughput comes from
batching the verify**, and how far that scales is set by whether the per-token wall is
bandwidth (amortizes well) or compute (amortizes only the fixed part).
