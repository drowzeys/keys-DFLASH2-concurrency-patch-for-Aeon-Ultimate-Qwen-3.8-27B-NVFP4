# keys-DFLASH2-concurrency-patch

Field notes and patches for making **DFlash2 speculative decoding scale under
concurrency** on a single **NVIDIA DGX Spark (GB10, sm_121a)** running vLLM, and for
exploiting that headroom end-to-end through an agent gateway.

Model: **Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4** (dense), draft
**z-lab/Qwen3.8-27B-DFlash2**. All numbers measured 2026-08-29.

## Quick start

One-shot: serve the model with every fix and point your agent gateway's main model **and
all subagents** at it — see **[RECIPE.md](RECIPE.md)**.

```bash
MODEL=/abs/path/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4 \
DRAFT=/abs/path/Qwen3.8-27B-DFlash2 \
./setup-aeon.sh
```

## The arc

1. **The ceiling.** DFlash2 (a block-diffusion speculative drafter) gives a big
   single-stream win but its *aggregate* throughput flatlines at ~2 concurrent
   streams. Adding sequences past that just queues — time-to-first-token explodes
   while tokens/sec stays put.
   → [docs/01-dflash2-concurrency-ceiling.md](docs/01-dflash2-concurrency-ceiling.md)

2. **Why, and the fix.** It is **spec-width inflation**, not a batching bug. With
   `n` draft tokens, every verify step processes `C×(1+n)` candidate positions;
   for a bandwidth-bound (dense) model that inflation cancels the concurrency
   amortization. The fix is to **taper the speculative width down as the running batch
   grows**, so speculation carries latency at low load and plain batching carries
   throughput at high load — from one endpoint. A `num_speculative_tokens_per_batch_size`
   schedule lifts C8 aggregate from **41 → 93 tok/s** with C1 latency intact.
   → [docs/01-dflash2-concurrency-ceiling.md](docs/01-dflash2-concurrency-ceiling.md)

3. **Making the gateway fast enough to use it.** A hermes agent turn was re-prefilling
   its full ~17k-token tool+system prompt every turn (prefix caching was off) — ~7–9 s
   of dead prefill before any answer. Enabling prefix caching cut warm TTFT **14×**
   (6.7 s → 0.47 s).
   → [docs/03-hermes-17k-prefill-slowdown.md](docs/03-hermes-17k-prefill-slowdown.md)

4. **The async workaround that takes advantage of the fix.** Single-stream decode on
   one GB10 is bandwidth-capped no matter what. The way to *use* the concurrency
   headroom is to fan a request out into **parallel background subagents** that all hit
   the taper-scheduled endpoint at once. Getting that reliable took three fixes
   (steering, a live backend, and a new **per-task model** capability so each subagent
   can run on a different model).
   → [docs/04-async-subagent-delegation.md](docs/04-async-subagent-delegation.md)

## Layout

| Path | What |
|---|---|
| `setup-aeon.sh` / `RECIPE.md` | **One-shot**: serve + point hermes' model and all subagents at aeon, applying every fix |
| `docs/01-…` | DFlash2 concurrency ceiling + the vLLM spec-width taper (Qwen3.8-27B-AEON case study) |
| `docs/03-…` | Hermes 17k-prefill slowdown → prefix caching |
| `docs/04-…` | Async subagent delegation + the per-task-model patch |
| `patches/` | The `delegate_task` per-task-model patch + the winning taper `--speculative-config` |
| `launchers/` | The live vLLM launchers (interactive taper + concurrency variant) |
| `benchmarks/` | Raw sweep tables |

## One-line takeaways

- Speculative decoding is a **low-concurrency latency** lever, not a throughput one.
- On a **dense** model the concurrency wall is **memory bandwidth**; taper the spec
  width and plain batching amortizes it (GB10: 93 tok/s @ C8).
- Turn prefix caching **on** for any agent gateway with a big stable system prompt.
- To spend concurrency headroom, **fan out into parallel subagents** — and let each pick
  its own model.

*Author: kevin.on.the.rocks — DGX Spark GB10.*
