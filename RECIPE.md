# One-shot recipe — point everything at Aeon-Ultimate-Qwen3.8-27B-NVFP4

Goal: on a single DGX Spark (GB10), serve the model with **all the fixes in this repo**
and make sure your agent harness routes **both** its main model **and every subagent /
delegation call** to that one serve — so nothing silently falls back to a different or
dead endpoint (the failure mode that wastes a whole fan-out).

## TL;DR

```bash
MODEL=/abs/path/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4 \
DRAFT=/abs/path/Qwen3.8-27B-DFlash2 \
./setup-aeon.sh
```

Idempotent, backs up every file it touches. If the serve is already running, add
`SKIP_SERVE=1`. Knobs: `PORT`, `SERVED_NAME`, `MAX_NUM_SEQS`, `GMU`, `MAX_MODEL_LEN`,
`HERMES_HOME`, `APPLY_PERTASK_PATCH=0`, `RESTART_GATEWAY=0`.

## What it applies

**Serve (vLLM):**
- DFlash2 **spec-width taper** `num_speculative_tokens_per_batch_size=[[1,3,7],[4,6,3],[7,1000,1]]`
  — C1 latency + concurrency throughput from one endpoint (C8 ~93 tok/s). See `docs/01`.
- `--enable-prefix-caching` — 14× warm TTFT on the big agent prompt. See `docs/03`.
- `--max-num-seqs 8`, fp8 KV, TRITON_ATTN, `qwen3_coder` tool parser + auto tool choice,
  `qwen3` reasoning parser, GMU 0.70, 1M context.

**Hermes gateway config:**
- `model.default → aeon`, `base_url → http://127.0.0.1:8000/v1`.
- `model.extra_body`: temp 0.6 / top_p 0.95 / top_k 20 / rep 1.05, and
  **`chat_template_kwargs.enable_thinking: true`** — REQUIRED, or the model answers from
  stale memory and won't plan tool use / delegate. See `docs/04`.
- **`delegation.model → aeon`, `base_url → the same serve`** — so every subagent lands on
  aeon, not a different or dead backend.
- The **parallel-delegation steering directive** prepended to `agent.environment_hint`
  (makes the model *call* `delegate_task` instead of narrating "I'll spawn subagents").
- `agent.environment_probe: false` — stabilizes the prompt prefix so prefix caching hits.

**Code patch (optional, `APPLY_PERTASK_PATCH=1`):**
- The `delegate_task` **per-task-model** patch (`patches/delegate_tool.per-task-model.md`)
  so a fan-out can run each subagent on a different model. Applied idempotently; auto-reverts
  if it doesn't compile against your hermes version.

## Verify it worked

```bash
# main model
hermes -z "In one sentence, what are you running on?"
# parallel fan-out — should dispatch 3 subagents onto the serve (watch running=3):
hermes -z "run three concurrent tasks: 1) ... 2) ... 3) ..."
watch -n1 'curl -s http://127.0.0.1:8000/metrics | awk "/num_requests_running/{print \$2}"'
```

Test parallel delegation in a **persistent session** (interactive TUI / chat platform),
not a `-z` one-shot — one-shots orphan the background subagents. See `docs/04` gotchas.

## Using a different agent harness

The script edits hermes' `config.yaml`, but the principle is harness-agnostic. Wherever
your harness stores model routing, set **two** things to the same serve:

1. **Primary model** endpoint → `http://<host>:8000/v1`, model `aeon`.
2. **Subagent / worker / delegation** endpoint → the *same* URL and model.

Then carry over the three behavioral fixes: `enable_thinking: true` (or your harness's
"reasoning on"), a system-prompt directive that tells the model to emit the real
fan-out/tool call rather than describe it, and (if it has one) turn on the serve's prefix
cache. Point #2 is the one people miss — a harness that leaves subagents on a default or
old model will fan out into failures even when the main chat works.
