# 04 — The async workaround: parallel subagents that spend the concurrency headroom

**Stack:** hermes agent gateway → vLLM aeon endpoint (taper-scheduled, `seqs=8`).

## Why an async workaround at all

Single-stream decode on one GB10 is **bandwidth-capped** (~22 tok/s for this 27B) no
matter how good the drafter is — see [01](01-dflash2-concurrency-ceiling.md). The
concurrency fix (spec-width taper) buys real *aggregate* headroom (C8 ≈ 93 tok/s), but
a single serial agent turn can never use it: it only ever issues one request at a time.

The way to **spend** that headroom is to turn one user request into **N parallel
background subagents** that all hit the taper-scheduled endpoint at once. That's the
`delegate_task` fan-out. Getting it reliable took three distinct fixes plus one new
feature.

## Problem 1 — the orchestrator narrated instead of delegating

The model would answer `"I'll spin up three parallel subagents…"` as plain text and
stop — no tool call, no subagents, serve idle.

- **Not a capability or parser bug.** Given the `delegate_task` tool directly, the model
  emits a perfect call (`finish_reason=tool_calls`, 3 tasks parsed) 4/4. `delegate_task`
  is a **core (never-deferred) tool**, so it was always visible.
- **It's steering under load.** In the full ~17k-token prompt with 53 tools, the
  gateway's own guidance ("answer in one shot," "greetings get one plain sentence")
  pushed the model to narrate. There was zero guidance to delegate.
- **Fix:** a top-priority delegation directive prepended to `agent.environment_hint`:
  *"the instant the user asks for parallel tasks your FIRST action is to CALL
  delegate_task with a populated tasks[]; you are forbidden to narrate 'I'll spawn
  subagents' or answer them yourself."*

Also required: `enable_thinking: true`. With thinking off the model was fast but didn't
*plan* tool use (it also stopped web-searching and answered from stale training data).
Thinking is what makes it reason → decide to delegate / call tools.

## Problem 2 — the subagents hit a dead backend

Once delegation fired, all three subagents failed: `APIConnectionError after 3 retries`.
They were routed (by the `delegation:` config block, read by
`delegate_tool._resolve_delegation_credentials` → `delegation.model/provider/base_url`)
to `deepseek-v4-flash-dspark @ http://10.100.10.1:8888/v1` — a backend that **wasn't
running**.

- **Fix:** repoint the `delegation:` block to the live serve
  (`model: aeon`, `base_url: http://127.0.0.1:8000/v1`).

**Verified end-to-end:** batch `deleg_662d1e37` ran all three subagents (all
`model=aeon`) concurrently — the serve held `running=3` for ~27 s, then drained
3→2→1→0 as each finished, each returning a real 4000+ char answer, **zero errors**.
That is the concurrency fix and the async fan-out working together: three parallel
subagents landing on one GB10, carried by the taper.

## Problem 3 / the feature — per-task model selection

A fan-out is most useful when subagents can run on **different** models (a fast model
for simple items, a strong model for hard ones; or spreading across fleet endpoints).
Stock hermes resolves **one** delegation model for the whole batch.

**Patch** (`patches/delegate_tool.per-task-model.md`): add optional `model` (and
`base_url`) to each task, and resolve credentials **per task** at dispatch — missing
fields fall back to the global `delegation:` config, so a bare `model` reuses the
default endpoint.

Two edits to `tools/delegate_tool.py`:

1. **Schema** — add `model` and `base_url` to `DELEGATE_TASK_SCHEMA.parameters.tasks.items.properties`.
   (`model` is not in `_MODEL_HIDDEN_TASK_FIELDS`, so it survives the model-facing field
   strip.)

2. **Dispatch** — in the child-build loop, when a task carries `model`/`base_url`/`provider`,
   build a per-task cfg (falling back to the batch `cfg = _load_config()`) and call
   `_resolve_delegation_credentials` for just that task; otherwise reuse the batch-level
   `creds`. Then build the child from `task_creds` instead of the global `creds`.

Unit-verified — distinct models resolve to distinct endpoints:

```
task override glm-5.2 @ .2:8000 -> {'model': 'glm-5.2', 'base_url': 'http://10.100.10.2:8000/v1'}
task override aeon    @ default -> {'model': 'aeon',    'base_url': 'http://127.0.0.1:8000/v1'}
```

Backwards-compatible: tasks with no `model` behave exactly as before (reuse batch
`creds`); the schema fields are optional.

### Usage

```json
{"tasks": [
  {"goal": "quick factual lookup",            "model": "aeon"},
  {"goal": "hard multi-step reasoning",        "model": "glm-5.2", "base_url": "http://10.100.10.2:8000/v1"},
  {"goal": "default — inherits delegation cfg"}
]}
```

## The end-to-end picture

```
user request (multi-part)
        │  (steering directive) → orchestrator CALLS delegate_task
        ▼
delegate_task fan-out ──▶ 3 background subagents (per-task model)
        │                        │  all hit :8000 at once
        ▼                        ▼
   returns handles      vLLM taper schedule: batch=3 → n=3 spec
   (async, non-blocking)        │  serve holds running=3
        ▼                        ▼
 results re-enter the        each subagent decodes in parallel,
 conversation when done      amortized by batching (the fix)
```

The concurrency fix made three parallel streams cheap; the async delegation is what
turns a single user turn into three of them.

## Gotchas logged along the way

- The `-z` one-shot CLI is a **bad harness** for this: top-level delegations run in the
  **background** and get orphaned when the one-shot exits — you'll see `run=1` and empty
  output even when delegation fired correctly. Test in a **persistent session**
  (Telegram / interactive TUI), where subagents actually complete.
- Config edits need `sudo systemctl restart hermes-gateway`; the gateway caches config
  at startup.
- Errors you see *after* a backend fix may be **old pre-fix batches** failing out in the
  TUI — check batch IDs against the fix time.
