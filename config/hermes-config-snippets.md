# Hermes config changes (sanitized)

Relevant `~/.hermes/config.yaml` sections. Secrets (tokens, API keys) are redacted —
substitute your own. Restart after edits: `sudo systemctl restart hermes-gateway`.

## 1. Point the agent at the taper-scheduled serve, thinking ON

```yaml
model:
  default: aeon
  provider: custom
  base_url: http://127.0.0.1:8000/v1
  api_key: EMPTY
  max_tokens: 8192            # room for reasoning + tool args (per-task fan-out safe)
  context_length: 1048576
  extra_body:
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    repetition_penalty: 1.05
    chat_template_kwargs:
      enable_thinking: true    # REQUIRED for tool planning / web search / delegation
```

`enable_thinking: false` makes the model fast but non-planning — it answers from stale
memory and won't call tools. Keep it **true** for agent use.

## 2. Delegation → a LIVE backend (was the dead .1:8888 endpoint)

```yaml
delegation:
  model: aeon
  provider: custom
  base_url: http://127.0.0.1:8000/v1
  # per-task `model`/`base_url` overrides these (see patches/delegate_tool.per-task-model.md)
```

## 3. The parallel-delegation steering directive (prepended to environment_hint)

The model must be told to *call* `delegate_task`, not narrate. Front-load this in
`agent.environment_hint` so it survives the ~17k-token prompt:

```
TOP PRIORITY RULE — PARALLEL DELEGATION: the instant the user asks for multiple
concurrent/parallel tasks, or lists several independent questions to run at once,
your FIRST and ONLY action is to CALL the delegate_task tool once with a
fully-populated tasks[] array — one object per item, each with a concrete
self-contained goal string. You are FORBIDDEN from writing any sentence like
"I'll spawn/dispatch/run subagents" or answering the items yourself; emitting that
text instead of the tool call is a failure. Just emit the delegate_task tool call
with the tasks filled in.
```

## 4. Serve-side: prefix caching ON

In the vLLM launcher (see `launchers/`), use `--enable-prefix-caching` (NOT
`--no-enable-prefix-caching`). 14× warm-TTFT win for the stable ~17k tool+system prefix.

## Notes

- `tool_search.enabled` can stay `auto` — `delegate_task` is a **core** (never-deferred)
  tool, so tiering was never the reason it wasn't called; steering was.
- The MoA `aggregator` block and other providers may still reference offline fleet
  endpoints — unrelated to the delegation path.
