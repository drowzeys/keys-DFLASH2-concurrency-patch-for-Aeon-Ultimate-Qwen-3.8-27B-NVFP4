# 03 — The hermes 17k-prefill slowdown → prefix caching

**Stack:** hermes agent gateway → vLLM aeon endpoint (Qwen3.8-27B-AEON) on GB10.

## Symptom

Every hermes turn took a long time "processing," regardless of the question — even a
trivial one. The serve was healthy and fast; decode wasn't the issue.

## Diagnosis: re-prefilling a big stable prompt every turn

A hermes agent turn ships a large, mostly-**stable** prompt every time: the system
prompt + all tool schemas ≈ **17k tokens** (`in=16843–17877` in the gateway logs). The
serve was launched (per the model runbook) with `--no-enable-prefix-caching`, so it
**re-prefilled the entire 17k prefix from scratch on every turn**.

Measured directly — the same 13.5k-token prompt, three times back-to-back:

```
prefix caching OFF:  run1 6.87s   run2 7.09s   run3 7.13s   (flat — no reuse)
```

~1,900 tok/s prefill × 17k tokens ≈ **7–9 s of dead prefill per turn**, growing as the
conversation grows. That was the entire "it's slow" symptom.

## Fix: `--enable-prefix-caching`

One flag. The stable prefix (system + tool schemas) is cached in the KV pool and reused
across turns; only the new suffix gets prefilled.

```
prefix caching ON:   run1 6.66s (cold)   run2 0.47s   run3 0.59s   → 14× warm
```

Live cache-hit rate settled at **~64%** almost immediately. End-to-end a hermes turn
that had been ~30 s dropped to ~15 s, and the remainder is now decode + the model's
in-band reasoning, **not** re-prefill.

The KV pool stayed at 1.1M tokens at 1M context — a 17k prefix is trivial to cache, so
nothing was traded away. Qwen3.8-27B is **dense**, so there's no memory blow-up on the
cached prefill.

## Takeaways

- For **any** agent gateway with a big stable system/tool prompt, prefix caching is
  mandatory — it's the difference between re-prefilling 17k tokens every turn and not.
- Don't inherit `--no-enable-prefix-caching` from a raw-serving runbook when the client
  is an agent loop.
- A related, larger lever for the *cold* first turn is trimming what the gateway sends
  up front (it mounts ~22 tool schemas / ~37 KB) — disabling unused toolsets shrinks
  the one-time prefill.
