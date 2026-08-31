#!/usr/bin/env bash
# =============================================================================
# One-shot setup for Aeon-Ultimate-Qwen3.8-27B-NVFP4 on a single DGX Spark (GB10).
#
# Does two things, applying every fix from this repo:
#   1. Serves the model with vLLM: DFlash2 spec-width taper (concurrency) +
#      prefix caching (agent-prompt latency).
#   2. Points your hermes gateway's MAIN model AND all subagent/delegation traffic
#      at that serve, sets thinking on, installs the parallel-delegation steering
#      directive, and (optionally) the per-task-model patch.
#
# Idempotent and safe to re-run. Backs up every file before editing.
# Using a different agent harness? See RECIPE.md — the principle is the same:
# point BOTH the primary model endpoint AND the subagent/delegation endpoint at
# this one serve.
#
# Usage:
#   MODEL=/abs/path/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4 \
#   DRAFT=/abs/path/Qwen3.8-27B-DFlash2 \
#   ./setup-aeon.sh
# =============================================================================
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
MODEL=${MODEL:?set MODEL=/abs/path/to/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4}
DRAFT=${DRAFT:?set DRAFT=/abs/path/to/Qwen3.8-27B-DFlash2}
IMAGE=${IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:2026-08-24-v0.27.1-omni}
CONTAINER=${CONTAINER:-aeon-ultimate}
PORT=${PORT:-8000}
SERVED_NAME=${SERVED_NAME:-aeon}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}/v1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}          # taper keeps 1M-KV headroom safe at 8
GMU=${GMU:-0.70}                          # runbook value for the 1M window
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
HERMES_HOME=${HERMES_HOME:-$HOME/.hermes}
HERMES_AGENT=${HERMES_AGENT:-$HERMES_HOME/hermes-agent}
APPLY_PERTASK_PATCH=${APPLY_PERTASK_PATCH:-1}
RESTART_GATEWAY=${RESTART_GATEWAY:-1}
SKIP_SERVE=${SKIP_SERVE:-0}              # set 1 if the serve is already running
TS=$(date +%Y%m%d-%H%M%S)
say(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---- 1. serve ---------------------------------------------------------------
if [ "$SKIP_SERVE" != "1" ]; then
  say "Launching vLLM serve '$CONTAINER' on :$PORT (taper + prefix caching)"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" --network host --gpus all --restart unless-stopped \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -v "$MODEL":/model:ro -v "$DRAFT":/draft:ro \
    --entrypoint vllm "$IMAGE" \
    serve /model \
    --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" \
    --quantization compressed-tensors --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 --kv-cache-dtype fp8 \
    --enable-chunked-prefill --enable-prefix-caching \
    --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3 \
    --attention-backend TRITON_ATTN --trust-remote-code \
    --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,3,7],[4,6,3],[7,1000,1]]}' \
    --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"repetition_penalty":1.05}'
fi

say "Waiting for /health on $BASE_URL (first boot loads weights + captures graphs)…"
HEALTH="${BASE_URL%/v1}/health"
for i in $(seq 1 240); do
  if curl -sf -m 2 "$HEALTH" >/dev/null 2>&1; then echo "  healthy."; break; fi
  docker ps -q -f name="$CONTAINER" | grep -q . || { echo "  container died:"; docker logs --tail 25 "$CONTAINER"; exit 1; }
  sleep 5
done
curl -sf -m 3 "$HEALTH" >/dev/null 2>&1 || { echo "  serve not healthy after wait"; exit 1; }

# ---- 2. point hermes (main model + delegation) at the serve -----------------
CFG="$HERMES_HOME/config.yaml"
if [ -f "$CFG" ]; then
  say "Patching hermes config: main model + delegation + directive + thinking"
  cp "$CFG" "$CFG.bak-pre-aeon-setup-$TS"
  # Prefer ruamel so YAML round-trips with formatting/comments intact.
  python3 -c "import ruamel.yaml" 2>/dev/null || pip install --quiet ruamel.yaml 2>/dev/null || true
  SERVED_NAME="$SERVED_NAME" BASE_URL="$BASE_URL" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  python3 - "$CFG" <<'PY'
import sys, io
path = sys.argv[1]
import os
served = os.environ["SERVED_NAME"]; base = os.environ["BASE_URL"]
ctx = int(os.environ["MAX_MODEL_LEN"])
DIRECTIVE = (
    "TOP PRIORITY RULE - PARALLEL DELEGATION: the instant the user asks for multiple "
    "concurrent/parallel tasks, or lists several independent questions to run at once, "
    "your FIRST and ONLY action is to CALL the delegate_task tool once with a "
    "fully-populated tasks[] array - one object per item, each with a concrete "
    "self-contained goal string. You are FORBIDDEN from writing any sentence like "
    "\"I'll spawn/dispatch/run subagents\" or answering the items yourself; emitting that "
    "text instead of the tool call is a failure. Just emit the delegate_task tool call "
    "with the tasks filled in. ||| "
)
try:
    from ruamel.yaml import YAML
    yaml = YAML(); yaml.preserve_quotes = True; yaml.width = 4096
    with open(path) as f: cfg = yaml.load(f)
    rt = True
except Exception:
    import yaml as pyyaml
    with open(path) as f: cfg = pyyaml.safe_load(f)
    rt = False

def ensure(d, k):
    if k not in d or not isinstance(d.get(k), dict): d[k] = {}
    return d[k]

# --- main model ---
m = ensure(cfg, "model")
m["default"] = served; m["provider"] = "custom"; m["base_url"] = base
m.setdefault("api_key", "EMPTY")
m["max_tokens"] = 8192; m["context_length"] = ctx
eb = ensure(m, "extra_body")
eb["temperature"] = 0.6; eb["top_p"] = 0.95; eb["top_k"] = 20; eb["repetition_penalty"] = 1.05
ctk = ensure(eb, "chat_template_kwargs")
ctk["enable_thinking"] = True          # REQUIRED for tool planning / delegation
ctk.pop("thinking", None)

# --- delegation / subagents -> same serve ---
dg = ensure(cfg, "delegation")
dg["model"] = served; dg["provider"] = "custom"; dg["base_url"] = base

# --- steering directive (idempotent prepend) ---
ag = ensure(cfg, "agent")
hint = ag.get("environment_hint") or ""
if "PARALLEL DELEGATION" not in hint:
    ag["environment_hint"] = DIRECTIVE + hint
# stabilize the prompt so the prefix cache actually hits across turns
ag["environment_probe"] = False

with open(path, "w") as f:
    if rt: yaml.dump(cfg, f)
    else:
        import yaml as pyyaml
        pyyaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=4096)
print("  hermes config patched (ruamel round-trip)" if rt
      else "  hermes config patched (pyyaml - formatting normalized; backup kept)")
PY
else
  say "No hermes config at $CFG - skipping gateway config (see RECIPE.md for other harnesses)"
fi

# ---- 3. per-task-model patch (optional) -------------------------------------
DT="$HERMES_AGENT/tools/delegate_tool.py"
if [ "$APPLY_PERTASK_PATCH" = "1" ] && [ -f "$DT" ]; then
  if grep -q "Per-task model/endpoint override" "$DT"; then
    say "Per-task-model patch already present - skipping"
  else
    say "Applying per-task-model patch to delegate_tool.py"
    cp "$DT" "$DT.bak-pre-pertask-$TS"
    python3 - "$DT" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read(); orig = s
# Edit 1: schema fields after the per-task `context` property
anchor = '"context": {'
if '"model": {' not in s.split('"tasks"',1)[-1][:2000]:
    ctx_block_end = s.find('},', s.find(anchor))
    if anchor in s and ctx_block_end != -1:
        inject = ('},\n'
          '                        "model": {\n'
          '                            "type": "string",\n'
          '                            "description": "Optional. Run THIS subagent on a specific model instead of the default delegation model. Omit to use the configured default.",\n'
          '                        },\n'
          '                        "base_url": {\n'
          '                            "type": "string",\n'
          '                            "description": "Optional. Direct OpenAI-compatible endpoint for this task\'s model (http://host:port/v1).",\n'
          '                        }')
        s = s[:ctx_block_end] + inject + s[ctx_block_end+1:]
# Edit 2: per-task creds in the child-build loop
needle = 'model=creds["model"],'
repl = (
 '# Per-task model/endpoint override (falls back to global delegation cfg).\n'
 '        _t_model = str(t.get("model") or "").strip()\n'
 '        _t_base = str(t.get("base_url") or "").strip()\n'
 '        _t_provider = str(t.get("provider") or "").strip()\n'
 '        if _t_model or _t_base or _t_provider:\n'
 '            _t_cfg = {"model": _t_model or cfg.get("model"), "provider": _t_provider or cfg.get("provider"),\n'
 '                      "base_url": _t_base or cfg.get("base_url"), "api_key": cfg.get("api_key"), "api_mode": cfg.get("api_mode")}\n'
 '            try:\n'
 '                task_creds = _resolve_delegation_credentials(_t_cfg, parent_agent)\n'
 '            except ValueError as exc:\n'
 '                return tool_error(f"Task {i} model override failed: {exc}")\n'
 '        else:\n'
 '            task_creds = creds\n'
 '        _PATCHED_model=task_creds["model"],'
)
if 'Per-task model/endpoint override' not in s and needle in s:
    s = s.replace('model=creds["model"],', repl, 1)
    for key in ("provider","base_url","api_key","api_mode"):
        s = s.replace(f'override_{key}=creds["{key}"]', f'override_{key}=task_creds["{key}"]')
    for key in ("request_overrides","max_output_tokens","command","args"):
        s = s.replace(f'creds.get("{key}")', f'task_creds.get("{key}")')
    s = s.replace('_PATCHED_model=task_creds["model"],', 'model=task_creds["model"],')
open(p,"w").write(s)
import py_compile
try:
    py_compile.compile(p, doraise=True); print("  patch applied + compiles clean")
except Exception as e:
    open(p,"w").write(orig); print(f"  PATCH REVERTED (compile failed: {e}); apply patches/delegate_tool.per-task-model.md by hand")
PY
  fi
fi

# ---- 4. restart gateway + verify --------------------------------------------
if [ "$RESTART_GATEWAY" = "1" ] && [ -f "$CFG" ]; then
  say "Restarting hermes gateway"
  sudo systemctl restart hermes-gateway 2>/dev/null || systemctl --user restart hermes-gateway 2>/dev/null || \
    echo "  (couldn't restart via systemctl - restart your gateway manually)"
  sleep 5
fi

say "Verify"
echo "  serve models: $(curl -s -m 3 "$BASE_URL/models" | python3 -c 'import json,sys;print([x["id"] for x in json.load(sys.stdin)["data"]])' 2>/dev/null || echo '?')"
echo "  main model    -> $SERVED_NAME @ $BASE_URL"
echo "  delegation    -> $SERVED_NAME @ $BASE_URL  (all subagents land here)"
echo
echo "Done. Fire a multi-part request to confirm parallel subagents fan out onto the serve."
