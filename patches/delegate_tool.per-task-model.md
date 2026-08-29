# Patch: per-task model selection for `delegate_task`

Target: hermes-agent `tools/delegate_tool.py` (v0.20.6 / 2026.8.27).
Adds an optional `model` (and `base_url`) to each task in a `delegate_task` batch and
resolves delegation credentials **per task**, falling back to the global `delegation:`
config for any field a task omits. Backwards-compatible: tasks without `model` behave
exactly as before.

---

## Edit 1 — schema: add `model` and `base_url` to the per-task items

In `DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]`,
after the `context` property and before `output_schema`:

```python
                        "model": {
                            "type": "string",
                            "description": (
                                "Optional. Run THIS subagent on a specific "
                                "model instead of the default delegation "
                                "model. Give a model name the fleet serves "
                                "(e.g. a fast model for simple tasks, a "
                                "stronger model for hard ones). Omit to use "
                                "the configured default."
                            ),
                        },
                        "base_url": {
                            "type": "string",
                            "description": (
                                "Optional. Direct OpenAI-compatible endpoint "
                                "for this task's model (e.g. "
                                "http://host:port/v1). Only needed when the "
                                "model isn't reachable via the default "
                                "delegation endpoint or a configured provider."
                            ),
                        },
```

`model` is intentionally **not** added to `_MODEL_HIDDEN_TASK_FIELDS`
(`{"acp_command", "acp_args"}`), so it survives `_strip_model_hidden_task_fields`.

---

## Edit 2 — dispatch: resolve credentials per task

In `delegate_task(...)`, inside the child-build loop
(`for i, t in enumerate(task_list):`), immediately **before** the
`child = _build_child_preserving_parent_tools(...)` call, insert the per-task
resolution, and change the child build to use `task_creds` instead of the global
`creds`. `cfg` here is the `cfg = _load_config()` resolved once at the top of
`delegate_task` (the `delegation:` config section).

```python
        # Per-task model/endpoint override. When a task carries its own `model`
        # (or `base_url`/`provider`), resolve credentials just for that task —
        # missing fields fall back to the global delegation config so a bare
        # `model` reuses the default endpoint. Otherwise reuse the batch-level
        # `creds` resolved once above.
        _t_model = str(t.get("model") or "").strip()
        _t_base = str(t.get("base_url") or "").strip()
        _t_provider = str(t.get("provider") or "").strip()
        if _t_model or _t_base or _t_provider:
            _t_cfg = {
                "model": _t_model or cfg.get("model"),
                "provider": _t_provider or cfg.get("provider"),
                "base_url": _t_base or cfg.get("base_url"),
                "api_key": cfg.get("api_key"),
                "api_mode": cfg.get("api_mode"),
            }
            try:
                task_creds = _resolve_delegation_credentials(_t_cfg, parent_agent)
            except ValueError as exc:
                return tool_error(
                    f"Task {i} model override ('{_t_model or _t_base or _t_provider}') "
                    f"failed to resolve: {exc}"
                )
        else:
            task_creds = creds
```

Then in the `_build_child_preserving_parent_tools(...)` call, replace every
`creds[...]` / `creds.get(...)` with the same key on `task_creds`:

```python
                model=task_creds["model"],
                ...
                override_provider=task_creds["provider"],
                override_base_url=task_creds["base_url"],
                override_api_key=task_creds["api_key"],
                override_api_mode=task_creds["api_mode"],
                override_request_overrides=task_creds.get("request_overrides"),
                override_max_tokens=task_creds.get("max_output_tokens"),
                override_acp_command=task_creds.get("command"),
                override_acp_args=task_creds.get("args"),
```

---

## Verification

```python
from tools.delegate_tool import _resolve_delegation_credentials
c1 = _resolve_delegation_credentials(
    {"model":"glm-5.2","provider":"custom","base_url":"http://10.100.10.2:8000/v1",
     "api_key":None,"api_mode":None}, parent)
c2 = _resolve_delegation_credentials(
    {"model":"aeon","provider":"custom","base_url":"http://127.0.0.1:8000/v1",
     "api_key":None,"api_mode":None}, parent)
assert c1["base_url"] != c2["base_url"] and c1["model"] != c2["model"]
```

→ distinct models route to distinct endpoints. `python3 -m py_compile` clean; gateway
restart required to pick up the schema (`sudo systemctl restart hermes-gateway`).
