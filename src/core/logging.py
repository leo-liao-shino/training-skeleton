"""
Wandb initialization helper. Project trainers call `init_wandb(...)` from their
`init_logging` hook so the YAML config — not Python — controls project, entity,
run name, tags, etc.

Schema:
    # Top-level config (shared across all stages)
    wandb:
      enabled: true|false
      project: <str>
      entity:  <str|null>
      group:   <str|null>
      tags:    [<str>, ...]
      mode:    online|offline|disabled
      name:    <str|null>      # rarely set globally; usually leave unset

    # Per-stage override (under each entry of `stages:`)
    stages:
      - name: phase1
        wandb:
          name:  <str|null>     # overrides top-level (and the caller's default_name)
          notes: <str|null>
          tags:  [<str>, ...]   # APPENDED to top-level tags

Merge semantics: name / project / entity / group / notes / mode / enabled —
stage overrides top-level. tags — top-level + stage-level concatenated.
`enabled: false` forces `mode="disabled"` so downstream `wandb.log` calls
become safe no-ops without the trainer needing extra branches.
"""

from typing import Any, Dict, Optional

import wandb


def init_wandb(
    config: Dict[str, Any],
    stage_cfg: Dict[str, Any],
    *,
    default_name: str,
    extra_run_config: Optional[Dict[str, Any]] = None,
):
    """Initialize wandb from config. Returns the Run (real or disabled-mode).

    `default_name` is the caller's auto-generated name, used when neither
    `config["wandb"]["name"]` nor `stage_cfg["wandb"]["name"]` is set.
    `extra_run_config` is merged into the run's logged hyperparameter dict
    (on top of `stage_cfg`).
    """
    top = dict(config.get("wandb") or {})
    stage = dict(stage_cfg.get("wandb") or {})

    enabled = stage.get("enabled", top.get("enabled", True))
    if not enabled:
        mode = "disabled"
    else:
        mode = stage.get("mode") or top.get("mode") or "online"

    tags = list(top.get("tags") or []) + list(stage.get("tags") or [])

    run_config = {**stage_cfg, **(extra_run_config or {})}

    return wandb.init(
        project=stage.get("project") or top.get("project"),
        entity=stage.get("entity") or top.get("entity"),
        name=stage.get("name") or top.get("name") or default_name,
        group=stage.get("group") or top.get("group"),
        notes=stage.get("notes") or top.get("notes"),
        tags=tags or None,
        mode=mode,
        config=run_config,
        resume="allow",
    )
