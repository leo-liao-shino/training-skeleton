"""
Config loading + stage resolution.

Schema:
    stages:
      - name: <str>            # used for checkpoint prefix + wandb run name
        resolution: [W, H]
        batch_size: ...
        learning_rate: ...
        ...
        validation:            # optional, project-defined keys
          num_samples: <int>
          num_steps: <int>
          num_loss_batches: <int>
          # ...any other project-specific keys

The `validation:` block is opt-in and untyped at the harness level —
`BaseTrainer` exposes it via `self.val_cfg` and `self.val_param(key, default)`,
and each project's validation_step decides which keys it honors. Projects
that ignore `val_cfg` are unaffected by the schema's existence.

Legacy schema (still supported):
    phase1: {...}              # auto-converted to stages[0] with name="phase1"
    phase2: {...}              # auto-converted to stages[1] with name="phase2"
"""

from typing import Any, Dict, List, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_stages(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a normalized list of stage dicts, each containing a "name" key.

    Prefers an explicit `stages: [...]` list. Falls back to legacy `phaseN`
    keys (sorted by N) if no `stages` key is present.
    """
    if "stages" in config:
        stages = list(config["stages"])
        for i, s in enumerate(stages):
            s.setdefault("name", f"stage{i + 1}")
        return stages

    legacy = []
    for k in sorted(k for k in config if k.startswith("phase") and k[5:].isdigit()):
        stage = dict(config[k])
        stage.setdefault("name", k)
        legacy.append(stage)
    return legacy


def select_stage(
    config: Dict[str, Any],
    *,
    index: Optional[int] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick a stage by 1-based index OR by name. Exactly one must be supplied."""
    if (index is None) == (name is None):
        raise ValueError("select_stage: pass exactly one of index or name")
    stages = resolve_stages(config)
    if not stages:
        raise ValueError("config has no stages (and no legacy phaseN keys)")
    if index is not None:
        if not (1 <= index <= len(stages)):
            raise IndexError(
                f"--phase {index} out of range; config has {len(stages)} stage(s)"
            )
        return stages[index - 1]
    for s in stages:
        if s["name"] == name:
            return s
    raise KeyError(f"stage {name!r} not found; available: {[s['name'] for s in stages]}")
