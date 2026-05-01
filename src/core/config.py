"""
Config loading + stage resolution.

Schema:
    project: <str>             # picks projects/<project>/trainer.py via TRAINERS

    stages:
      - name: <str>            # used for checkpoint prefix + run name
        batch_size: ...
        learning_rate: ...
        weight_decay: ...
        warmup_steps: ...
        max_steps: ...
        gradient_accumulation: ...
        val_every: ...
        save_every: ...
        precision: bf16 | fp16 | fp32
        max_grad_norm: 1.0     # optional, default 1.0
        ...                    # plus any project-specific keys
"""

from typing import Any, Dict, List, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_stages(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a normalized list of stage dicts, each containing a "name" key."""
    if "stages" not in config:
        raise ValueError("config has no `stages:` list")
    stages = list(config["stages"])
    for i, s in enumerate(stages):
        s.setdefault("name", f"stage{i + 1}")
    return stages


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
        raise ValueError("config has no stages")
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
