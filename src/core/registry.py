"""
Tiny component registry. Projects register builders by name; configs reference
components by string. Keeps the harness decoupled from any one project.

Usage:
    from core.registry import TRAINERS

    class MyTrainer(BaseTrainer):
        ...

    TRAINERS["my_project"] = MyTrainer
"""

from typing import Any, Callable, Dict

MODELS: Dict[str, Callable[..., Any]] = {}
DATASETS: Dict[str, Callable[..., Any]] = {}
LOSSES: Dict[str, Callable[..., Any]] = {}
TRAINERS: Dict[str, Callable[..., Any]] = {}


def register(kind: Dict[str, Callable], name: str):
    def deco(fn: Callable) -> Callable:
        if name in kind:
            raise ValueError(f"{name!r} is already registered in {kind!r}")
        kind[name] = fn
        return fn

    return deco


def build(kind: Dict[str, Callable], name: str, *args, **kwargs):
    if name not in kind:
        raise KeyError(
            f"{name!r} not registered. Available: {sorted(kind.keys())}"
        )
    return kind[name](*args, **kwargs)
