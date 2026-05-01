"""
Generic checkpoint save/load. Saves the unwrapped (non-DDP) state_dict so
keys load cleanly into single-GPU inference.
"""

from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap(module):
    """Return underlying module from DDP wrapper, else passthrough."""
    return module.module if isinstance(module, DDP) else module


def save(model, optimizer, scheduler, step: int, path: str) -> None:
    """Write a training checkpoint to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


def load(model, optimizer, scheduler, path: str, map_location) -> int:
    """Restore model/optimizer/scheduler from `path`. Returns the step."""
    ckpt = torch.load(path, map_location=map_location)
    unwrap(model).load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("step", 0)
