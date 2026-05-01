"""
Training launcher. Thin entry point — actual trainer lives in
projects.<project>.trainer (selected by the `project:` key in the config).

Usage:
    python src/train.py --config configs/example.yaml --phase 1
    python src/train.py --config configs/example.yaml --stage train
    python src/train.py --config configs/example.yaml --phase 2 --resume checkpoints/stage1_final.pt
"""

import argparse
import importlib

import torch
import torch.multiprocessing as mp

from core.config import load_config, select_stage
from core.distributed import (
    setup as ddp_setup,
    cleanup as ddp_cleanup,
    is_main_process,
)
from core.registry import TRAINERS


def _get_trainer_cls(project_name: str):
    """Import the project's trainer module (which self-registers in TRAINERS),
    then return the trainer class. Raises a clear error if the project module
    or its registration is missing."""
    if project_name not in TRAINERS:
        try:
            importlib.import_module(f"projects.{project_name}.trainer")
        except ImportError as e:
            raise ImportError(
                f"could not import projects.{project_name}.trainer: {e}"
            ) from e
    if project_name not in TRAINERS:
        raise KeyError(
            f"project {project_name!r} did not register a Trainer in "
            f"TRAINERS. Available: {sorted(TRAINERS)}"
        )
    return TRAINERS[project_name]


def train(config, stage_index=None, stage_name=None, resume_path=None,
          max_steps_override=None, rank=0, world_size=1):
    """Pick the project's Trainer class and run fit() on the selected stage."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    stage_cfg = dict(select_stage(config, index=stage_index, name=stage_name))
    if max_steps_override is not None:
        stage_cfg["max_steps"] = max_steps_override
        if is_main_process():
            print(f"[override] max_steps = {max_steps_override}")
    name = stage_cfg["name"]
    project = config["project"]
    if is_main_process():
        print(f"{'=' * 60}")
        print(f"[{project}] stage {name!r}  (world_size={world_size})")
        print(f"{'=' * 60}")

    trainer_cls = _get_trainer_cls(project)
    trainer = trainer_cls(
        config, stage_cfg, device, rank, world_size, stage_name=name,
    )
    trainer.fit(resume_path=resume_path)


def _ddp_worker(rank, world_size, config, stage_index, stage_name,
                resume_path, max_steps_override):
    """Per-rank entry point invoked by mp.spawn."""
    try:
        ddp_setup(rank, world_size)
        train(config, stage_index=stage_index, stage_name=stage_name,
              resume_path=resume_path, max_steps_override=max_steps_override,
              rank=rank, world_size=world_size)
    finally:
        ddp_cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # Stage selection: pass --phase N (1-based index) OR --stage NAME.
    parser.add_argument("--phase", type=int, default=None,
                        help="1-based stage index")
    parser.add_argument("--stage", type=str, default=None,
                        help="Stage name (matches `stages[].name` in the config)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from config (useful for smoke tests)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use (DDP). Default 1 = single-GPU.")
    args = parser.parse_args()

    if (args.phase is None) == (args.stage is None):
        parser.error("pass exactly one of --phase or --stage")

    config = load_config(args.config)
    if "project" not in config:
        parser.error(f"config {args.config!r} must define a top-level `project:` key")
    # Validate stage selection up-front so DDP workers don't all hit the error.
    select_stage(config, index=args.phase, name=args.stage)

    if args.gpus > 1:
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
            available = torch.cuda.device_count() if torch.cuda.is_available() else 0
            raise RuntimeError(
                f"--gpus={args.gpus} requested but only {available} CUDA devices available"
            )
        mp.spawn(
            _ddp_worker,
            nprocs=args.gpus,
            args=(args.gpus, config, args.phase, args.stage, args.resume, args.max_steps),
            join=True,
        )
    else:
        train(config, stage_index=args.phase, stage_name=args.stage,
              resume_path=args.resume, max_steps_override=args.max_steps,
              rank=0, world_size=1)


if __name__ == "__main__":
    main()
