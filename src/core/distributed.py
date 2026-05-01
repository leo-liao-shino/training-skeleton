"""
DDP helpers. Self-spawn launcher pattern: train/inference scripts call
`mp.spawn(_worker, nprocs=N, args=(N, ...))`, each worker calls `setup(rank, N)`
on entry and `cleanup()` on exit.
"""

import os

import torch
import torch.distributed as dist


def setup(rank: int, world_size: int, port: int = 12355) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
    )


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier_safe() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
