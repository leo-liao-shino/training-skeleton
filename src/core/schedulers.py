"""
Generic LR schedulers used across projects.
"""

import math

import torch


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Cosine annealing with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
