"""
ExampleTrainer — a tiny project that exercises the framework end-to-end with
no external data, no heavy model, no GPU required.

Trains a 2-layer MLP on synthetic 28x28 tensors with random labels. The point
is *not* to learn anything — it's to validate the loop, optimizer, scheduler,
checkpoint cadence, validation hook, and config plumbing on any machine.

Use it as a copy-paste template when adding your own project under
src/projects/<name>/.

Run:
    python src/train.py --config configs/example.yaml --stage train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.distributed import is_main_process
from core.registry import TRAINERS
from core.trainer import BaseTrainer


def _build_synthetic_dataset(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return TensorDataset(x, y)


class _MLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


class ExampleTrainer(BaseTrainer):
    """MLP classifier with cross-entropy loss."""

    def __init__(self, config, phase_cfg, device, rank, world_size, *, stage_name):
        super().__init__(
            config, phase_cfg, device, rank, world_size, stage_name=stage_name,
        )
        self.dataset_cfg = config.get("dataset", {})

    def init_logging(self):
        if is_main_process():
            print(f"[example] stage={self.stage_name} bs={self.phase_cfg['batch_size']} "
                  f"max_steps={self.phase_cfg['max_steps']}")

    def build_model(self):
        return _MLP(hidden=self.config["model"].get("hidden", 128)).to(self.device)

    def build_data(self):
        train_n = self.dataset_cfg.get("train_n", 1024)
        val_n = self.dataset_cfg.get("val_n", 256)
        seed = self.dataset_cfg.get("seed", 0)
        train_ds = _build_synthetic_dataset(train_n, seed)
        val_ds = _build_synthetic_dataset(val_n, seed + 1)
        return (
            DataLoader(train_ds, batch_size=self.phase_cfg["batch_size"], shuffle=True),
            DataLoader(val_ds, batch_size=self.phase_cfg["batch_size"], shuffle=False),
        )

    def forward_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        with torch.no_grad():
            acc = (logits.argmax(-1) == y).float().mean()
        return loss, {"loss_total": float(loss.detach()), "acc": float(acc)}

    def validation_step(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device); y = y.to(self.device)
                pred = self.model(x).argmax(-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        self.model.train()
        if is_main_process():
            print(f"  [val] acc={correct / max(total, 1):.3f} ({correct}/{total})")


TRAINERS["example"] = ExampleTrainer
