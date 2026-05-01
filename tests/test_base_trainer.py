"""
Minimal end-to-end smoke test for BaseTrainer.fit(). Uses a toy linear model
and an in-memory random dataset — no real model, no cache, no CUDA required.

This exists to catch regressions in the loop machinery (grad accum, AMP,
checkpoint cadence, validation cadence, logging hooks) when refactoring core/.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.trainer import BaseTrainer  # noqa: E402


class ToyTrainer(BaseTrainer):
    """Minimal trainer: y = w * x, MSE loss, fake validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_calls = 0
        self.step_log_calls = 0
        self.val_calls = 0

    def build_model(self):
        return nn.Linear(4, 4).to(self.device)

    def build_data(self):
        x = torch.randn(64, 4)
        y = torch.randn(64, 4)
        ds = TensorDataset(x, y)
        return (
            DataLoader(ds, batch_size=self.phase_cfg["batch_size"], shuffle=True),
            DataLoader(ds, batch_size=self.phase_cfg["batch_size"], shuffle=False),
        )

    def forward_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        pred = self.model(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss, {"loss_total": float(loss.detach())}

    def validation_step(self):
        self.val_calls += 1

    def on_step_log(self, log_dict):
        self.step_log_calls += 1

    def on_window_log(self, avg_loss, log_dict):
        self.window_calls += 1


def _phase_cfg(max_steps, **overrides):
    cfg = {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 2,
        "max_steps": max_steps,
        "gradient_accumulation": 1,
        "precision": "bf16" if torch.cuda.is_available() else "fp32",
        "val_every": 5,
        "save_every": 10,
        "max_grad_norm": 1.0,
    }
    cfg.update(overrides)
    return cfg


def test_basic_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as ckpt_dir:
        cfg = _phase_cfg(max_steps=10)
        if device.type == "cpu":
            cfg["precision"] = "fp32"
        trainer = ToyTrainer({}, cfg, device, rank=0, world_size=1,
                             checkpoint_dir=ckpt_dir, checkpoint_prefix="toy")
        if device.type == "cpu":
            trainer.use_scaler = False
            trainer.autocast_dtype = torch.float32
        trainer.fit()

        assert trainer.step == 10, f"expected step=10, got {trainer.step}"
        assert trainer.val_calls >= 1, "validation_step never called"
        assert trainer.window_calls == 0, (
            "on_window_log should fire at step%100==0; with max_steps=10 it should not fire"
        )
        files = sorted(os.listdir(ckpt_dir))
        assert any("step10" in f for f in files), f"expected step10 ckpt; got {files}"
        assert any("final" in f for f in files), f"expected final ckpt; got {files}"
        print(f"[OK] basic loop: step={trainer.step}, val_calls={trainer.val_calls}, "
              f"step_log_calls={trainer.step_log_calls}, ckpts={files}")


def test_grad_accum_and_resume():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as ckpt_dir:
        cfg = _phase_cfg(max_steps=8, gradient_accumulation=2)
        if device.type == "cpu":
            cfg["precision"] = "fp32"
        t1 = ToyTrainer({}, cfg, device, rank=0, world_size=1,
                        checkpoint_dir=ckpt_dir, checkpoint_prefix="toy")
        if device.type == "cpu":
            t1.use_scaler = False
            t1.autocast_dtype = torch.float32
        t1.fit()
        assert t1.step == 8
        final_ckpt = next(p for p in os.listdir(ckpt_dir) if "final" in p)

        cfg2 = _phase_cfg(max_steps=12, gradient_accumulation=2)
        if device.type == "cpu":
            cfg2["precision"] = "fp32"
        t2 = ToyTrainer({}, cfg2, device, rank=0, world_size=1,
                        checkpoint_dir=ckpt_dir, checkpoint_prefix="toy")
        if device.type == "cpu":
            t2.use_scaler = False
            t2.autocast_dtype = torch.float32
        t2.fit(resume_path=os.path.join(ckpt_dir, final_ckpt))
        assert t2.step == 12, f"expected step=12 after resume from 8 + 4, got {t2.step}"
        print(f"[OK] grad_accum + resume: t1.step=8 -> t2.step={t2.step}")


if __name__ == "__main__":
    test_basic_loop()
    test_grad_accum_and_resume()
    print("All BaseTrainer smoke tests passed.")
