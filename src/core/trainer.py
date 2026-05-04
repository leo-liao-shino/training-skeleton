"""
BaseTrainer: generic training loop scaffolding (DDP wrap, AMP, grad accum,
checkpoint cadence, validation cadence, logging cadence). Project-specific
behavior lives in subclass hooks: build_model, build_data, forward_step,
validation_step, log_samples, init_logging, finalize_logging.

The base class is intentionally framework-agnostic: it knows nothing about
diffusion, latents, masks, or text conditioning.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.checkpoint import load as load_checkpoint, save as save_checkpoint, unwrap
from core.distributed import barrier_safe, is_main_process
from core.schedulers import cosine_with_warmup


class BaseTrainer:
    """Generic single-stage training loop.

    Subclasses must implement: build_model, build_data, forward_step.
    Subclasses may override: build_optimizer, validation_step, log_samples,
    init_logging, finalize_logging, on_step_log.

    Per-stage validation knobs: an optional `validation:` block in each stage
    config is exposed to subclasses as `self.val_cfg` (raw dict) and
    `self.val_param(key, default)` (accessor). The base does not validate or
    interpret keys — each project picks what it reads (e.g. multiview reads
    num_samples/num_steps/num_loss_batches). Projects that ignore `val_cfg`
    are unaffected. See `core/config.py` for the documented schema.
    """

    def __init__(
        self,
        config: dict,
        phase_cfg: dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        *,
        stage_name: str = "stage",
        checkpoint_dir: str = "checkpoints",
        checkpoint_prefix: Optional[str] = None,
    ) -> None:
        self.config = config
        self.phase_cfg = phase_cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.stage_name = stage_name
        self.checkpoint_dir = checkpoint_dir
        # Default checkpoint_prefix to stage_name so phase1_stepN.pt etc. naming
        # falls out automatically; subclasses can still override explicitly.
        self.checkpoint_prefix = checkpoint_prefix or stage_name

        self.step = 0
        self.running_loss = 0.0
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler: Optional[torch.amp.GradScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        precision = phase_cfg.get("precision", "bf16")
        if precision == "fp32":
            self.autocast_dtype = None  # skip autocast entirely
        elif precision == "bf16":
            self.autocast_dtype = torch.bfloat16
        elif precision == "fp16":
            self.autocast_dtype = torch.float16
        else:
            raise ValueError(f"unknown precision: {precision!r}")
        self.use_scaler = precision == "fp16"

        # Per-stage validation knobs. Untyped — see class docstring.
        self.val_cfg: Dict[str, Any] = dict(phase_cfg.get("validation") or {})

    def val_param(self, key: str, default: Any = None) -> Any:
        """Convenience accessor for self.val_cfg.get(key, default)."""
        return self.val_cfg.get(key, default)

    # ----------------------------------------------------------------- hooks

    def build_model(self) -> torch.nn.Module:
        """Construct and return the model (placed on self.device)."""
        raise NotImplementedError

    def build_data(self) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader)."""
        raise NotImplementedError

    def build_optimizer(self) -> Tuple[torch.optim.Optimizer, Any]:
        """Default: AdamW + cosine warmup over trainable params."""
        params = [p for p in unwrap(self.model).parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=self.phase_cfg["learning_rate"],
            weight_decay=self.phase_cfg["weight_decay"],
            betas=(0.9, 0.999),
        )
        sch = cosine_with_warmup(
            opt, self.phase_cfg["warmup_steps"], self.phase_cfg["max_steps"]
        )
        return opt, sch

    def forward_step(self, batch: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run forward + return (scalar loss tensor, scalar log dict).

        Called inside the autocast context. The base loop divides the loss
        by grad_accum and runs backward, so return the *unnormalized* loss.
        """
        raise NotImplementedError

    def validation_step(self) -> None:
        """Run validation. Called on rank 0 only at val_every cadence."""
        return None

    def init_logging(self) -> None:
        """Rank-0-only logging setup (e.g., wandb.init). Default: noop."""

    def finalize_logging(self) -> None:
        """Rank-0-only logging teardown (e.g., wandb.finish). Default: noop."""

    def on_step_log(self, log_dict: Dict[str, float]) -> None:
        """Hook for per-step rank-0 logging. Default: noop."""

    def on_window_log(self, avg_loss: float, log_dict: Dict[str, float]) -> None:
        """Hook for windowed (every-100-step) rank-0 logging. Default: noop."""

    # --------------------------------------------------------------- helpers

    def _wrap_ddp(self) -> None:
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )

    def _maybe_resume(self, resume_path: Optional[str]) -> None:
        if resume_path and os.path.exists(resume_path):
            if is_main_process():
                print(f"Resuming from {resume_path}")
            self.step = load_checkpoint(
                self.model, self.optimizer, self.scheduler, resume_path, self.device
            )

    def _checkpoint_path(self, tag: str) -> str:
        return f"{self.checkpoint_dir}/{self.checkpoint_prefix}_{tag}.pt"

    def _save_checkpoint(self, tag: str) -> None:
        path = self._checkpoint_path(tag)
        save_checkpoint(self.model, self.optimizer, self.scheduler, self.step, path)
        print(f"  Checkpoint saved: {path}")

    # ----------------------------------------------------------------- main

    def fit(self, resume_path: Optional[str] = None) -> None:
        if is_main_process():
            self.init_logging()

        self.model = self.build_model()
        self.model.train()
        self._wrap_ddp()

        self.train_loader, self.val_loader = self.build_data()
        self.optimizer, self.scheduler = self.build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None
        self._maybe_resume(resume_path)

        grad_accum = self.phase_cfg["gradient_accumulation"]
        max_steps = self.phase_cfg["max_steps"]
        max_grad_norm = self.phase_cfg.get("max_grad_norm", 1.0)
        val_every = self.phase_cfg["val_every"]
        save_every = self.phase_cfg["save_every"]

        trainable = [p for p in unwrap(self.model).parameters() if p.requires_grad]

        if is_main_process():
            eff_bsz = self.phase_cfg["batch_size"] * grad_accum * self.world_size
            print(f"Starting training from step {self.step}")
            print(
                f"  Per-rank batch size: {self.phase_cfg['batch_size']} x "
                f"{grad_accum} accum x {self.world_size} ranks = {eff_bsz} effective"
            )
            print(f"  Max steps: {max_steps}")

        pbar = tqdm(
            initial=self.step,
            total=max_steps,
            desc="Training",
            unit="step",
            disable=not is_main_process(),
        )
        self.optimizer.zero_grad()

        while self.step < max_steps:
            for batch in self.train_loader:
                if self.step >= max_steps:
                    break

                amp_ctx = (
                    torch.amp.autocast("cuda", dtype=self.autocast_dtype)
                    if self.autocast_dtype is not None
                    else nullcontext()
                )
                with amp_ctx:
                    loss, log_dict = self.forward_step(batch)
                    loss = loss / grad_accum

                sync_now = (self.step + 1) % grad_accum == 0
                sync_ctx = (
                    self.model.no_sync()
                    if (self.world_size > 1 and not sync_now)
                    else nullcontext()
                )
                with sync_ctx:
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                self.running_loss += log_dict.get("loss_total", float(loss.detach()))

                if sync_now:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, max_norm=max_grad_norm
                    )
                    log_dict["grad_norm"] = float(grad_norm)
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.step += 1
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{log_dict.get('loss_total', 0):.2f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

                # Per-step lightweight log
                if is_main_process():
                    self.on_step_log(log_dict)

                # Windowed log every 100 steps
                if self.step % 100 == 0:
                    avg = self.running_loss / 100
                    if is_main_process():
                        lr = self.scheduler.get_last_lr()[0]
                        tqdm.write(f"  step={self.step}, loss={avg:.4f}, lr={lr:.2e}")
                        self.on_window_log(avg, log_dict)
                    self.running_loss = 0.0

                # Validation
                if self.step % val_every == 0:
                    if is_main_process():
                        self.validation_step()
                    barrier_safe()

                # Checkpoint
                if self.step % save_every == 0:
                    if is_main_process():
                        self._save_checkpoint(f"step{self.step}")
                    barrier_safe()

        pbar.close()

        if is_main_process():
            self._save_checkpoint("final")
            print("Training complete.")
            self.finalize_logging()
        barrier_safe()
