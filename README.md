# Training Skeleton

A minimal, framework-agnostic training harness for PyTorch projects. Provides a
generic training loop (DDP, AMP, grad accumulation, checkpoint/validation
cadence), a config-driven launcher, and a tiny component registry. Bring your
own model, data, and loss.

## Layout

```
skeleton/
├── README.md
├── requirements.txt
├── configs/
│   └── example.yaml             # one config per experiment
├── tests/
│   └── test_base_trainer.py     # BaseTrainer smoke test (CPU, no deps)
└── src/
    ├── train.py                 # CLI entry point (dispatches by config['project'])
    ├── core/                    # generic, project-agnostic
    │   ├── trainer.py           #   BaseTrainer (the loop)
    │   ├── config.py            #   YAML loading + stage selection
    │   ├── registry.py          #   TRAINERS / MODELS / DATASETS / LOSSES
    │   ├── checkpoint.py        #   save/load/unwrap (DDP-safe)
    │   ├── distributed.py       #   setup/cleanup/is_main_process/barrier_safe
    │   └── schedulers.py        #   cosine_with_warmup
    └── projects/
        ├── __init__.py
        └── example/             # template project — copy this
            ├── __init__.py
            └── trainer.py       #   subclasses BaseTrainer, registers in TRAINERS
```

`core/` should not depend on anything in `projects/`. `projects/<name>/` may
import freely from `core/`.

## Quickstart

```bash
pip install -r requirements.txt

# Run the example project (CPU, ~5 seconds, no data download)
python src/train.py --config configs/example.yaml --stage train

# Run the BaseTrainer smoke test
python tests/test_base_trainer.py
```

## CLI

```
python src/train.py --config <path> ( --phase N | --stage NAME ) [--resume PATH] [--max-steps N] [--gpus N]
```

| Flag           | Meaning                                                              |
|----------------|----------------------------------------------------------------------|
| `--config`     | Path to a YAML config (required).                                    |
| `--phase N`    | 1-based stage index. Mutually exclusive with `--stage`.              |
| `--stage NAME` | Stage name (matches `stages[].name`). Mutually exclusive with `--phase`. |
| `--resume`     | Checkpoint to resume from (model + optimizer + scheduler + step).    |
| `--max-steps`  | Override `stages[].max_steps`. Useful for smoke tests.               |
| `--gpus`       | Number of GPUs for DDP. Default `1` (single-process, no spawn).      |

## Config schema

```yaml
project: example          # picks src/projects/<project>/trainer.py via TRAINERS

# Anything else here is project-specific and passed through as `self.config`.
dataset: { ... }
model:   { ... }

stages:                   # list of training stages; pick one per run
  - name: train           # used as checkpoint prefix and run name
    batch_size: 64
    learning_rate: 1.0e-3
    weight_decay: 0.0
    warmup_steps: 5
    max_steps: 30
    gradient_accumulation: 1
    val_every: 10
    save_every: 20
    precision: fp32       # bf16 | fp16 | fp32
    max_grad_norm: 1.0    # optional, default 1.0
```

Multiple stages let you express a curriculum (e.g. low-res pretrain → high-res
finetune): pick one with `--phase 1` then `--phase 2 --resume ...`.

## Adding a new project

Four steps. Copy `src/projects/example/` and edit.

### 1. Subclass `BaseTrainer`

```python
# src/projects/myproj/trainer.py
from core.distributed import is_main_process
from core.registry import TRAINERS
from core.trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def build_model(self):
        return MyModel(...).to(self.device)

    def build_data(self):
        return train_loader, val_loader   # both are torch DataLoaders

    def forward_step(self, batch):
        loss = ...
        return loss, {"loss_total": float(loss.detach())}   # second item is a log dict

    # Optional overrides:
    def validation_step(self): ...        # called rank-0-only at val_every cadence
    def init_logging(self): ...           # e.g. wandb.init(...)
    def finalize_logging(self): ...       # e.g. wandb.finish()
    def on_step_log(self, log_dict): ...  # rank-0, every step
    def on_window_log(self, avg_loss, log_dict): ...  # rank-0, every 100 steps
    def build_optimizer(self): ...        # default = AdamW + cosine_with_warmup
```

### 2. Register the trainer

At the bottom of `trainer.py`:

```python
TRAINERS["myproj"] = MyTrainer
```

The launcher discovers it by importing `projects.myproj.trainer`. The string
key must match the `project:` field in the config.

### 3. Write a config

```yaml
# configs/myproj.yaml
project: myproj
dataset: { ... }
model:   { ... }
stages:
  - name: phase1
    batch_size: 8
    learning_rate: 1.0e-4
    weight_decay: 0.01
    warmup_steps: 100
    max_steps: 5000
    gradient_accumulation: 1
    val_every: 100
    save_every: 1000
    precision: bf16
```

### 4. Run

```bash
python src/train.py --config configs/myproj.yaml --phase 1
```

## What `BaseTrainer.fit()` handles for you

- Builds model, wraps in DDP if `world_size > 1`, builds optimizer/scheduler.
- Runs the main loop with `autocast(precision)` and optional `GradScaler` (fp16 only).
- Gradient accumulation with correct DDP `no_sync()` on non-sync steps.
- Gradient clipping at `max_grad_norm` (default 1.0) on each optimizer step.
- Calls `validation_step()` at `val_every` (rank 0 only).
- Saves a checkpoint at `save_every` and a final checkpoint at the end.
- Logs `step / loss / lr` to a tqdm bar; calls `on_step_log` and `on_window_log`
  for your own logging backend (wandb, tensorboard, ...).
- Resumes model + optimizer + scheduler + step from `--resume PATH`.

What it does *not* do: data preprocessing, model construction, loss design,
metrics. Those live in your project.

## Multi-GPU (DDP)

Just add `--gpus N`. The launcher uses `torch.multiprocessing.spawn` to fan out
to N ranks, each calling `core.distributed.setup(rank, N)`. The trainer wraps
the model in `DistributedDataParallel` automatically.

If you're sharing data across ranks, use a `DistributedSampler` in
`build_data()` and seed it per-step (the loop doesn't do this for you because
not every project needs it).

## Component registry

`core/registry.py` exposes four dicts: `TRAINERS`, `MODELS`, `DATASETS`,
`LOSSES`. The launcher only uses `TRAINERS`. The other three are conventions
for projects that want to dispatch their model / dataset / loss by string key
from the config too — use them or ignore them.

```python
from core.registry import register, build, MODELS

@register(MODELS, "my_model_v2")
def _build(cfg, device):
    return MyModel(**cfg).to(device)

# elsewhere:
model = build(MODELS, config["model"]["name"], config["model"], device)
```

## Testing

```bash
python tests/test_base_trainer.py
```

Runs a CPU-only smoke test that exercises the loop, grad accumulation,
checkpoint cadence, validation cadence, and resume. Add tests under `tests/`
for your own projects in the same style.
