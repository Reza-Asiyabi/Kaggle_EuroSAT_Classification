"""
Trainer
=======
Full training loop with:
  • mixed-precision (torch.cuda.amp)
  • gradient clipping
  • LR scheduling with linear warm-up
  • checkpoint saving (periodic + best-model)
  • optional Weights & Biases logging
  • resume-from-checkpoint
  • clean per-epoch console summary
"""
from __future__ import annotations

import heapq
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import save_checkpoint, load_checkpoint
from src.utils.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    opt_name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True
        )
    raise ValueError(f"Unknown optimizer: {opt_name}")


def _build_scheduler(cfg: Config, optimizer: torch.optim.Optimizer, num_epochs: int):
    """Return (scheduler, uses_epoch_step) pair.

    *uses_epoch_step=True* means ``scheduler.step()`` is called once per epoch;
    *False* means it is called every batch.
    """
    name = cfg.training.scheduler.lower()

    if name == "cosine":
        # Cosine annealing from base LR → min_lr
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=cfg.training.min_lr,
        )
        return sch, True

    if name == "step":
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.step_size,
            gamma=cfg.training.gamma,
        )
        return sch, True

    if name == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=True,
        )
        return sch, True  # called with val_acc

    if name in ("none", ""):
        return None, True

    raise ValueError(f"Unknown scheduler: {name}")


class _WarmupWrapper:
    """Linear warm-up wrapper around any PyTorch LR scheduler."""

    def __init__(
        self,
        inner,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_lr: float,
        is_plateau: bool = False,
    ) -> None:
        self._inner = inner
        self._optimizer = optimizer
        self._warmup_epochs = warmup_epochs
        self._base_lr = base_lr
        self._is_plateau = is_plateau
        self._current_epoch = 0

    def step(self, metric: Optional[float] = None) -> None:
        self._current_epoch += 1
        if self._current_epoch <= self._warmup_epochs:
            scale = self._current_epoch / max(1, self._warmup_epochs)
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._base_lr * scale
        else:
            if self._inner is not None:
                if self._is_plateau and metric is not None:
                    self._inner.step(metric)
                else:
                    self._inner.step()

    def state_dict(self) -> dict:
        return {
            "current_epoch": self._current_epoch,
            "inner": self._inner.state_dict() if self._inner else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self._current_epoch = state.get("current_epoch", 0)
        inner_state = state.get("inner")
        if self._inner is not None and inner_state is not None:
            self._inner.load_state_dict(inner_state)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self._optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint manager (keep top-k best by val_acc)
# ─────────────────────────────────────────────────────────────────────────────

class _TopKCheckpointManager:
    """Keep the *k* best checkpoints (by val accuracy) on disk."""

    def __init__(self, ckpt_dir: Path, k: int = 3) -> None:
        self._dir = ckpt_dir
        self._k = k
        # min-heap of (val_acc, path)
        self._heap: list[tuple[float, str]] = []

    def update(
        self,
        val_acc: float,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        metrics: dict,
        cfg: Config,
    ) -> tuple[bool, str]:
        """Save checkpoint if it belongs to the top-k.

        Returns (is_best, path).
        """
        path = str(self._dir / f"ckpt_epoch{epoch:03d}_acc{val_acc:.4f}.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, cfg)

        is_best = False
        if len(self._heap) < self._k or val_acc > self._heap[0][0]:
            heapq.heappush(self._heap, (val_acc, path))
            is_best = val_acc >= self._heap[-1][0]

            # Evict worst checkpoint when over capacity
            if len(self._heap) > self._k:
                _, worst_path = heapq.heappop(self._heap)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    logger.debug("Removed old checkpoint: %s", worst_path)
        else:
            # Not in top-k → remove immediately
            if os.path.exists(path):
                os.remove(path)

        return is_best, path

    @property
    def best_path(self) -> Optional[str]:
        if not self._heap:
            return None
        return max(self._heap, key=lambda x: x[0])[1]


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer class
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates training, validation, checkpointing and logging.

    Parameters
    ----------
    model:
        Instantiated model on the correct device.
    train_loader, val_loader:
        DataLoaders for training and validation splits.
    cfg:
        Loaded Config object.
    device:
        Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Training components ────────────────────────────────────────────────
        self.optimizer = _build_optimizer(cfg, model)
        _inner_sch, _ = _build_scheduler(cfg, self.optimizer, cfg.training.epochs)
        is_plateau = cfg.training.scheduler.lower() == "plateau"
        self.scheduler = _WarmupWrapper(
            _inner_sch,
            self.optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            base_lr=cfg.training.learning_rate,
            is_plateau=is_plateau,
        )

        label_smoothing = getattr(cfg.training, "label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.training.mixed_precision and torch.cuda.is_available()
        )

        # ── Checkpoint & resume ────────────────────────────────────────────────
        ckpt_dir = Path(cfg.training.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_manager = _TopKCheckpointManager(
            ckpt_dir, k=getattr(cfg.training, "keep_top_k", 3)
        )
        self._start_epoch = 1
        self._best_val_acc = 0.0

        # ── Metrics history ────────────────────────────────────────────────────
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # ── Optional W&B ──────────────────────────────────────────────────────
        self._wandb_run = None
        if cfg.logging.use_wandb:
            self._init_wandb()

    # ── Resume ────────────────────────────────────────────────────────────────

    def resume(self, checkpoint_path: str) -> None:
        """Restore model, optimizer, scheduler state from *checkpoint_path*."""
        meta = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device,
        )
        self._start_epoch = meta["epoch"] + 1
        self._best_val_acc = meta["metrics"].get("val_acc", 0.0)
        logger.info(
            "Resuming from epoch %d (best val_acc=%.4f)",
            self._start_epoch,
            self._best_val_acc,
        )

    # ── Training entry point ──────────────────────────────────────────────────

    def fit(self) -> dict[str, list[float]]:
        """Run the complete training loop.

        Returns
        -------
        dict
            Per-epoch metric history.
        """
        cfg = self.cfg
        total_epochs = cfg.training.epochs
        log_every = cfg.logging.log_every_n_steps
        grad_clip = cfg.training.grad_clip

        # Optional resume
        if cfg.training.resume_from:
            self.resume(cfg.training.resume_from)

        best_path = str(Path(cfg.training.checkpoint_dir) / "best_model.pt")

        for epoch in range(self._start_epoch, total_epochs + 1):
            epoch_start = time.perf_counter()

            # ── Train one epoch ────────────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(epoch, log_every, grad_clip)

            # ── Validate ───────────────────────────────────────────────────────
            val_loss, val_acc = self._val_epoch()

            # ── LR step ────────────────────────────────────────────────────────
            self.scheduler.step(metric=val_acc)
            current_lr = self.scheduler.get_last_lr()[0]

            # ── Record history ─────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            epoch_time = time.perf_counter() - epoch_start

            logger.info(
                "Epoch [%d/%d]  "
                "train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f  "
                "lr=%.2e  time=%.1fs",
                epoch, total_epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, epoch_time,
            )

            # ── Checkpointing ─────────────────────────────────────────────────
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }

            is_best, ckpt_path = self._ckpt_manager.update(
                val_acc, epoch, self.model, self.optimizer, self.scheduler, metrics, cfg
            )

            # Always keep a periodic snapshot regardless of quality
            if epoch % cfg.training.save_every_n_epochs == 0:
                periodic_path = str(
                    Path(cfg.training.checkpoint_dir) / f"periodic_epoch{epoch:03d}.pt"
                )
                save_checkpoint(
                    periodic_path, self.model, self.optimizer,
                    self.scheduler, epoch, metrics, cfg
                )

            if is_best or val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                save_checkpoint(
                    best_path, self.model, self.optimizer,
                    self.scheduler, epoch, metrics, cfg
                )
                logger.info("  *** New best model (val_acc=%.4f) ***", val_acc)

            # ── W&B ───────────────────────────────────────────────────────────
            if self._wandb_run:
                self._wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    }
                )

        logger.info("Training complete.  Best val_acc=%.4f", self._best_val_acc)
        return self.history

    # ── One epoch helpers ─────────────────────────────────────────────────────

    def _train_epoch(
        self, epoch: int, log_every: int, grad_clip: float
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )

        for step, (images, labels) in enumerate(pbar, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            if step % log_every == 0:
                running_loss = total_loss / total
                running_acc = correct / total
                pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        avg_loss = total_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(
            self.val_loader, desc="Validation", leave=False, dynamic_ncols=True
        ):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        return total_loss / total, correct / total

    # ── W&B setup ─────────────────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        try:
            import wandb

            entity = self.cfg.logging.wandb_entity or None
            self._wandb_run = wandb.init(
                project=self.cfg.logging.wandb_project,
                entity=entity,
                config=self.cfg.to_dict(),
                name=self.cfg.project.name,
                resume="allow",
            )
            logger.info("W&B run initialised: %s", self._wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")
        except Exception as exc:
            logger.warning("W&B initialisation failed: %s", exc)
