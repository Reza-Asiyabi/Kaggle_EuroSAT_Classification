"""
Model factory using the *timm* library.

Any architecture available in ``timm.list_models()`` can be used by setting
``model.architecture`` in the config.  The classifier head is replaced with a
new linear layer sized to ``data.num_classes``.

Supported preset names (all pretrained on ImageNet):
    • efficientnet_b0  (default, ~5 M params, fast)
    • efficientnet_b3
    • resnet50
    • convnext_tiny
    • vit_base_patch16_224    (ViT-B/16)
    • vit_small_patch16_224   (ViT-S/16, lighter)
    • swin_tiny_patch4_window7_224
"""
from __future__ import annotations

import logging
from typing import Optional

import timm
import torch
import torch.nn as nn

from src.utils.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Config, device: Optional[torch.device] = None) -> nn.Module:
    """Create, configure and move a timm model to *device*.

    The function:
    1. Instantiates the backbone via ``timm.create_model``.
    2. Replaces the classification head with a new linear layer.
    3. Optionally freezes backbone weights (for head-only warm-up).
    4. Logs parameter counts.

    Parameters
    ----------
    cfg:
        Loaded Config object.
    device:
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    nn.Module
        Ready-to-train model on *device*.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architecture: str = cfg.model.architecture
    pretrained: bool = cfg.model.pretrained
    num_classes: int = cfg.data.num_classes
    dropout_rate: float = cfg.model.dropout_rate

    logger.info(
        "Building model: %s  pretrained=%s  num_classes=%d",
        architecture,
        pretrained,
        num_classes,
    )

    # ── Create backbone ───────────────────────────────────────────────────────
    try:
        model = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,          # Remove original head; we add our own
            drop_rate=dropout_rate,
        )
    except RuntimeError as exc:
        available = timm.list_models(architecture + "*")
        raise RuntimeError(
            f"Architecture '{architecture}' not found in timm.  "
            f"Did you mean one of: {available[:10]}?"
        ) from exc

    # ── Custom classifier head ────────────────────────────────────────────────
    in_features: int = model.num_features
    classifier = _build_classifier(in_features, num_classes, dropout_rate)
    model = _ModelWithHead(model, classifier)

    # ── Optionally freeze backbone ────────────────────────────────────────────
    if cfg.model.freeze_backbone:
        _freeze_backbone(model)
        logger.info("Backbone frozen — only classifier head will be trained.")

    model = model.to(device)

    # ── Log parameter counts ──────────────────────────────────────────────────
    total, trainable = _count_parameters(model)
    logger.info(
        "Parameters — total: %s  trainable: %s",
        _fmt_params(total),
        _fmt_params(trainable),
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class _ModelWithHead(nn.Module):
    """Wraps a timm feature extractor with a custom classifier head."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(x)
        return self.classifier(features)

    def get_backbone(self) -> nn.Module:
        return self.backbone

    def get_classifier(self) -> nn.Module:
        return self.classifier


def _build_classifier(
    in_features: int,
    num_classes: int,
    dropout_rate: float,
) -> nn.Sequential:
    """Two-layer MLP head: BN → Dropout → Linear → (no activation — raw logits)."""
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )


def _freeze_backbone(model: _ModelWithHead) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (used by trainer + evaluate scripts)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    cfg: Config,
) -> None:
    """Save a full training checkpoint to *path*."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": cfg.to_dict(),
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load checkpoint from *path*, returning the metadata dict.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.
    model:
        Model instance whose weights will be loaded.
    optimizer, scheduler:
        Optional: if provided, their states are also restored.
    device:
        Map location for tensors.

    Returns
    -------
    dict
        ``{"epoch": int, "metrics": dict, "config": dict}``
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint from %s (epoch %d, val_acc=%.4f)",
        path,
        ckpt.get("epoch", -1),
        ckpt.get("metrics", {}).get("val_acc", float("nan")),
    )
    return {
        "epoch": ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
        "config": ckpt.get("config", {}),
    }
