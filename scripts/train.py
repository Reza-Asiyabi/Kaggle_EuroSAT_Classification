#!/usr/bin/env python3
"""
train.py — EuroSAT Classification Training Entry Point
=======================================================

Usage examples
--------------
# Basic run with default config:
    python scripts/train.py

# Override config values from the CLI:
    python scripts/train.py --config configs/config.yaml \\
        --override training.epochs=50 \\
        --override model.architecture=resnet50 \\
        --override data.dataset_path=/path/to/EuroSAT

# Resume training from a checkpoint:
    python scripts/train.py \\
        --override training.resume_from=outputs/checkpoints/best_model.pt

# Run with a ViT:
    python scripts/train.py \\
        --override model.architecture=vit_small_patch16_224 \\
        --override training.batch_size=32 \\
        --override training.learning_rate=3e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Make project root importable regardless of working directory ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.datasets.eurosat import build_dataloaders
from src.evaluation.evaluator import Evaluator
from src.models.model import build_model, load_checkpoint
from src.training.trainer import Trainer
from src.utils import get_logger, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EuroSAT classification model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value.  Can be specified multiple times.\n"
            "Example: --override training.epochs=50"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config, overrides=args.overrides)

    # ── Logging ────────────────────────────────────────────────────────────────
    output_dir = Path(cfg.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        level="info",
        log_file=output_dir / "train.log",
    )
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("  EuroSAT Classification — Training")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    if args.overrides:
        logger.info("Overrides: %s", args.overrides)

    # ── Reproducibility ────────────────────────────────────────────────────────
    set_seed(cfg.project.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  (VRAM: %.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading dataset from: %s", cfg.data.dataset_path)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg, seed=cfg.project.seed
    )
    logger.info(
        "Batches — train: %d  val: %d  test: %d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, device=device)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg, device=device)
    history = trainer.fit()

    # ── Save training curves ───────────────────────────────────────────────────
    Evaluator.plot_training_curves(history, output_dir=cfg.evaluation.output_dir)

    # ── Save history JSON ─────────────────────────────────────────────────────
    history_path = output_dir / "history.json"
    with history_path.open("w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("History saved → %s", history_path)

    # ── Final test evaluation on best model ───────────────────────────────────
    best_ckpt = Path(cfg.training.checkpoint_dir) / "best_model.pt"
    if best_ckpt.exists():
        logger.info("Loading best model for final test evaluation...")
        load_checkpoint(str(best_ckpt), model, device=device)
    else:
        logger.warning("best_model.pt not found; using last epoch weights.")

    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        class_names=class_names,
        output_dir=cfg.evaluation.output_dir,
        device=device,
    )
    results = evaluator.run()

    logger.info("=" * 60)
    logger.info("  Final Test Results")
    logger.info("  Accuracy  : %.4f", results["accuracy"])
    logger.info("  F1 Macro  : %.4f", results["f1_macro"])
    logger.info("  F1 Weighted: %.4f", results["f1_weighted"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
