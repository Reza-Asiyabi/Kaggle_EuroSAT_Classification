#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation on any split / checkpoint
=============================================================

Usage examples
--------------
# Evaluate best model on test split (default):
    python scripts/evaluate.py

# Evaluate a specific checkpoint on the test split:
    python scripts/evaluate.py \\
        --checkpoint outputs/checkpoints/best_model.pt

# Evaluate on the validation split:
    python scripts/evaluate.py --split val

# Use a different config:
    python scripts/evaluate.py \\
        --config configs/config.yaml \\
        --checkpoint outputs/checkpoints/best_model.pt \\
        --override data.dataset_path=/kaggle/input/eurosat-dataset/EuroSAT
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.datasets.eurosat import build_dataloaders
from src.evaluation.evaluator import Evaluator
from src.models.model import build_model, load_checkpoint
from src.utils import get_logger, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EuroSAT model on a dataset split",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a .pt checkpoint file.  "
            "Defaults to outputs/checkpoints/best_model.pt"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which data split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value (can be specified multiple times).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    output_dir = Path(cfg.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="info", log_file=output_dir / "evaluate.log")
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("  EuroSAT Classification — Evaluation (%s split)", args.split)
    logger.info("=" * 60)

    set_seed(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg, seed=cfg.project.seed
    )
    split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = split_map[args.split]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, device=device)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        cfg_ckpt = getattr(cfg.evaluation, "checkpoint_path", None)
        if cfg_ckpt:
            ckpt_path = cfg_ckpt
        else:
            ckpt_path = str(Path(cfg.training.checkpoint_dir) / "best_model.pt")

    if not Path(ckpt_path).exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        logger.error("Run training first: python scripts/train.py")
        sys.exit(1)

    load_checkpoint(ckpt_path, model, device=device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = Evaluator(
        model=model,
        data_loader=loader,
        class_names=class_names,
        output_dir=cfg.evaluation.output_dir,
        device=device,
    )
    results = evaluator.run()

    logger.info("=" * 60)
    logger.info("  Results on [%s] split", args.split)
    logger.info("  Accuracy  : %.4f (%.2f%%)", results["accuracy"], results["accuracy"] * 100)
    logger.info("  F1 Macro  : %.4f", results["f1_macro"])
    logger.info("  F1 Weighted: %.4f", results["f1_weighted"])
    logger.info("  Output dir: %s", cfg.evaluation.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
