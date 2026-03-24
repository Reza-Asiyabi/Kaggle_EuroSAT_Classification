"""
EuroSAT Dataset
===============
Loads the EuroSAT RGB land-use / land-cover dataset from a folder whose
structure is ``<root>/<class_name>/<image>.jpg``.

Kaggle source : https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
               (or any mirror — the folder structure is identical)

Classes (10)
------------
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.config import Config

logger = logging.getLogger(__name__)

# ImageNet statistics — standard for fine-tuning pretrained models
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class EuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB images stored in ImageFolder format.

    Parameters
    ----------
    file_paths:
        List of absolute paths to image files.
    labels:
        Integer class index for each file path.
    class_names:
        Ordered list of class name strings.
    transform:
        Optional torchvision transform pipeline applied to each sample.
    """

    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        class_names: list[str],
        transform: Optional[Callable] = None,
    ) -> None:
        assert len(file_paths) == len(labels), "Lengths must match"
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label  # type: ignore[return-value]

    # ── Convenience ───────────────────────────────────────────────────────────

    def class_counts(self) -> dict[str, int]:
        """Return a mapping from class name → sample count."""
        counts: dict[str, int] = {name: 0 for name in self.class_names}
        for lbl in self.labels:
            counts[self.class_names[lbl]] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transform(cfg: Config) -> transforms.Compose:
    """Build the training augmentation pipeline from config."""
    aug = cfg.augmentation.train
    img_size: int = cfg.data.image_size

    pipeline: list[Callable] = []

    if getattr(aug, "random_resized_crop", False):
        pipeline.append(
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        )
    else:
        pipeline.append(transforms.Resize((img_size, img_size)))

    if getattr(aug, "random_horizontal_flip", False):
        pipeline.append(transforms.RandomHorizontalFlip(p=0.5))

    if getattr(aug, "random_vertical_flip", False):
        pipeline.append(transforms.RandomVerticalFlip(p=0.5))

    rotation = getattr(aug, "random_rotation", 0)
    if rotation > 0:
        pipeline.append(transforms.RandomRotation(degrees=rotation))

    cj = getattr(aug, "color_jitter", None)
    if cj is not None:
        pipeline.append(
            transforms.ColorJitter(
                brightness=getattr(cj, "brightness", 0),
                contrast=getattr(cj, "contrast", 0),
                saturation=getattr(cj, "saturation", 0),
                hue=getattr(cj, "hue", 0),
            )
        )

    pipeline.append(transforms.ToTensor())

    norm = aug.normalize
    pipeline.append(transforms.Normalize(mean=norm.mean, std=norm.std))

    re_cfg = getattr(aug, "random_erasing", None)
    if re_cfg is not None and getattr(re_cfg, "enabled", False):
        pipeline.append(
            transforms.RandomErasing(
                p=getattr(re_cfg, "probability", 0.25),
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value=0,
            )
        )

    return transforms.Compose(pipeline)


def build_val_transform(cfg: Config) -> transforms.Compose:
    """Build the validation / test transform (no augmentation)."""
    aug = cfg.augmentation.val
    img_size: int = cfg.data.image_size

    norm = aug.normalize
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm.mean, std=norm.std),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _discover_files(
    root: Path,
    class_names: list[str],
) -> tuple[list[Path], list[int]]:
    """Walk *root* and collect (path, label) pairs.

    Only files whose parent directory name is in *class_names* are included.
    This tolerates extra sub-directories (e.g. `__MACOSX`).
    """
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    file_paths: list[Path] = []
    labels: list[int] = []

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    found_classes: set[str] = set()

    for class_name, idx in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            logger.warning("Class directory not found: %s", class_dir)
            continue
        found_classes.add(class_name)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                file_paths.append(img_path)
                labels.append(idx)

    missing = set(class_names) - found_classes
    if missing:
        logger.warning("Missing class directories: %s", missing)

    logger.info(
        "Discovered %d images across %d classes in %s",
        len(file_paths),
        len(found_classes),
        root,
    )
    return file_paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: Config,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train / val / test DataLoaders from the config.

    The dataset is split in a stratified fashion so that class proportions
    are preserved across all three splits.

    Parameters
    ----------
    cfg:
        Loaded Config object.
    seed:
        Random seed for reproducible splitting.

    Returns
    -------
    tuple of (train_loader, val_loader, test_loader, class_names)
    """
    root = Path(cfg.data.dataset_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {root}\n"
            "Please check `data.dataset_path` in your config."
        )

    class_names: list[str] = cfg.data.classes

    all_paths, all_labels = _discover_files(root, class_names)

    if len(all_paths) == 0:
        raise RuntimeError(
            f"No images found under {root}. "
            "Verify the dataset structure matches: <root>/<class>/<image>.*"
        )

    # ── Stratified train / (val+test) split ────────────────────────────────
    val_test_ratio = cfg.data.val_split + cfg.data.test_split
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=seed,
    )

    # ── Stratified val / test split ────────────────────────────────────────
    relative_test = cfg.data.test_split / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths,
        rest_labels,
        test_size=relative_test,
        stratify=rest_labels,
        random_state=seed,
    )

    logger.info(
        "Split sizes → train: %d  val: %d  test: %d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    train_tf = build_train_transform(cfg)
    val_tf = build_val_transform(cfg)

    train_ds = EuroSATDataset(train_paths, train_labels, class_names, train_tf)
    val_ds = EuroSATDataset(val_paths, val_labels, class_names, val_tf)
    test_ds = EuroSATDataset(test_paths, test_labels, class_names, val_tf)

    num_workers: int = cfg.data.num_workers
    pin_memory: bool = cfg.data.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names
