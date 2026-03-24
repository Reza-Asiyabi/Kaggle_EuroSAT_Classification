"""Reproducibility utilities: set all relevant random seeds."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, PyTorch (CPU + CUDA).

    Also configures cuDNN for deterministic behaviour.  Note that full
    determinism may slow training on some GPU models.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
