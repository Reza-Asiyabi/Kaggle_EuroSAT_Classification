"""Logging configuration: produces clean, readable console + file output."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Mapping of string level names (from config) to logging constants
_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logging(
    level: str | int = "info",
    log_file: str | Path | None = None,
) -> None:
    """Configure the root logger with a formatter, console handler, and
    optionally a rotating file handler.

    Call this once at the start of each entry-point script.

    Parameters
    ----------
    level:
        Log level as a string (``"info"``, ``"debug"``, …) or an integer.
    log_file:
        If provided, log messages are also written to this file.
    """
    if isinstance(level, str):
        level = _LEVEL_MAP.get(level.lower(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate output on re-import
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ────────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── File handler ───────────────────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress overly verbose third-party loggers
    for noisy in ("PIL", "matplotlib", "timm", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
