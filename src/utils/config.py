"""
Configuration management: YAML loading with dot-notation access and
CLI override support.

Usage
-----
    cfg = load_config("configs/config.yaml")
    cfg = load_config("configs/config.yaml", overrides=["training.epochs=50", "model.architecture=resnet50"])
    print(cfg.training.learning_rate)
"""
from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Recursive dot-notation wrapper around a plain dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    # ── dict-like helpers ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain nested dict."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    # ── merge / update ─────────────────────────────────────────────────────────

    def update(self, overrides: dict[str, Any]) -> None:
        """Merge a flat dict of dot-paths into this config in-place.

        Example::

            cfg.update({"training.learning_rate": 5e-4})
        """
        for dotpath, value in overrides.items():
            self._set_dotpath(dotpath, value)

    def _set_dotpath(self, dotpath: str, value: Any) -> None:
        keys = dotpath.split(".")
        node = self
        for key in keys[:-1]:
            if not hasattr(node, key):
                setattr(node, key, Config({}))
            node = getattr(node, key)
        # Cast to existing type when possible
        last_key = keys[-1]
        if hasattr(node, last_key):
            existing = getattr(node, last_key)
            if existing is not None:
                try:
                    value = type(existing)(value)
                except (TypeError, ValueError):
                    pass
        setattr(node, last_key, value)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> Config:
    """Load a YAML config file, optionally applying CLI overrides.

    Parameters
    ----------
    config_path:
        Path to the primary YAML configuration file.
    overrides:
        List of ``"section.key=value"`` strings that override individual
        config values after loading.  Values are cast to the type of the
        existing field.

    Returns
    -------
    Config
        Dot-notation config object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = Config(raw)

    if overrides:
        parsed: dict[str, Any] = {}
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Override must be in 'section.key=value' format, got: {override!r}"
                )
            dotpath, _, raw_value = override.partition("=")
            # Try to parse as Python literal for booleans / numbers / None
            parsed_value: Any = _parse_scalar(raw_value)
            parsed[dotpath.strip()] = parsed_value
        cfg.update(parsed)

    return cfg


def _parse_scalar(value: str) -> Any:
    """Convert a string CLI value to an appropriate Python type."""
    # YAML scalar parsing handles null, true, false, numbers, etc.
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value
