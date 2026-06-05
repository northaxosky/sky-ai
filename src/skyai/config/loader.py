"""YAML config loading with hierarchical extends and CLI overrides"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skyai.config.schema import RunConfig


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML config, resolve any `extends:` chain, apply overrides, validate"""
    merged = _load_and_merge(Path(path), seen=set())
    if overrides:
        for override in overrides:
            _apply_override(merged, override)
    return RunConfig.model_validate(merged)


def _load_and_merge(path: Path, seen: set[Path]) -> dict[str, Any]:
    """Load a YAML file and recursively merge with any `extends:` parent"""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found @: {path}")
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(path) for path in seen) + f" -> {resolved}"
        raise ValueError(f"Cyclic extends chain: {chain}")
    seen = seen | {resolved}

    with path.open() as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    parent_path = (path.parent / parent_ref).resolve()
    parent_data = _load_and_merge(parent_path, seen)
    return _deep_merge(parent_data, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, Override wins for non-dict leaves"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_override(data: dict[str, Any], override: str) -> None:
    """Apply a `a.b.c = value` override into a nested dict in place"""
    if "=" not in override:
        raise ValueError(f"Override must be in 'key=value`; got {override!r}")
    key, _, value = override.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"Empty key in override: {override!r}")

    parts = key.split(".")
    cursor: dict[str, Any] = data
    for part in parts[:-1]:
        next_cursor = cursor.get(part)
        if not isinstance(next_cursor, dict):
            next_cursor = {}
            cursor[part] = next_cursor
        cursor = next_cursor
    cursor[parts[-1]] = _parse_scalar(value)


def _parse_scalar(value: str) -> Any:
    """Parse a CLI override value string"""
    value = value.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("none", "null"):
        return None

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
