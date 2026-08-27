"""YAML configuration loading with deterministic deep merging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


Config = dict[str, Any]


def deep_merge(base: Config, override: Mapping[str, Any]) -> Config:
    merged: Config = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config_file(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return dict(payload)


def load_config_files(paths: Sequence[Path]) -> Config:
    if not paths:
        raise ValueError("At least one config path is required.")
    config: Config = {}
    for path in paths:
        config = deep_merge(config, load_config_file(path))
    return config


def parse_override(override: str) -> Config:
    if "=" not in override:
        raise ValueError("Config overrides must use key=value syntax.")
    key, raw_value = override.split("=", 1)
    if not key:
        raise ValueError("Config override key must not be empty.")
    value = yaml.safe_load(raw_value)
    if value is None and raw_value.lower() not in {"null", "none", "~"}:
        value = raw_value

    nested: Config = {}
    cursor = nested
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError("Config override keys must not contain empty path segments.")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return nested


def apply_overrides(config: Config, overrides: Sequence[str]) -> Config:
    merged = dict(config)
    for override in overrides:
        merged = deep_merge(merged, parse_override(override))
    return merged
