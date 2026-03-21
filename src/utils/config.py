# Phase 1: YAML configuration loader
"""YAML configuration loader with inheritance support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

import yaml


class Config:
    """Configuration manager with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = {
            key: self._wrap(value)
            for key, value in config_dict.items()
        }

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value.to_dict()
        if isinstance(value, list):
            return [cls._unwrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        try:
            return self._config[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def __iter__(self) -> Iterator[str]:
        return iter(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: self._unwrap(value)
            for key, value in self._config.items()
        }

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def load_config(config_path: str, base_dir: str = "configs") -> Config:
    """
    Load YAML config with inheritance support.
    
    Args:
        config_path: Path to config file (relative to base_dir or absolute)
        base_dir: Base directory for config files
    
    Returns:
        Config object with merged settings
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(base_dir) / config_path
    
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'inherit' in config:
        parent_path = config.pop('inherit')
        if not parent_path.endswith('.yaml'):
            parent_path += '.yaml'
        parent_config = load_config(parent_path, base_dir).to_dict()
        # Merge: child overrides parent
        merged = _deep_merge(parent_config, config)
        config = merged
    
    return Config(config)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
