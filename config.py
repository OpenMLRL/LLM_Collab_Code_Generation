"""Configuration helpers for code-generation experiments."""

from __future__ import annotations

import argparse
import ast
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Model loading and sampling settings from a ``*_model`` config section."""

    name: str
    type: str = "qwen"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    special_tokens: Dict[str, str] = field(default_factory=dict)
    torch_dtype: Optional[str] = None
    attn_implementation: Optional[str] = "sdpa"

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        *,
        require_sampling: bool = True,
    ) -> "ModelConfig":
        if require_sampling:
            missing = [
                key
                for key in ("temperature", "top_p", "top_k")
                if key not in config_dict
            ]
            if missing:
                raise ValueError(
                    f"agent_model is missing required sampling fields: {', '.join(missing)}"
                )

        temperature = _optional_float(config_dict.get("temperature"))
        top_p = _optional_float(config_dict.get("top_p"))
        top_k = _optional_int(config_dict.get("top_k"))
        if require_sampling and (temperature is None or top_p is None):
            raise ValueError(
                "agent_model.temperature and agent_model.top_p must be non-null."
            )

        return cls(
            name=config_dict.get("name", ""),
            type=config_dict.get("type", "qwen"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            special_tokens=config_dict.get("special_tokens", {}),
            torch_dtype=config_dict.get("torch_dtype") or config_dict.get("dtype"),
            attn_implementation=config_dict.get("attn_implementation", "sdpa"),
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value: {value}") from exc


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid int value: {value}") from exc


class Config:
    """YAML experiment configuration with dotted-key access."""

    def __init__(self, config_path: str):
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with self.path.open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at a dotted key such as ``agent_model.name``."""
        value: Any = self.data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a copy of a top-level section ({} when missing or not a mapping)."""
        value = self.data.get(section, {})
        return dict(value) if isinstance(value, dict) else {}

    def get_agent_model_config(self) -> ModelConfig:
        model_section = self.get_section("agent_model")
        if not model_section:
            raise ValueError("No 'agent_model' section found in configuration")
        return ModelConfig.from_dict(model_section, require_sampling=True)

    def get_critic_model_config(self, required: bool = True) -> Optional[ModelConfig]:
        critic_section = self.get_section("critic_model")
        if not critic_section:
            if required:
                raise ValueError("No 'critic_model' section found in configuration")
            return None
        return ModelConfig.from_dict(critic_section, require_sampling=False)

    def update(self, updates: Dict[str, Any]) -> None:
        """Deep-merge ``updates`` into the configuration."""
        self._deep_merge(self.data, updates)

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: Optional[str] = None) -> None:
        """Write the configuration as YAML (to the loaded file by default)."""
        path = Path(output_path) if output_path else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.dump(self.data, handle, default_flow_style=False, sort_keys=False)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--override", nargs="*", help="Override config values (format: key=value)"
    )
    return parser


def _parse_override_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_overrides(overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``key.path=value`` overrides into a nested dictionary."""
    result: Dict[str, Any] = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: {override}. Use key=value format."
            )
        key, value = override.split("=", 1)
        keys = key.split(".")
        current = result
        for part in keys[:-1]:
            current = current.setdefault(part, {})
        current[keys[-1]] = _parse_override_value(value)
    return result


def run_id() -> str:
    """Return a time-pid id so concurrent launches write to separate directories."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
