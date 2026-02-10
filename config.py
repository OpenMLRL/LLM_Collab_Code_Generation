"""
Configuration management for MLRL experiments.
Handles YAML loading and model configuration.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model loading and generation."""

    name: str
    type: str = "qwen"
    temperature: float = 0.7
    top_p: float = 0.9
    max_length: int = 2048
    special_tokens: Dict[str, str] = field(default_factory=dict)
    torch_dtype: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(
            name=config_dict.get("name", ""),
            type=config_dict.get("type", "qwen"),
            temperature=config_dict.get("temperature", 0.7),
            top_p=config_dict.get("top_p", 0.9),
            max_length=config_dict.get("max_length", 2048),
            special_tokens=config_dict.get("special_tokens", {}),
            torch_dtype=(
                config_dict.get("torch_dtype")
                or config_dict.get("dtype")
            ),
        )


class Config:
    """Simple configuration manager for YAML files."""

    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(self.path, "r") as f:
            self.data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value using dot notation (e.g., 'agent_model.name')."""
        keys = key.split(".")
        value = self.data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.data.get(section, {})

    def get_agent_model_config(self) -> ModelConfig:
        """Get agent model configuration as ModelConfig object."""
        model_section = self.get_section("agent_model")
        if not model_section:
            raise ValueError("No 'agent_model' section found in configuration")
        return ModelConfig.from_dict(model_section)

    def get_critic_model_config(self, required: bool = True) -> Optional[ModelConfig]:
        """Get critic model configuration as ModelConfig object."""
        critic_section = self.get_section("critic_model")
        if not critic_section:
            if required:
                raise ValueError("No 'critic_model' section found in configuration")
            return None
        return ModelConfig.from_dict(critic_section)

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values (deep merge)."""
        self._deep_merge(self.data, updates)

    def _deep_merge(self, base: dict, updates: dict):
        """Recursively merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: Optional[str] = None):
        """Save configuration to YAML file."""
        save_path = Path(output_path) if output_path else self.path
        with open(save_path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add configuration arguments to parser."""
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--override", nargs="*", help="Override config values (format: key=value)"
    )
    return parser


def parse_overrides(overrides: list) -> Dict[str, Any]:
    """Parse command-line overrides into nested dictionary."""
    if not overrides:
        return {}

    result = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: {override}. Use key=value format."
            )

        key, value = override.split("=", 1)
        keys = key.split(".")

        try:
            import ast

            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep as string

        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    return result
