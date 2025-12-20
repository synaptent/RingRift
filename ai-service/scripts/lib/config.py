"""
Configuration Management Library

Provides unified configuration access for training scripts:
- Hyperparameter loading and merging
- Environment-based configuration
- Validation

Usage:
    from scripts.lib.config import TrainingConfig, get_config

    config = get_config("square8_2p")
    print(config.learning_rate)
    print(config.num_filters)
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    num_filters: int = 192
    num_res_blocks: int = 12
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    dropout: float = 0.1
    use_batch_norm: bool = False
    use_spectral_norm: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Learning rate
    learning_rate: float = 0.0003
    fine_tune_lr: float | None = None
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5

    # Batch and epochs
    batch_size: int = 256
    epochs: int = 50
    early_stopping_patience: int = 15

    # Regularization
    weight_decay: float = 0.0001
    dropout: float = 0.1
    label_smoothing: float = 0.0

    # Loss weights
    value_weight: float = 1.0
    policy_weight: float = 1.0

    # Data
    sampling_weights: str = "victory_type"
    validation_split: float = 0.15

    # Model architecture
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create config from dictionary."""
        model_data = {}
        config_data = {}

        for key, value in data.items():
            if key in ("num_filters", "num_res_blocks", "hidden_dim", "num_hidden_layers"):
                model_data[key] = value
            elif key == "model":
                model_data.update(value)
            else:
                config_data[key] = value

        config_data["model"] = ModelConfig(**model_data) if model_data else ModelConfig()
        return cls(**config_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Flatten model config for backwards compatibility
        model = result.pop("model")
        result.update(model)
        return result


@dataclass
class BoardConfig:
    """Board-specific configuration."""
    board_type: str
    num_players: int
    board_size: int = 8

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"

    @classmethod
    def from_config_key(cls, config_key: str) -> BoardConfig:
        """Parse from config key like 'square8_2p'."""
        parts = config_key.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid config key: {config_key}")

        board_type = parts[0]
        num_players = int(parts[1].rstrip("p"))

        # Infer board size
        board_size = 8
        if "19" in board_type:
            board_size = 19

        return cls(
            board_type=board_type,
            num_players=num_players,
            board_size=board_size,
        )


class ConfigManager:
    """Manages loading and merging configurations."""

    DEFAULT_HP_PATH = Path("config/hyperparameters.json")
    ENV_PREFIX = "RINGRIFT_"

    def __init__(self, hp_path: Path | None = None):
        """Initialize config manager.

        Args:
            hp_path: Path to hyperparameters.json (uses default if not specified)
        """
        self.hp_path = hp_path or self.DEFAULT_HP_PATH
        self._cache: dict[str, TrainingConfig] = {}
        self._raw_config: dict | None = None

    def _load_hp_file(self) -> dict[str, Any]:
        """Load hyperparameters.json file."""
        if self._raw_config is not None:
            return self._raw_config

        if not self.hp_path.exists():
            return {"defaults": {}, "configs": {}}

        try:
            with open(self.hp_path) as f:
                self._raw_config = json.load(f)
                return self._raw_config
        except Exception as e:
            print(f"Warning: Failed to load {self.hp_path}: {e}")
            return {"defaults": {}, "configs": {}}

    def get_config(
        self,
        config_key: str,
        override: dict[str, Any] | None = None,
    ) -> TrainingConfig:
        """Get training configuration for a board/player config.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            override: Optional overrides to apply

        Returns:
            TrainingConfig with merged settings
        """
        cache_key = f"{config_key}:{hash(str(override))}"
        if cache_key in self._cache and override is None:
            return self._cache[cache_key]

        hp_data = self._load_hp_file()

        # Start with defaults
        merged = dict(hp_data.get("defaults", {}))

        # Apply config-specific settings
        config_hp = hp_data.get("configs", {}).get(config_key, {}).get("hyperparameters")
        if config_hp:
            merged.update(config_hp)

        # Apply large_model settings if specified
        if hp_data.get("large_model") and merged.get("use_large_model"):
            merged.update(hp_data["large_model"])

        # Apply environment overrides
        merged = self._apply_env_overrides(merged)

        # Apply explicit overrides
        if override:
            merged.update(override)

        config = TrainingConfig.from_dict(merged)
        self._cache[cache_key] = config

        return config

    def _apply_env_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides."""
        env_mappings = {
            "LEARNING_RATE": ("learning_rate", float),
            "BATCH_SIZE": ("batch_size", int),
            "EPOCHS": ("epochs", int),
            "NUM_FILTERS": ("num_filters", int),
            "NUM_RES_BLOCKS": ("num_res_blocks", int),
            "HIDDEN_DIM": ("hidden_dim", int),
            "DROPOUT": ("dropout", float),
            "WEIGHT_DECAY": ("weight_decay", float),
        }

        result = dict(config)

        for env_name, (config_key, type_fn) in env_mappings.items():
            env_value = os.environ.get(f"{self.ENV_PREFIX}{env_name}")
            if env_value:
                with contextlib.suppress(ValueError):
                    result[config_key] = type_fn(env_value)

        return result

    def get_all_configs(self) -> dict[str, TrainingConfig]:
        """Get all defined configurations."""
        hp_data = self._load_hp_file()
        configs = {}

        for config_key in hp_data.get("configs", {}):
            configs[config_key] = self.get_config(config_key)

        return configs

    def get_defaults(self) -> TrainingConfig:
        """Get default configuration."""
        hp_data = self._load_hp_file()
        return TrainingConfig.from_dict(hp_data.get("defaults", {}))


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(
    config_key: str,
    override: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Get training configuration for a config key.

    Args:
        config_key: Configuration key (e.g., "square8_2p")
        override: Optional overrides

    Returns:
        TrainingConfig
    """
    return get_config_manager().get_config(config_key, override)


def get_board_config(config_key: str) -> BoardConfig:
    """Get board configuration from config key.

    Args:
        config_key: Configuration key (e.g., "square8_2p")

    Returns:
        BoardConfig
    """
    return BoardConfig.from_config_key(config_key)
