"""Training Targets Configuration Loader.

December 2025: Provides typed access to training targets defined in training_targets.yaml.

Usage:
    from app.config.training_targets import (
        get_training_targets,
        get_all_configs,
        is_target_reached,
        get_priority_configs,
    )

    # Get targets for a specific config
    targets = get_training_targets("hex8_2p")
    print(f"Elo target: {targets.elo_target}")

    # Check if training target is reached
    if is_target_reached("hex8_2p", current_elo=1650):
        print("Target reached!")

    # Get high-priority configs for selfplay scheduling
    high_priority = get_priority_configs("high")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingTargets",
    "get_training_targets",
    "get_all_configs",
    "is_target_reached",
    "get_priority_configs",
    "get_priority_weight",
    "reload_targets",
]


@dataclass
class TrainingTargets:
    """Training targets for a specific board configuration."""

    config_key: str

    # Elo targets
    elo_target: int = 1800
    elo_tolerance: int = 50

    # Selfplay parameters
    min_games_for_training: int = 1000
    max_games_per_day: int = 10000

    # Training parameters
    training_epochs: int = 50
    early_stopping_patience: int = 10

    # Evaluation parameters
    gauntlet_games: int = 50
    promotion_threshold_random: float = 0.85
    promotion_threshold_heuristic: float = 0.80  # Dec 2025: raised from 0.60 for 2000+ Elo

    # Priority
    priority: str = "medium"

    def is_reached(self, current_elo: int) -> bool:
        """Check if the Elo target is reached within tolerance."""
        return current_elo >= (self.elo_target - self.elo_tolerance)

    def games_needed(self, current_games: int) -> int:
        """Calculate games needed before next training run."""
        return max(0, self.min_games_for_training - current_games)


# Cache for loaded configuration
_config_cache: dict[str, Any] | None = None
_targets_cache: dict[str, TrainingTargets] = {}


def _get_config_path() -> Path:
    """Get path to training_targets.yaml."""
    return Path(__file__).parent / "training_targets.yaml"


def _load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    config_path = _get_config_path()
    if not config_path.exists():
        logger.warning(f"Training targets file not found: {config_path}")
        return {"defaults": {}, "configs": {}, "priority_weights": {}}

    try:
        with open(config_path) as f:
            _config_cache = yaml.safe_load(f)
            return _config_cache
    except yaml.YAMLError as e:
        logger.error(f"Error parsing training targets: {e}")
        return {"defaults": {}, "configs": {}, "priority_weights": {}}


def reload_targets() -> None:
    """Reload configuration from disk (clears cache)."""
    global _config_cache, _targets_cache
    _config_cache = None
    _targets_cache = {}
    logger.info("Training targets cache cleared")


def get_training_targets(config_key: str) -> TrainingTargets:
    """Get training targets for a specific configuration.

    Args:
        config_key: Configuration key like "hex8_2p", "square8_4p"

    Returns:
        TrainingTargets with merged defaults and config-specific overrides
    """
    if config_key in _targets_cache:
        return _targets_cache[config_key]

    config = _load_config()
    defaults = config.get("defaults", {})
    config_overrides = config.get("configs", {}).get(config_key, {})

    # Merge defaults with overrides
    merged = {**defaults, **config_overrides}

    targets = TrainingTargets(
        config_key=config_key,
        elo_target=merged.get("elo_target", 1800),
        elo_tolerance=merged.get("elo_tolerance", 50),
        min_games_for_training=merged.get("min_games_for_training", 1000),
        max_games_per_day=merged.get("max_games_per_day", 10000),
        training_epochs=merged.get("training_epochs", 50),
        early_stopping_patience=merged.get("early_stopping_patience", 10),
        gauntlet_games=merged.get("gauntlet_games", 50),
        promotion_threshold_random=merged.get("promotion_threshold_random", 0.85),
        promotion_threshold_heuristic=merged.get("promotion_threshold_heuristic", 0.60),
        priority=merged.get("priority", "medium"),
    )

    _targets_cache[config_key] = targets
    return targets


def get_all_configs() -> list[str]:
    """Get list of all configured config keys."""
    config = _load_config()
    return list(config.get("configs", {}).keys())


def is_target_reached(config_key: str, current_elo: int) -> bool:
    """Check if training target is reached for a configuration.

    Args:
        config_key: Configuration key like "hex8_2p"
        current_elo: Current Elo rating for this configuration

    Returns:
        True if current Elo is within tolerance of target
    """
    targets = get_training_targets(config_key)
    return targets.is_reached(current_elo)


def get_priority_configs(priority: str) -> list[str]:
    """Get all configurations with a specific priority level.

    Args:
        priority: Priority level ("high", "medium", "low")

    Returns:
        List of config keys with that priority
    """
    config = _load_config()
    configs = config.get("configs", {})

    return [
        key for key, cfg in configs.items()
        if cfg.get("priority", "medium") == priority
    ]


def get_priority_weight(priority: str) -> float:
    """Get the weight for a priority level (for selfplay scheduling).

    Args:
        priority: Priority level ("high", "medium", "low")

    Returns:
        Weight as float (0.0 to 1.0)
    """
    config = _load_config()
    weights = config.get("priority_weights", {})
    return weights.get(priority, 0.33)


def get_configs_by_priority() -> dict[str, list[str]]:
    """Get all configurations grouped by priority.

    Returns:
        Dict mapping priority -> list of config keys
    """
    return {
        "high": get_priority_configs("high"),
        "medium": get_priority_configs("medium"),
        "low": get_priority_configs("low"),
    }
