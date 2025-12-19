"""Hyperparameter loading and management utilities.

This module provides functions to load optimized hyperparameters
for different board/player configurations from the central config.

Usage:
    from app.config.hyperparameters import get_hyperparameters

    # Get optimized params for square8 2p
    params = get_hyperparameters("square8", 2)

    # Use in training
    train_nnue(
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        hidden_dim=params["hidden_dim"],
        ...
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

# Path to hyperparameters config
CONFIG_PATH = AI_SERVICE_ROOT / "config" / "hyperparameters.json"

# Default hyperparameters (fallback if config not found)
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 0.0003,
    "batch_size": 256,
    "hidden_dim": 256,
    "num_hidden_layers": 2,
    "weight_decay": 0.0001,
    "dropout": 0.1,
    "epochs": 50,
    "early_stopping_patience": 15,
    "warmup_epochs": 5,
    "value_weight": 1.0,
    "policy_weight": 1.0,
}

# Cache for loaded config
_config_cache: Optional[Dict[str, Any]] = None


def _load_config() -> Dict[str, Any]:
    """Load hyperparameters config from file."""
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    if not CONFIG_PATH.exists():
        logger.warning(f"Hyperparameters config not found at {CONFIG_PATH}")
        return {"defaults": DEFAULT_HYPERPARAMETERS, "configs": {}}

    try:
        with open(CONFIG_PATH) as f:
            _config_cache = json.load(f)
            return _config_cache
    except Exception as e:
        logger.warning(f"Failed to load hyperparameters config: {e}")
        return {"defaults": DEFAULT_HYPERPARAMETERS, "configs": {}}


def reload_config() -> None:
    """Force reload of hyperparameters config."""
    global _config_cache
    _config_cache = None
    _load_config()


def get_config_key(board_type: str, num_players: int) -> str:
    """Get config key for a board/player combination."""
    return f"{board_type}_{num_players}p"


def get_hyperparameters(
    board_type: str,
    num_players: int,
    *,
    require_optimized: bool = False,
) -> Dict[str, Any]:
    """Get hyperparameters for a specific board/player configuration.

    Args:
        board_type: Board type (square8, square19, hexagonal)
        num_players: Number of players (2, 3, 4)
        require_optimized: If True, only return params if they've been optimized

    Returns:
        Dictionary of hyperparameters

    Raises:
        ValueError: If require_optimized=True and params not optimized
    """
    config = _load_config()
    defaults = config.get("defaults", DEFAULT_HYPERPARAMETERS)
    config_key = get_config_key(board_type, num_players)

    # Check if we have config-specific params
    config_entry = config.get("configs", {}).get(config_key, {})
    optimized = config_entry.get("optimized", False)
    config_params = config_entry.get("hyperparameters")

    if require_optimized and not optimized:
        raise ValueError(f"Hyperparameters not optimized for {config_key}")

    # Merge defaults with config-specific params
    params = dict(defaults)
    if config_params:
        params.update(config_params)

    return params


def get_hyperparameter_info(board_type: str, num_players: int) -> Dict[str, Any]:
    """Get hyperparameter metadata for a configuration.

    Returns info about tuning status, confidence, etc.
    """
    config = _load_config()
    config_key = get_config_key(board_type, num_players)
    config_entry = config.get("configs", {}).get(config_key, {})

    return {
        "config_key": config_key,
        "optimized": config_entry.get("optimized", False),
        "confidence": config_entry.get("confidence", "none"),
        "tuning_method": config_entry.get("tuning_method"),
        "last_tuned": config_entry.get("last_tuned"),
        "tuning_trials": config_entry.get("tuning_trials", 0),
        "notes": config_entry.get("notes"),
    }


def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter info for all configurations."""
    config = _load_config()
    return config.get("configs", {})


def is_optimized(board_type: str, num_players: int) -> bool:
    """Check if hyperparameters have been optimized for a config."""
    info = get_hyperparameter_info(board_type, num_players)
    return info.get("optimized", False)


def get_confidence(board_type: str, num_players: int) -> str:
    """Get confidence level for a configuration's hyperparameters.

    Returns one of: "high", "medium", "low", "none"
    """
    info = get_hyperparameter_info(board_type, num_players)
    return info.get("confidence", "none")


def needs_tuning(board_type: str, num_players: int, min_confidence: str = "medium") -> bool:
    """Check if a configuration needs hyperparameter tuning.

    Args:
        board_type: Board type
        num_players: Number of players
        min_confidence: Minimum acceptable confidence level

    Returns:
        True if tuning is needed
    """
    confidence_levels = {"none": 0, "low": 1, "medium": 2, "high": 3}
    current = get_confidence(board_type, num_players)
    return confidence_levels.get(current, 0) < confidence_levels.get(min_confidence, 2)


def get_training_command_args(board_type: str, num_players: int) -> list[str]:
    """Get command-line arguments for train_nnue.py from optimized params.

    Returns list of args like ["--learning-rate", "0.001", "--batch-size", "256", ...]
    """
    params = get_hyperparameters(board_type, num_players)
    args = []

    param_to_arg = {
        "learning_rate": "--learning-rate",
        "batch_size": "--batch-size",
        "hidden_dim": "--hidden-dim",
        "num_hidden_layers": "--num-hidden-layers",
        "weight_decay": "--weight-decay",
        "epochs": "--epochs",
        "early_stopping_patience": "--early-stopping-patience",
    }

    for param, arg in param_to_arg.items():
        if param in params:
            args.extend([arg, str(params[param])])

    return args
