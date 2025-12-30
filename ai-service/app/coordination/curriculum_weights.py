"""Curriculum weight management for selfplay prioritization.

This module provides functions to persist and load curriculum weights
that are used to prioritize selfplay jobs across the cluster.

Weights are stored in a JSON file and read by:
- SelfplayScheduler (for priority-based allocation)
- QueuePopulator (for work queue population)
- P2P Orchestrator (for distributed coordination)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "CURRICULUM_WEIGHTS_PATH",
    "CURRICULUM_WEIGHTS_STALE_SECONDS",
    "export_curriculum_weights",
    "get_curriculum_weight",
    "load_curriculum_weights",
]

# Path for curriculum weights shared across components
CURRICULUM_WEIGHTS_PATH = AI_SERVICE_ROOT / "data" / "curriculum_weights.json"

# Staleness threshold (weights older than this are ignored)
CURRICULUM_WEIGHTS_STALE_SECONDS = 7200  # 2 hours


def export_curriculum_weights(weights: dict[str, float]) -> bool:
    """Export curriculum weights to JSON file.

    Used by AdaptiveCurriculum to publish weights for other components.

    Args:
        weights: Dictionary of config_key -> weight multiplier

    Returns:
        True if export succeeded, False otherwise
    """
    try:
        CURRICULUM_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": weights,
            "updated_at": time.time(),
            "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Atomic write: write to temp file then rename
        temp_path = CURRICULUM_WEIGHTS_PATH.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(CURRICULUM_WEIGHTS_PATH)
        return True
    except Exception as e:
        logger.error(f"Failed to export curriculum weights: {e}")
        return False


def load_curriculum_weights(max_age_seconds: float = CURRICULUM_WEIGHTS_STALE_SECONDS) -> dict[str, float]:
    """Load curriculum weights from JSON file.

    Used by SelfplayScheduler, QueuePopulator, and P2P to read curriculum priorities.

    Args:
        max_age_seconds: Maximum age in seconds before weights are considered stale

    Returns:
        Dictionary of config_key -> weight multiplier, or empty dict if unavailable/stale
    """
    try:
        if not CURRICULUM_WEIGHTS_PATH.exists():
            return {}
        with open(CURRICULUM_WEIGHTS_PATH) as f:
            data = json.load(f)
        # Handle null, array, or other non-dict JSON
        if not isinstance(data, dict):
            return {}
        # Check staleness - handle non-numeric updated_at
        updated_at = data.get("updated_at", 0)
        if not isinstance(updated_at, (int, float)):
            return {}
        if time.time() - updated_at > max_age_seconds:
            return {}  # Stale weights
        return data.get("weights", {})
    except (FileNotFoundError, OSError, PermissionError, json.JSONDecodeError, TypeError):
        return {}


def get_curriculum_weight(config_key: str, default: float = 1.0) -> float:
    """Get curriculum weight for a specific config.

    Convenience function for getting a single config's weight.

    Args:
        config_key: Config key like "hex8_2p" or "square8_4p"
        default: Default weight if not found

    Returns:
        Weight multiplier for the config
    """
    weights = load_curriculum_weights()
    return weights.get(config_key, default)
