"""Priority boost multiplier calculations for selfplay scheduling.

January 2026: Extracted from selfplay_scheduler.py as part of code quality
decomposition. These pure functions calculate various boost factors that
affect config priority in the selfplay allocation algorithm.

Usage:
    from app.coordination.selfplay.priority_boosts import (
        get_cascade_priority,
        get_improvement_boosts,
        get_momentum_multipliers,
        get_architecture_boosts,
    )

    # Each returns a boost factor or dict of factors
    cascade = get_cascade_priority("hex8_2p")
    improvement = get_improvement_boosts()  # Returns dict[str, float]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import config list from priority_calculator (canonical source)
from app.coordination.priority_calculator import ALL_CONFIGS
from app.coordination.budget_calculator import parse_config_key

__all__ = [
    "get_cascade_priority",
    "get_improvement_boosts",
    "get_momentum_multipliers",
    "get_architecture_boosts",
]


def get_cascade_priority(config_key: str) -> float:
    """Get cascade training priority boost for a config.

    Dec 29, 2025: Cascade training for multiplayer bootstrapping.
    Configs that are blocking cascade advancement (2p → 3p → 4p)
    get boosted priority to accelerate training.

    Args:
        config_key: Config key like "hex8_2p"

    Returns:
        Priority multiplier (1.0 = normal, >1.0 = boosted)
    """
    try:
        from app.coordination.cascade_training import get_cascade_orchestrator

        orchestrator = get_cascade_orchestrator()
        if orchestrator:
            return orchestrator.get_bootstrap_priority(config_key)
    except ImportError:
        logger.debug("[priority_boosts] cascade_training not available")
    except (AttributeError, RuntimeError) as e:
        logger.debug(f"[priority_boosts] Error getting cascade priority for {config_key}: {e}")

    return 1.0  # No boost by default


def get_improvement_boosts() -> dict[str, float]:
    """Get improvement boosts from ImprovementOptimizer per config.

    Phase 5 (Dec 2025): Connects selfplay scheduling to training success signals.
    When a config is on a promotion streak, boost its selfplay priority.

    Returns:
        Dict mapping config_key to boost value (-0.10 to +0.15)
    """
    result: dict[str, float] = {}

    try:
        from app.training.improvement_optimizer import get_selfplay_priority_boost

        for config_key in ALL_CONFIGS:
            boost = get_selfplay_priority_boost(config_key)
            if boost != 0.0:
                result[config_key] = boost
                logger.debug(
                    f"[priority_boosts] Improvement boost for {config_key}: {boost:+.2f}"
                )

    except ImportError:
        logger.debug("[priority_boosts] improvement_optimizer not available")
    except Exception as e:
        logger.debug(f"[priority_boosts] Error getting improvement boosts: {e}")

    return result


def get_momentum_multipliers() -> dict[str, float]:
    """Get momentum multipliers from FeedbackAccelerator per config.

    Phase 19 (Dec 2025): Connects selfplay scheduling to Elo momentum.
    This provides Elo momentum → Selfplay rate coupling:
    - ACCELERATING: 1.5x (capitalize on positive momentum)
    - IMPROVING: 1.25x (boost for continued improvement)
    - STABLE: 1.0x (normal rate)
    - PLATEAU: 1.1x (slight boost to try to break plateau)
    - REGRESSING: 0.75x (reduce noise, focus on quality)

    Returns:
        Dict mapping config_key to multiplier value (0.5 to 1.5)
    """
    result: dict[str, float] = {}

    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        accelerator = get_feedback_accelerator()

        for config_key in ALL_CONFIGS:
            multiplier = accelerator.get_selfplay_multiplier(config_key)
            if multiplier != 1.0:  # Only log non-default values
                logger.debug(
                    f"[priority_boosts] Momentum multiplier for {config_key}: {multiplier:.2f}x"
                )
            result[config_key] = multiplier

    except ImportError:
        logger.debug("[priority_boosts] feedback_accelerator not available")
    except Exception as e:
        logger.debug(f"[priority_boosts] Error getting momentum multipliers: {e}")

    return result


def get_architecture_boosts() -> dict[str, float]:
    """Get architecture-based boosts per config.

    Phase 5B (Dec 2025): Connects selfplay scheduling to architecture performance.
    Configs where the best architecture is performing well get boosted priority.
    This creates a feedback loop where successful architectures get more training data.

    Returns:
        Dict mapping config_key to boost value (0.0 to +0.30)
    """
    result: dict[str, float] = {}

    try:
        from app.training.architecture_tracker import get_allocation_weights

        for config_key in ALL_CONFIGS:
            # Parse config_key to get board_type and num_players
            board_type, num_players = parse_config_key(config_key)

            # Get allocation weights for this config
            weights = get_allocation_weights(board_type, num_players)

            if weights:
                # Boost = max weight * 0.3 (so a dominant architecture with weight 1.0 gives +0.30)
                max_weight = max(weights.values())
                boost = max_weight * 0.30

                if boost > 0.01:  # Only log significant boosts
                    result[config_key] = boost
                    logger.debug(
                        f"[priority_boosts] Architecture boost for {config_key}: +{boost:.2f} "
                        f"(best arch weight: {max_weight:.2f})"
                    )

    except ImportError:
        logger.debug("[priority_boosts] architecture_tracker not available")
    except Exception as e:
        logger.debug(f"[priority_boosts] Error getting architecture boosts: {e}")

    return result
