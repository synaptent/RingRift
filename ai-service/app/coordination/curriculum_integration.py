"""Curriculum Integration - Bridges feedback loops for AI self-improvement.

This module provides the missing connections between feedback systems:
1. FeedbackAccelerator momentum → CurriculumFeedback weights
2. PFSP weak opponent detection → CurriculumFeedback (reduce weight for mastered configs)
3. Quality scores → Temperature scheduling (increase exploration on low quality)

Usage:
    from app.coordination.curriculum_integration import (
        wire_all_feedback_loops,
        get_integration_status,
    )

    # Wire all feedback loop connections at startup
    status = wire_all_feedback_loops()

    # Check integration health
    print(get_integration_status())

Created: December 2025
Purpose: Close missing feedback loops in AI training self-improvement cycle
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.coordination.curriculum_momentum_bridge import MomentumToCurriculumBridge
from app.coordination.curriculum_strategies import (
    ArchitectureToCurriculumBridge,
    PFSPWeaknessWatcher,
    PromotionCompletedToCurriculumWatcher,
    PromotionFailedToCurriculumWatcher,
    QualityPenaltyToCurriculumWatcher,
    QualityToTemperatureWatcher,
    RegressionCriticalToCurriculumWatcher,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Integration State
# =============================================================================

_integration_active = False
_integration_lock = threading.Lock()
_watcher_instances: dict[str, Any] = {}


# =============================================================================
# 1. FeedbackAccelerator Momentum → CurriculumFeedback Weights
# =============================================================================



# =============================================================================
# 2. PFSP Weak Opponent Detection → CurriculumFeedback
# =============================================================================



# =============================================================================
# 2.5. QUALITY_PENALTY_APPLIED → Curriculum Weight Reduction
# =============================================================================




# =============================================================================
# 2.5.1. PROMOTION_COMPLETED → Curriculum Advancement/Regression (December 29, 2025)
# =============================================================================




# =============================================================================
# 2.4.1. REGRESSION_CRITICAL → Curriculum Weight Boost (December 27, 2025)
# =============================================================================




# =============================================================================
# 2.5. QUALITY_PENALTY_APPLIED → Curriculum Weight Reduction
# =============================================================================




# =============================================================================
# 2.8. ARCHITECTURE_WEIGHTS_UPDATED → Curriculum Weight Boost for Underperformers
# =============================================================================




# =============================================================================
# 3. Quality Scores → Temperature Scheduling
# =============================================================================



# =============================================================================
# Unified Wiring
# =============================================================================


def wire_all_feedback_loops(
    enable_momentum_bridge: bool = True,
    enable_pfsp_weakness: bool = True,
    enable_promotion_failed: bool = True,
    enable_promotion_completed: bool = True,
    enable_quality_penalty: bool = True,
    enable_regression_critical: bool = True,
    enable_quality_temperature: bool = True,
    enable_curriculum_feedback: bool = True,
    enable_architecture_curriculum: bool = True,
) -> dict[str, Any]:
    """Wire all feedback loop connections at once.

    This is the main entry point for connecting all feedback systems.
    Call this at startup to enable the full self-improvement loop.

    Args:
        enable_momentum_bridge: Enable FeedbackAccelerator → CurriculumFeedback
        enable_pfsp_weakness: Enable PFSP weak opponent → CurriculumFeedback
        enable_promotion_failed: Enable PROMOTION_FAILED → CurriculumFeedback
        enable_promotion_completed: Enable PROMOTION_COMPLETED → CurriculumFeedback (Dec 29, 2025)
        enable_quality_penalty: Enable QUALITY_PENALTY_APPLIED → CurriculumFeedback
        enable_regression_critical: Enable REGRESSION_CRITICAL → CurriculumFeedback
        enable_quality_temperature: Enable Quality → Temperature
        enable_curriculum_feedback: Enable all curriculum_feedback.py watchers
        enable_architecture_curriculum: Enable ARCHITECTURE_WEIGHTS_UPDATED → CurriculumFeedback (Jan 2026)

    Returns:
        Dict with status of each integration
    """
    global _integration_active, _watcher_instances

    with _integration_lock:
        if _integration_active:
            return {"status": "already_active", "watchers": list(_watcher_instances.keys())}

        status: dict[str, Any] = {"watchers": []}

        # 1. Momentum → Curriculum bridge
        if enable_momentum_bridge:
            try:
                bridge = MomentumToCurriculumBridge()
                bridge.start()
                _watcher_instances["momentum_bridge"] = bridge
                status["watchers"].append("momentum_bridge")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: feedback modules not available
                # AttributeError: bridge method missing
                # TypeError: invalid configuration
                # RuntimeError: bridge start failed
                status["momentum_bridge_error"] = str(e)
                logger.warning(f"Failed to start momentum bridge: {e}")

        # 2. PFSP Weakness watcher
        if enable_pfsp_weakness:
            try:
                watcher = PFSPWeaknessWatcher()
                watcher.start()
                _watcher_instances["pfsp_weakness"] = watcher
                status["watchers"].append("pfsp_weakness")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: pfsp modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher start failed
                status["pfsp_weakness_error"] = str(e)
                logger.warning(f"Failed to start PFSP weakness watcher: {e}")

        # 2.5. Promotion Failed → Curriculum Weight watcher (December 2025)
        if enable_promotion_failed:
            try:
                watcher = PromotionFailedToCurriculumWatcher()
                watcher.subscribe()
                _watcher_instances["promotion_failed_curriculum"] = watcher
                status["watchers"].append("promotion_failed_curriculum")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: event modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher subscribe failed
                status["promotion_failed_curriculum_error"] = str(e)
                logger.warning(f"Failed to start promotion failed curriculum watcher: {e}")

        # 2.5.1. Promotion Completed → Curriculum Advancement/Regression (December 29, 2025)
        if enable_promotion_completed:
            try:
                watcher = PromotionCompletedToCurriculumWatcher()
                watcher.subscribe()
                _watcher_instances["promotion_completed_curriculum"] = watcher
                status["watchers"].append("promotion_completed_curriculum")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: event modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher subscribe failed
                status["promotion_completed_curriculum_error"] = str(e)
                logger.warning(f"Failed to start promotion completed curriculum watcher: {e}")

        # 2.6. Quality Penalty → Curriculum Weight watcher (December 2025)
        if enable_quality_penalty:
            try:
                watcher = QualityPenaltyToCurriculumWatcher()
                watcher.subscribe()
                _watcher_instances["quality_penalty_curriculum"] = watcher
                status["watchers"].append("quality_penalty_curriculum")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: quality modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher subscribe failed
                status["quality_penalty_curriculum_error"] = str(e)
                logger.warning(f"Failed to start quality penalty curriculum watcher: {e}")

        # 2.7. Regression Critical → Curriculum Weight watcher (December 2025)
        if enable_regression_critical:
            try:
                watcher = RegressionCriticalToCurriculumWatcher()
                watcher.subscribe()
                _watcher_instances["regression_critical_curriculum"] = watcher
                status["watchers"].append("regression_critical_curriculum")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: event modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher subscribe failed
                status["regression_critical_curriculum_error"] = str(e)
                logger.warning(f"Failed to start regression critical curriculum watcher: {e}")

        # 2.8. Architecture → Curriculum Weight watcher (January 2026)
        if enable_architecture_curriculum:
            try:
                watcher = ArchitectureToCurriculumBridge()
                watcher.subscribe()
                _watcher_instances["architecture_curriculum"] = watcher
                status["watchers"].append("architecture_curriculum")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: architecture modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher subscribe failed
                status["architecture_curriculum_error"] = str(e)
                logger.warning(f"Failed to start architecture curriculum watcher: {e}")

        # 3. Quality → Temperature watcher
        if enable_quality_temperature:
            try:
                watcher = QualityToTemperatureWatcher()
                watcher.subscribe()
                _watcher_instances["quality_temperature"] = watcher
                status["watchers"].append("quality_temperature")
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: quality modules not available
                # AttributeError: watcher method missing
                # TypeError: invalid configuration
                # RuntimeError: watcher subscribe failed
                status["quality_temperature_error"] = str(e)
                logger.warning(f"Failed to start quality temperature watcher: {e}")

        # 4. All curriculum_feedback.py watchers
        if enable_curriculum_feedback:
            try:
                from app.training.curriculum_feedback import wire_all_curriculum_feedback
                curriculum_watchers = wire_all_curriculum_feedback()
                _watcher_instances["curriculum_feedback"] = curriculum_watchers
                status["watchers"].append("curriculum_feedback")
                status["curriculum_watchers"] = list(curriculum_watchers.keys())
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: curriculum_feedback module not available
                # AttributeError: wire function missing
                # TypeError: invalid configuration
                # RuntimeError: wiring failed
                status["curriculum_feedback_error"] = str(e)
                logger.warning(f"Failed to wire curriculum feedback: {e}")

        _integration_active = True
        status["status"] = "active"
        status["active_count"] = len(status["watchers"])

        logger.info(
            f"[wire_all_feedback_loops] Wired {status['active_count']} feedback integrations: "
            f"{', '.join(status['watchers'])}"
        )

        return status


def unwire_all_feedback_loops() -> None:
    """Stop all feedback loop connections."""
    global _integration_active, _watcher_instances

    with _integration_lock:
        for name, watcher in list(_watcher_instances.items()):
            try:
                if hasattr(watcher, 'stop'):
                    watcher.stop()
                elif hasattr(watcher, 'unsubscribe'):
                    watcher.unsubscribe()
            except (AttributeError, TypeError, RuntimeError) as e:
                # AttributeError: method missing
                # TypeError: invalid stop arguments
                # RuntimeError: stop operation failed
                logger.warning(f"Error stopping {name}: {e}")

        _watcher_instances.clear()
        _integration_active = False

        logger.info("[unwire_all_feedback_loops] All feedback loops stopped")


def get_integration_status() -> dict[str, Any]:
    """Get status of all feedback loop integrations.

    Returns:
        Dict with integration health status
    """
    with _integration_lock:
        status = {
            "active": _integration_active,
            "watchers": list(_watcher_instances.keys()),
        }

        # Get detailed status from each watcher
        for name, watcher in _watcher_instances.items():
            if hasattr(watcher, 'get_statistics'):
                status[f"{name}_stats"] = watcher.get_statistics()
            elif hasattr(watcher, 'get_mastered_matchups'):
                status[f"{name}_mastered"] = len(watcher.get_mastered_matchups())
            elif hasattr(watcher, 'get_all_boosts'):
                status[f"{name}_boosts"] = watcher.get_all_boosts()

        return status


# =============================================================================
# Convenience Functions
# =============================================================================


def get_exploration_boost(config_key: str) -> float:
    """Get exploration boost for a config (from quality watcher).

    Args:
        config_key: Config identifier

    Returns:
        Exploration boost factor (1.0 = normal)
    """
    watcher = _watcher_instances.get("quality_temperature")
    if watcher and isinstance(watcher, QualityToTemperatureWatcher):
        return watcher.get_exploration_boost(config_key)
    return 1.0


def get_mastered_opponents() -> list[tuple[str, str]]:
    """Get list of mastered (current_model, opponent) matchups."""
    watcher = _watcher_instances.get("pfsp_weakness")
    if watcher and isinstance(watcher, PFSPWeaknessWatcher):
        return watcher.get_mastered_matchups()
    return []


def force_momentum_sync() -> dict[str, float]:
    """Force immediate sync of momentum-based curriculum weights."""
    bridge = _watcher_instances.get("momentum_bridge")
    if bridge and isinstance(bridge, MomentumToCurriculumBridge):
        return bridge.force_sync()
    return {}


def get_quality_penalty_weights() -> dict[str, float]:
    """Get current quality penalty-based weight factors.

    Returns:
        Dict mapping config_key to weight factor (1.0 = no penalty, <1.0 = penalized)
    """
    watcher = _watcher_instances.get("quality_penalty_curriculum")
    if watcher and isinstance(watcher, QualityPenaltyToCurriculumWatcher):
        return watcher.get_penalty_weights()
    return {}


def reset_quality_penalty(config_key: str) -> None:
    """Reset quality penalty for a config (when quality recovers)."""
    watcher = _watcher_instances.get("quality_penalty_curriculum")
    if watcher and isinstance(watcher, QualityPenaltyToCurriculumWatcher):
        watcher.reset_penalty(config_key)


def get_promotion_failure_counts() -> dict[str, int]:
    """Get current promotion failure counts.

    Returns:
        Dict mapping config_key to consecutive failure count
    """
    watcher = _watcher_instances.get("promotion_failed_curriculum")
    if watcher and isinstance(watcher, PromotionFailedToCurriculumWatcher):
        return watcher.get_failure_counts()
    return {}


def reset_promotion_failure_count(config_key: str) -> None:
    """Reset promotion failure count for a config (when promotion succeeds)."""
    watcher = _watcher_instances.get("promotion_failed_curriculum")
    if watcher and isinstance(watcher, PromotionFailedToCurriculumWatcher):
        watcher.reset_failure_count(config_key)


def get_promotion_success_streaks() -> dict[str, int]:
    """Get current promotion success streaks.

    December 29, 2025: Added for Phase 4 training loop improvements.

    Returns:
        Dict mapping config_key to consecutive success count
    """
    watcher = _watcher_instances.get("promotion_completed_curriculum")
    if watcher and isinstance(watcher, PromotionCompletedToCurriculumWatcher):
        return watcher.get_success_streaks()
    return {}


def get_regression_critical_counts() -> dict[str, int]:
    """Get current regression critical counts.

    Returns:
        Dict mapping config_key to consecutive regression count
    """
    watcher = _watcher_instances.get("regression_critical_curriculum")
    if watcher and isinstance(watcher, RegressionCriticalToCurriculumWatcher):
        return watcher.get_regression_counts()
    return {}


def reset_regression_critical_count(config_key: str) -> None:
    """Reset regression critical count for a config (when model recovers)."""
    watcher = _watcher_instances.get("regression_critical_curriculum")
    if watcher and isinstance(watcher, RegressionCriticalToCurriculumWatcher):
        watcher.reset_regression_count(config_key)


def get_architecture_status() -> dict[str, Any]:
    """Get architecture performance status across all configs.

    Returns:
        Dict mapping config_key to architecture state including:
        - underperformers: list of underperforming architecture names
        - disparity: weight disparity ratio (max/min)
        - last_weights: dict of architecture weights
    """
    watcher = _watcher_instances.get("architecture_curriculum")
    if watcher and isinstance(watcher, ArchitectureToCurriculumBridge):
        return watcher.get_architecture_status()
    return {}


def get_underperforming_configs() -> list[str]:
    """Get configs that currently have underperforming architectures.

    Returns:
        List of config keys with underperforming architectures
    """
    watcher = _watcher_instances.get("architecture_curriculum")
    if watcher and isinstance(watcher, ArchitectureToCurriculumBridge):
        return watcher.get_underperforming_configs()
    return []


__all__ = [
    # Main wiring functions
    "wire_all_feedback_loops",
    "unwire_all_feedback_loops",
    "get_integration_status",
    # Individual components
    "MomentumToCurriculumBridge",
    "PFSPWeaknessWatcher",
    "PromotionFailedToCurriculumWatcher",
    "PromotionCompletedToCurriculumWatcher",
    "QualityPenaltyToCurriculumWatcher",
    "RegressionCriticalToCurriculumWatcher",
    "QualityToTemperatureWatcher",
    "ArchitectureToCurriculumBridge",
    # Convenience functions
    "get_exploration_boost",
    "get_mastered_opponents",
    "force_momentum_sync",
    "get_quality_penalty_weights",
    "reset_quality_penalty",
    "get_promotion_failure_counts",
    "reset_promotion_failure_count",
    "get_promotion_success_streaks",
    "get_regression_critical_counts",
    "reset_regression_critical_count",
    "get_architecture_status",
    "get_underperforming_configs",
]
