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
import time
from typing import Any

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

class MomentumToCurriculumBridge:
    """Bridges FeedbackAccelerator momentum changes to CurriculumFeedback weights.

    When FeedbackAccelerator detects ACCELERATING momentum for a config,
    this bridge pushes updated weights to CurriculumFeedback to increase
    training resources for that config.

    Event flow:
    1. FeedbackAccelerator.record_elo_update() updates momentum
    2. This bridge polls momentum state or subscribes to events
    3. CurriculumFeedback._current_weights updated
    4. CURRICULUM_REBALANCED event emitted
    """

    def __init__(
        self,
        poll_interval_seconds: float = 60.0,
        momentum_weight_boost: float = 0.3,
    ):
        self.poll_interval_seconds = poll_interval_seconds
        self.momentum_weight_boost = momentum_weight_boost

        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._last_weights: dict[str, float] = {}

    def start(self) -> None:
        """Start the momentum-to-curriculum bridge."""
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="MomentumCurriculumBridge",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info("[MomentumToCurriculumBridge] Started")

    def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
            self._poll_thread = None
        logger.info("[MomentumToCurriculumBridge] Stopped")

    def _poll_loop(self) -> None:
        """Poll FeedbackAccelerator and update CurriculumFeedback."""
        while self._running:
            try:
                self._sync_weights()
            except Exception as e:
                logger.warning(f"[MomentumToCurriculumBridge] Error syncing: {e}")

            time.sleep(self.poll_interval_seconds)

    def _sync_weights(self) -> None:
        """Sync weights from FeedbackAccelerator to CurriculumFeedback."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator
            from app.training.curriculum_feedback import get_curriculum_feedback

            accelerator = get_feedback_accelerator()
            feedback = get_curriculum_feedback()

            # Get momentum-based weights from accelerator
            accelerator_weights = accelerator.get_curriculum_weights()

            if not accelerator_weights:
                return

            # Check for significant changes
            changed_configs = []
            for config_key, new_weight in accelerator_weights.items():
                old_weight = self._last_weights.get(config_key, 1.0)
                if abs(new_weight - old_weight) > 0.1:
                    changed_configs.append(config_key)

            if not changed_configs:
                return

            # Update CurriculumFeedback weights
            for config_key, weight in accelerator_weights.items():
                feedback._current_weights[config_key] = weight

            self._last_weights = dict(accelerator_weights)

            # Emit event
            self._emit_rebalance_event(changed_configs, accelerator_weights)

            logger.info(
                f"[MomentumToCurriculumBridge] Synced {len(changed_configs)} weight changes: "
                f"{', '.join(changed_configs)}"
            )

        except ImportError as e:
            logger.debug(f"[MomentumToCurriculumBridge] Import error: {e}")

    def _emit_rebalance_event(
        self,
        changed_configs: list[str],
        weights: dict[str, float],
    ) -> None:
        """Emit CURRICULUM_REBALANCED event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "momentum_sync",
                    "changed_configs": changed_configs,
                    "new_weights": weights,
                    "timestamp": time.time(),
                },
                source="momentum_curriculum_bridge",
            )
        except Exception as e:
            logger.debug(f"Failed to emit rebalance event: {e}")

    def force_sync(self) -> dict[str, float]:
        """Force immediate weight sync."""
        self._sync_weights()
        return self._last_weights


# =============================================================================
# 2. PFSP Weak Opponent Detection → CurriculumFeedback
# =============================================================================

class PFSPWeaknessWatcher:
    """Watches PFSP for weak opponent detection and adjusts curriculum.

    When PFSP detects that a model consistently beats an opponent (>85% win rate),
    this indicates the matchup is too easy and resources should be reallocated.

    Event flow:
    1. PFSP.record_game_result() updates win rates
    2. This watcher checks for win_rate > threshold
    3. Emits OPPONENT_MASTERED event
    4. CurriculumFeedback reduces weight for that config
    """

    # Thresholds
    MASTERY_THRESHOLD = 0.85  # Win rate above this = opponent mastered
    MIN_GAMES_FOR_MASTERY = 20  # Minimum games to declare mastery
    CHECK_INTERVAL = 120.0  # Seconds between checks

    def __init__(self):
        self._running = False
        self._check_thread: threading.Thread | None = None
        self._mastered_matchups: set[tuple[str, str]] = set()

    def start(self) -> None:
        """Start the weakness watcher."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(
            target=self._check_loop,
            name="PFSPWeaknessWatcher",
            daemon=True,
        )
        self._check_thread.start()
        logger.info("[PFSPWeaknessWatcher] Started")

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
            self._check_thread = None
        logger.info("[PFSPWeaknessWatcher] Stopped")

    def _check_loop(self) -> None:
        """Periodically check for mastered opponents."""
        while self._running:
            try:
                self._check_for_mastery()
            except Exception as e:
                logger.warning(f"[PFSPWeaknessWatcher] Error checking: {e}")

            time.sleep(self.CHECK_INTERVAL)

    def _check_for_mastery(self) -> None:
        """Check all matchups for mastery conditions."""
        try:
            from app.training.pfsp_opponent_selector import get_pfsp_selector

            selector = get_pfsp_selector()

            new_masteries = []

            # Check each model's matchups
            for (current_model, opponent), record in selector._matchups.items():
                # Skip if already detected
                if (current_model, opponent) in self._mastered_matchups:
                    continue

                # Check mastery conditions
                if record.total_games >= self.MIN_GAMES_FOR_MASTERY:
                    if record.win_rate >= self.MASTERY_THRESHOLD:
                        self._mastered_matchups.add((current_model, opponent))
                        new_masteries.append({
                            "current_model": current_model,
                            "opponent": opponent,
                            "win_rate": record.win_rate,
                            "games": record.total_games,
                        })

            # Process new masteries
            for mastery in new_masteries:
                self._on_opponent_mastered(mastery)

        except ImportError:
            pass  # PFSP not available

    def _on_opponent_mastered(self, mastery: dict[str, Any]) -> None:
        """Handle detection of a mastered opponent."""
        current_model = mastery["current_model"]
        opponent = mastery["opponent"]
        win_rate = mastery["win_rate"]

        # Extract config from model ID (convention: {config}_v{version})
        config_key = self._extract_config(current_model)

        logger.info(
            f"[PFSPWeaknessWatcher] Opponent mastered: {current_model} vs {opponent} "
            f"({win_rate:.1%} win rate) - config: {config_key}"
        )

        # Emit event
        self._emit_opponent_mastered(config_key, mastery)

        # Update curriculum feedback
        self._update_curriculum_weight(config_key)

    def _extract_config(self, model_id: str) -> str:
        """Extract config key from model ID."""
        # Convention: hex8_2p_v123 -> hex8_2p
        # Or: canonical_hex8_2p -> hex8_2p
        parts = model_id.split("_")
        if len(parts) >= 2:
            # Try to find board_Np pattern
            for i, part in enumerate(parts):
                if part.endswith("p") and part[:-1].isdigit():
                    # Found player count, return board_Np
                    return "_".join(parts[max(0, i - 1):i + 1])

            # Fallback: first two parts
            return "_".join(parts[:2])
        return model_id

    def _emit_opponent_mastered(self, config_key: str, mastery: dict[str, Any]) -> None:
        """Emit OPPONENT_MASTERED event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "OPPONENT_MASTERED",
                {
                    "config": config_key,
                    "current_model": mastery["current_model"],
                    "opponent": mastery["opponent"],
                    "win_rate": mastery["win_rate"],
                    "games": mastery["games"],
                    "timestamp": time.time(),
                },
                source="pfsp_weakness_watcher",
            )
        except Exception as e:
            logger.debug(f"Failed to emit opponent mastered event: {e}")

    def _update_curriculum_weight(self, config_key: str) -> None:
        """Reduce curriculum weight for mastered config.

        When we're dominating an opponent type, we may not need as much
        training focus on that config.
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()

            current_weight = feedback._current_weights.get(config_key, 1.0)

            # Reduce weight slightly (opponent too easy = less learning value)
            # But don't reduce too much - we still need diversity
            new_weight = max(feedback.weight_min, current_weight * 0.9)

            if new_weight < current_weight:
                feedback._current_weights[config_key] = new_weight
                logger.info(
                    f"[PFSPWeaknessWatcher] Reduced curriculum weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f}"
                )

        except ImportError:
            pass

    def get_mastered_matchups(self) -> list[tuple[str, str]]:
        """Get list of mastered matchups."""
        return list(self._mastered_matchups)


# =============================================================================
# 3. Quality Scores → Temperature Scheduling
# =============================================================================

class QualityToTemperatureWatcher:
    """Adjusts exploration temperature based on training data quality.

    Low quality data indicates the model may be stuck in a local minimum.
    This watcher increases exploration temperature to generate more diverse
    training data.

    Event flow:
    1. QualityFeedbackWatcher detects low quality
    2. This watcher receives QUALITY_FEEDBACK_ADJUSTED event
    3. Updates temperature schedule exploration_boost
    """

    # Thresholds
    LOW_QUALITY_THRESHOLD = 0.4
    EXPLORATION_BOOST_FACTOR = 1.3  # Increase temperature by 30% on low quality

    def __init__(self):
        self._subscribed = False
        self._quality_boosts: dict[str, float] = {}  # config -> boost factor

    def subscribe(self) -> bool:
        """Subscribe to quality events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.subscribe("QUALITY_FEEDBACK_ADJUSTED", self._on_quality_adjusted)
            router.subscribe("QUALITY_SCORE_UPDATED", self._on_quality_updated)
            self._subscribed = True
            logger.info("[QualityToTemperatureWatcher] Subscribed to quality events")
            return True
        except Exception as e:
            logger.warning(f"[QualityToTemperatureWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.unsubscribe("QUALITY_FEEDBACK_ADJUSTED", self._on_quality_adjusted)
            router.unsubscribe("QUALITY_SCORE_UPDATED", self._on_quality_updated)
            self._subscribed = False
        except Exception:
            pass

    def _on_quality_adjusted(self, event: Any) -> None:
        """Handle QUALITY_FEEDBACK_ADJUSTED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config_key", "")
        avg_quality = payload.get("avg_quality", 0.5)

        if not config_key:
            return

        self._update_exploration_boost(config_key, avg_quality)

    def _on_quality_updated(self, event: Any) -> None:
        """Handle QUALITY_SCORE_UPDATED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config") or payload.get("config_key", "")
        quality = payload.get("quality_score", payload.get("new_score", 0.5))

        if not config_key:
            return

        self._update_exploration_boost(config_key, quality)

    def _update_exploration_boost(self, config_key: str, quality: float) -> None:
        """Update exploration boost based on quality."""
        old_boost = self._quality_boosts.get(config_key, 1.0)

        if quality < self.LOW_QUALITY_THRESHOLD:
            # Low quality = boost exploration
            new_boost = self.EXPLORATION_BOOST_FACTOR
        else:
            # Normal/high quality = normal exploration
            new_boost = 1.0

        if abs(new_boost - old_boost) > 0.05:
            self._quality_boosts[config_key] = new_boost
            self._apply_temperature_boost(config_key, new_boost)

            logger.info(
                f"[QualityToTemperatureWatcher] {config_key} quality={quality:.2f}, "
                f"exploration boost: {old_boost:.2f} → {new_boost:.2f}"
            )

    def _apply_temperature_boost(self, config_key: str, boost: float) -> None:
        """Apply exploration boost to temperature scheduler."""
        try:
            from app.training.temperature_scheduling import get_active_schedulers

            schedulers = get_active_schedulers()
            scheduler = schedulers.get(config_key)
            if scheduler and hasattr(scheduler, 'set_exploration_boost'):
                scheduler.set_exploration_boost(boost)
                logger.debug(f"Applied exploration boost {boost:.2f} to {config_key} scheduler")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to apply temperature boost: {e}")

        # Also emit event for downstream systems
        self._emit_exploration_boost(config_key, boost)

    def _emit_exploration_boost(self, config_key: str, boost: float) -> None:
        """Emit EXPLORATION_BOOST event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "EXPLORATION_BOOST",
                {
                    "config": config_key,
                    "boost_factor": boost,
                    "reason": "low_quality_data",
                    "timestamp": time.time(),
                },
                source="quality_temperature_watcher",
            )
        except Exception:
            pass

    def get_exploration_boost(self, config_key: str) -> float:
        """Get the current exploration boost for a config."""
        return self._quality_boosts.get(config_key, 1.0)

    def get_all_boosts(self) -> dict[str, float]:
        """Get all current exploration boosts."""
        return dict(self._quality_boosts)


# =============================================================================
# Unified Wiring
# =============================================================================


def wire_all_feedback_loops(
    enable_momentum_bridge: bool = True,
    enable_pfsp_weakness: bool = True,
    enable_quality_temperature: bool = True,
    enable_curriculum_feedback: bool = True,
) -> dict[str, Any]:
    """Wire all feedback loop connections at once.

    This is the main entry point for connecting all feedback systems.
    Call this at startup to enable the full self-improvement loop.

    Args:
        enable_momentum_bridge: Enable FeedbackAccelerator → CurriculumFeedback
        enable_pfsp_weakness: Enable PFSP weak opponent → CurriculumFeedback
        enable_quality_temperature: Enable Quality → Temperature
        enable_curriculum_feedback: Enable all curriculum_feedback.py watchers

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
            except Exception as e:
                status["momentum_bridge_error"] = str(e)
                logger.warning(f"Failed to start momentum bridge: {e}")

        # 2. PFSP Weakness watcher
        if enable_pfsp_weakness:
            try:
                watcher = PFSPWeaknessWatcher()
                watcher.start()
                _watcher_instances["pfsp_weakness"] = watcher
                status["watchers"].append("pfsp_weakness")
            except Exception as e:
                status["pfsp_weakness_error"] = str(e)
                logger.warning(f"Failed to start PFSP weakness watcher: {e}")

        # 3. Quality → Temperature watcher
        if enable_quality_temperature:
            try:
                watcher = QualityToTemperatureWatcher()
                watcher.subscribe()
                _watcher_instances["quality_temperature"] = watcher
                status["watchers"].append("quality_temperature")
            except Exception as e:
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
            except Exception as e:
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
            except Exception as e:
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


__all__ = [
    # Main wiring functions
    "wire_all_feedback_loops",
    "unwire_all_feedback_loops",
    "get_integration_status",
    # Individual components
    "MomentumToCurriculumBridge",
    "PFSPWeaknessWatcher",
    "QualityToTemperatureWatcher",
    # Convenience functions
    "get_exploration_boost",
    "get_mastered_opponents",
    "force_momentum_sync",
]
