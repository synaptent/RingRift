"""Curriculum watcher and weighting strategies extracted from curriculum_integration."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from app.coordination.curriculum_router import CurriculumSignalBridge
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import parse_config_key
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


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

    # Thresholds - imported from centralized defaults (December 28, 2025)
    # Can be overridden via environment variables:
    # - RINGRIFT_MASTERY_THRESHOLD (default: 0.85)
    # - RINGRIFT_CURRICULUM_CHECK_INTERVAL (default: 120.0)
    # - RINGRIFT_MIN_GAMES_FOR_UPDATE (default: 100, MIN_GAMES_FOR_MASTERY uses 20)
    try:
        from app.config.coordination_defaults import CurriculumDefaults
        MASTERY_THRESHOLD = CurriculumDefaults.MASTERY_THRESHOLD
        CHECK_INTERVAL = CurriculumDefaults.CHECK_INTERVAL
    except ImportError:
        # Fallback for standalone testing
        MASTERY_THRESHOLD = 0.85
        CHECK_INTERVAL = 120.0
    MIN_GAMES_FOR_MASTERY = 20  # Minimum games to declare mastery (not centralized)

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
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: pfsp module not available
                # AttributeError: selector method missing
                # TypeError: invalid data types
                # RuntimeError: check operation failed
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
        """Extract config key from model ID using canonical utility."""
        # Convention: hex8_2p_v123 -> hex8_2p
        # Or: canonical_hex8_2p -> hex8_2p
        # Strip common prefixes first
        name = model_id
        for prefix in ("canonical_", "ringrift_best_", "selfplay_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        # Strip version suffixes like _v123, _v1, _v2.0
        name = re.sub(r"_v\d+(?:\.\d+)?$", "", name)
        parsed = parse_config_key(name)
        if parsed:
            return f"{parsed.board_type}_{parsed.num_players}p"
        return model_id

    def _emit_opponent_mastered(self, config_key: str, mastery: dict[str, Any]) -> None:
        """Emit OPPONENT_MASTERED event."""
        try:
            from app.coordination.event_router import DataEventType, publish_sync

            # P1.4 Dec 2025: Use DataEventType enum for type-safe emission
            publish_sync(
                DataEventType.OPPONENT_MASTERED,
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
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid event arguments
            # RuntimeError: publish failed
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

    def health_check(self) -> "HealthCheckResult":
        """Check watcher health for DaemonManager integration.

        December 2025: Added for unified health monitoring.
        """
        from app.coordination.protocols import HealthCheckResult

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="PFSPWeaknessWatcher not running",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Tracking {len(self._mastered_matchups)} mastered matchups",
            details={
                "mastered_count": len(self._mastered_matchups),
                "thread_alive": self._check_thread.is_alive() if self._check_thread else False,
            },
        )

class PromotionFailedToCurriculumWatcher(CurriculumSignalBridge):
    """Increases curriculum weight when model promotion fails.

    When a model fails promotion (emits PROMOTION_FAILED), this watcher
    increases that config's curriculum weight to generate more diverse
    training data for the next training cycle.

    Event flow (December 2025):
    1. Promotion process fails (validation, gauntlet, etc.)
    2. Emits PROMOTION_FAILED with config_key and error details
    3. This watcher subscribes and increases curriculum weight
    4. CurriculumFeedback allocates more selfplay to affected configs
    5. Emits CURRICULUM_REBALANCED to notify downstream systems

    December 30, 2025: Migrated to use CurriculumSignalBridge base class (P4.2).
    Reduces ~170 LOC of boilerplate to ~50 LOC of specific logic.
    """

    WATCHER_NAME = "promotion_failed_curriculum_watcher"
    EVENT_TYPES = ["PROMOTION_FAILED"]  # From RingRiftEventType

    # Weight increase factor per consecutive failure (cumulative)
    WEIGHT_INCREASE_PER_FAILURE = 0.20  # 20% increase per failure
    MAX_WEIGHT_MULTIPLIER = 2.5

    def _compute_weight_multiplier(
        self,
        config_key: str,
        payload: dict[str, Any],
    ) -> float | None:
        """Compute weight multiplier based on consecutive failures.

        Returns:
            Weight multiplier (1.2 for first failure, increasing by 0.2 per failure)
            Capped at 2.5x maximum.
        """
        # Track consecutive failures in state
        failure_key = f"{config_key}:failure_count"
        failure_count = self.get_state(failure_key, 0) + 1
        self.set_state(failure_key, failure_count)

        # Increase weight: 20% per failure, up to 2.5x max
        # failure_count=1 -> 1.2x, failure_count=2 -> 1.4x, etc.
        multiplier = min(
            self.MAX_WEIGHT_MULTIPLIER,
            1.0 + (failure_count * self.WEIGHT_INCREASE_PER_FAILURE),
        )
        return multiplier

    def _extract_event_details(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract promotion failure details for logging and events."""
        config_key = extract_config_key(payload)
        failure_key = f"{config_key}:failure_count" if config_key else ""
        return {
            "error": payload.get("error", "unknown"),
            "model_id": payload.get("model_id", ""),
            "failure_count": self.get_state(failure_key, 0) if failure_key else 0,
        }

    def reset_failure_count(self, config_key: str) -> None:
        """Reset failure count for a config (called when promotion succeeds)."""
        failure_key = f"{config_key}:failure_count"
        if self.get_state(failure_key) is not None:
            self.reset_state(config_key)
            logger.info(f"[{self.WATCHER_NAME}] Reset failure count for {config_key}")

    def get_failure_counts(self) -> dict[str, int]:
        """Get current failure counts."""
        result = {}
        for key, value in self._state.items():
            if key.endswith(":failure_count"):
                config_key = key.rsplit(":", 1)[0]
                result[config_key] = value
        return result

class PromotionCompletedToCurriculumWatcher:
    """Advances or regresses curriculum based on unified PROMOTION_COMPLETED events.

    This watcher subscribes to the unified PROMOTION_COMPLETED event emitted by
    AutoPromotionDaemon after both successful and failed promotion attempts.
    Based on the event payload:
    - On success: Resets failure tracking, optionally advances curriculum
    - On consecutive failures (≥3): Reduces curriculum weight (regression)

    Event payload fields used:
    - config_key: str - The board_numPlayers config identifier
    - success: bool - Whether the promotion succeeded
    - elo_change: float - Change in Elo from previous evaluation
    - consecutive_failures: int - Number of consecutive failed promotions

    December 29, 2025: Part of Phase 4 training loop improvements.
    """

    # Weight reduction per consecutive failure after threshold
    WEIGHT_REDUCTION_PER_REGRESSION = 0.15  # 15% reduction
    CONSECUTIVE_FAILURE_THRESHOLD = 3  # Start regression after 3 failures

    # Weight boost on successful promotion
    WEIGHT_BOOST_ON_SUCCESS = 0.10  # 10% boost on success

    def __init__(self):
        self._subscribed = False
        self._success_streak: dict[str, int] = {}  # config -> consecutive successes

    def subscribe(self) -> bool:
        """Subscribe to PROMOTION_COMPLETED events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import subscribe

            # Subscribe to PROMOTION_COMPLETED (string type, as emitted by auto_promotion_daemon)
            subscribe("PROMOTION_COMPLETED", self._on_promotion_completed)
            self._subscribed = True
            logger.info("[PromotionCompletedToCurriculumWatcher] Subscribed to PROMOTION_COMPLETED")
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[PromotionCompletedToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import unsubscribe

            unsubscribe("PROMOTION_COMPLETED", self._on_promotion_completed)
            self._subscribed = False
        except (ImportError, AttributeError, TypeError, RuntimeError):
            pass

    def _on_promotion_completed(self, event) -> None:
        """Handle PROMOTION_COMPLETED event - advance or regress curriculum.

        December 29, 2025: Unified handler for promotion outcomes.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            config_key = extract_config_key(payload)
            success = payload.get("success", False)
            elo_change = payload.get("elo_change", 0.0)
            consecutive_failures = payload.get("consecutive_failures", 0)
            consecutive_passes = payload.get("consecutive_passes", 0)

            if not config_key:
                return

            if success:
                self._on_promotion_success(config_key, elo_change, consecutive_passes)
            else:
                self._on_promotion_failure(config_key, elo_change, consecutive_failures)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[PromotionCompletedToCurriculumWatcher] Error handling promotion: {e}")

    def _on_promotion_success(
        self,
        config_key: str,
        elo_change: float,
        consecutive_passes: int,
    ) -> None:
        """Handle successful promotion - advance curriculum."""
        # Track success streak
        self._success_streak[config_key] = consecutive_passes

        logger.info(
            f"[PromotionCompletedToCurriculumWatcher] Promotion succeeded for {config_key}: "
            f"elo_change={elo_change:+.1f}, consecutive_passes={consecutive_passes}"
        )

        # Optionally boost curriculum weight on success (reward momentum)
        if elo_change > 20:  # Significant improvement
            self._boost_curriculum_weight(config_key, elo_change)

        # Reset failure counts in the failure watcher
        self._reset_failure_watcher(config_key)

    def _on_promotion_failure(
        self,
        config_key: str,
        elo_change: float,
        consecutive_failures: int,
    ) -> None:
        """Handle failed promotion - regress curriculum if threshold exceeded."""
        # Clear success streak
        if config_key in self._success_streak:
            del self._success_streak[config_key]

        logger.warning(
            f"[PromotionCompletedToCurriculumWatcher] Promotion failed for {config_key}: "
            f"elo_change={elo_change:+.1f}, consecutive_failures={consecutive_failures}"
        )

        # Only regress curriculum after threshold consecutive failures
        if consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
            self._regress_curriculum_weight(config_key, consecutive_failures)

    def _boost_curriculum_weight(self, config_key: str, elo_change: float) -> None:
        """Boost curriculum weight on significant promotion success."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            current_weight = feedback._current_weights.get(config_key, 1.0)

            # Boost proportional to Elo gain, capped at 20%
            boost_factor = min(0.20, self.WEIGHT_BOOST_ON_SUCCESS + (elo_change / 500))
            new_weight = min(feedback.weight_max, current_weight * (1 + boost_factor))

            if new_weight > current_weight:
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[PromotionCompletedToCurriculumWatcher] Boosted curriculum weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (elo_change={elo_change:+.1f})"
                )

                # Emit CURRICULUM_REBALANCED event
                self._emit_rebalance_event(config_key, new_weight, "promotion_success", elo_change)

        except ImportError as e:
            logger.debug(f"[PromotionCompletedToCurriculumWatcher] curriculum_feedback import error: {e}")
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"[PromotionCompletedToCurriculumWatcher] Error boosting weight: {e}")

    def _regress_curriculum_weight(self, config_key: str, consecutive_failures: int) -> None:
        """Reduce curriculum weight after consecutive failures (regression)."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            current_weight = feedback._current_weights.get(config_key, 1.0)

            # Reduce weight: 15% per failure beyond threshold, capped at 50% total reduction
            failures_over_threshold = consecutive_failures - self.CONSECUTIVE_FAILURE_THRESHOLD + 1
            reduction = min(0.50, failures_over_threshold * self.WEIGHT_REDUCTION_PER_REGRESSION)
            new_weight = max(feedback.weight_min, current_weight * (1 - reduction))

            if new_weight < current_weight:
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[PromotionCompletedToCurriculumWatcher] Reduced curriculum weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (consecutive_failures={consecutive_failures})"
                )

                # Emit CURRICULUM_REBALANCED event
                self._emit_rebalance_event(
                    config_key, new_weight, "promotion_regression", consecutive_failures
                )

        except ImportError as e:
            logger.debug(f"[PromotionCompletedToCurriculumWatcher] curriculum_feedback import error: {e}")
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"[PromotionCompletedToCurriculumWatcher] Error regressing weight: {e}")

    def _reset_failure_watcher(self, config_key: str) -> None:
        """Reset failure count in PromotionFailedToCurriculumWatcher on success."""
        try:
            from app.coordination.curriculum_integration import _watcher_instances

            watcher = _watcher_instances.get("promotion_failed_curriculum")
            if watcher and isinstance(watcher, PromotionFailedToCurriculumWatcher):
                watcher.reset_failure_count(config_key)
        except (KeyError, TypeError, AttributeError):
            pass  # Watcher not available, skip reset

    def _emit_rebalance_event(
        self,
        config_key: str,
        new_weight: float,
        trigger: str,
        value: float,
    ) -> None:
        """Emit CURRICULUM_REBALANCED event for downstream systems."""
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": trigger,
                    "changed_configs": [config_key],
                    "new_weights": {config_key: new_weight},
                    "value": value,
                    "timestamp": time.time(),
                },
                source="promotion_completed_curriculum_watcher",
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Failed to emit rebalance event: {e}")

    def get_success_streaks(self) -> dict[str, int]:
        """Get current success streaks."""
        return dict(self._success_streak)

    def health_check(self) -> "HealthCheckResult":
        """Check watcher health for DaemonManager integration."""
        from app.coordination.protocols import HealthCheckResult

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="PromotionCompletedToCurriculumWatcher not subscribed",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Tracking {len(self._success_streak)} configs with success streaks",
            details={"success_streaks": dict(self._success_streak)},
        )

class RegressionCriticalToCurriculumWatcher(CurriculumSignalBridge):
    """Boosts curriculum weight when critical model regression is detected.

    When GauntletFeedbackController detects a severe Elo regression or
    consecutive regressions (emits REGRESSION_CRITICAL), this watcher
    increases that config's curriculum weight to generate more diverse
    training data for recovery.

    Event flow (December 2025):
    1. GauntletFeedbackController detects Elo drop > threshold or consecutive regressions
    2. Emits REGRESSION_CRITICAL with severity, elo_drop, recommendation
    3. This watcher subscribes and increases curriculum weight
    4. CurriculumFeedback allocates more selfplay to affected configs
    5. Emits CURRICULUM_REBALANCED to notify downstream systems

    The weight increase is more aggressive than promotion failures since
    regression indicates the model is actively getting worse and needs
    immediate attention.

    December 30, 2025: Migrated to use CurriculumSignalBridge base class (P4.2).
    Reduces ~200 LOC of boilerplate to ~60 LOC of specific logic.
    """

    WATCHER_NAME = "regression_critical_curriculum_watcher"
    EVENT_TYPES = ["REGRESSION_CRITICAL"]  # DataEventType.REGRESSION_CRITICAL

    # Weight increase factor per regression severity
    WEIGHT_INCREASE_MODERATE = 0.25  # 25% for moderate regressions
    WEIGHT_INCREASE_SEVERE = 0.50  # 50% for severe regressions
    MAX_WEIGHT_MULTIPLIER = 3.0

    def _compute_weight_multiplier(
        self,
        config_key: str,
        payload: dict[str, Any],
    ) -> float | None:
        """Compute weight multiplier based on regression severity.

        Returns:
            Weight multiplier (1.25 for moderate, 1.50 for severe)
            Plus 0.1 per consecutive regression. Capped at 3.0x.
        """
        severity = payload.get("severity", "moderate")
        consecutive_regressions = payload.get("consecutive_regressions", 1)

        # Track consecutive regressions in state
        regression_key = f"{config_key}:regression_count"
        self.set_state(regression_key, consecutive_regressions)

        # Calculate weight increase based on severity
        if severity == "severe":
            base_increase = self.WEIGHT_INCREASE_SEVERE
        else:
            base_increase = self.WEIGHT_INCREASE_MODERATE

        # Additional increase for consecutive regressions
        multiplier = 1.0 + base_increase + (0.1 * (consecutive_regressions - 1))
        return min(self.MAX_WEIGHT_MULTIPLIER, multiplier)

    def _extract_event_details(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract regression details for logging and events."""
        return {
            "severity": payload.get("severity", "unknown"),
            "elo_drop": payload.get("elo_drop", 0),
            "consecutive_regressions": payload.get("consecutive_regressions", 1),
            "recommendation": payload.get("recommendation", ""),
        }

    def reset_regression_count(self, config_key: str) -> None:
        """Reset regression count for a config (called when model improves)."""
        regression_key = f"{config_key}:regression_count"
        if self.get_state(regression_key) is not None:
            self.reset_state(config_key)
            logger.info(f"[{self.WATCHER_NAME}] Reset regression count for {config_key}")

    def get_regression_counts(self) -> dict[str, int]:
        """Get current regression counts."""
        result = {}
        for key, value in self._state.items():
            if key.endswith(":regression_count"):
                config_key = key.rsplit(":", 1)[0]
                result[config_key] = value
        return result

class QualityPenaltyToCurriculumWatcher(CurriculumSignalBridge):
    """Reduces curriculum weight when quality penalties are applied.

    When AdaptiveController applies a quality penalty to a config (emits
    QUALITY_PENALTY_APPLIED), this watcher reduces that config's curriculum
    weight proportionally. This focuses training resources away from configs
    that are producing low-quality data.

    Event flow (December 2025):
    1. AdaptiveController detects low quality data
    2. Emits QUALITY_PENALTY_APPLIED with rate_multiplier and penalty amount
    3. This watcher subscribes and reduces curriculum weight
    4. CurriculumFeedback allocates less selfplay to affected configs
    5. Emits CURRICULUM_REBALANCED to notify downstream systems

    December 30, 2025: Migrated to use CurriculumSignalBridge base class (P4.2).
    Reduces ~200 LOC of boilerplate to ~60 LOC of specific logic.
    """

    WATCHER_NAME = "quality_penalty_curriculum_watcher"
    EVENT_TYPES = ["QUALITY_PENALTY_APPLIED"]  # DataEventType.QUALITY_PENALTY_APPLIED

    # Weight reduction factor per penalty unit (cumulative with penalties)
    WEIGHT_REDUCTION_PER_PENALTY = 0.15  # 15% reduction per penalty unit
    MIN_WEIGHT_MULTIPLIER = 0.3  # Never reduce below 30%

    def _compute_weight_multiplier(
        self,
        config_key: str,
        payload: dict[str, Any],
    ) -> float | None:
        """Compute weight multiplier based on penalty severity.

        Returns:
            Weight multiplier (< 1.0 for reduction).
            penalty=0 → 1.0, penalty=1 → 0.85, penalty=2 → 0.70
            Minimum 0.3x to prevent complete starvation.
        """
        new_penalty = payload.get("new_penalty", 0.0)

        # Track penalty in state
        penalty_key = f"{config_key}:penalty"
        old_penalty = self.get_state(penalty_key, 0.0)

        # Only apply if penalty changed significantly
        if abs(new_penalty - old_penalty) < 0.02:
            return None  # Skip - no significant change

        self.set_state(penalty_key, new_penalty)

        # Calculate weight reduction based on penalty severity
        # penalty=0 → weight=1.0, penalty=1 → weight=0.85, penalty=2 → weight=0.70
        multiplier = max(
            self.MIN_WEIGHT_MULTIPLIER,
            1.0 - (new_penalty * self.WEIGHT_REDUCTION_PER_PENALTY),
        )
        return multiplier

    def _extract_event_details(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract quality penalty details for logging and events."""
        return {
            "penalty": payload.get("new_penalty", 0.0),
            "rate_multiplier": payload.get("rate_multiplier", 1.0),
            "reason": payload.get("reason", ""),
        }

    def get_penalty_weights(self) -> dict[str, float]:
        """Get current penalty-based weight factors."""
        result = {}
        for key, value in self._state.items():
            if key.endswith(":penalty"):
                config_key = key.rsplit(":", 1)[0]
                # Convert penalty to weight factor
                result[config_key] = max(
                    self.MIN_WEIGHT_MULTIPLIER,
                    1.0 - (value * self.WEIGHT_REDUCTION_PER_PENALTY),
                )
        return result

    def reset_penalty(self, config_key: str) -> None:
        """Reset penalty weight for a config (called when quality recovers)."""
        penalty_key = f"{config_key}:penalty"
        if self.get_state(penalty_key) is not None:
            self.reset_state(config_key)
            logger.info(f"[{self.WATCHER_NAME}] Reset penalty for {config_key}")

class ArchitectureToCurriculumBridge(CurriculumSignalBridge):
    """Boosts curriculum weight for configs with underperforming architectures.

    When ArchitectureFeedbackController emits ARCHITECTURE_WEIGHTS_UPDATED,
    this bridge checks if any architecture is significantly underperforming
    (weight < threshold). For underperforming architectures, it boosts the
    curriculum weight to generate more diverse training data.

    This enables cross-architecture learning by ensuring:
    1. Underperforming architectures get more training data
    2. High weight disparity triggers exploration for lagging architectures
    3. Knowledge can transfer between architectures via shared curriculum focus

    Event flow (January 2026):
    1. ArchitectureFeedbackController detects architecture performance
    2. Emits ARCHITECTURE_WEIGHTS_UPDATED with {config_key, weights: {arch: weight}}
    3. This bridge checks for underperforming architectures
    4. If found, boosts curriculum weight for the config
    5. Emits CURRICULUM_REBALANCED to notify downstream systems

    Expected Elo improvement: +25-35 Elo from better cross-architecture learning.

    January 2026 Sprint 17: Part of cross-architecture curriculum signals implementation.
    """

    WATCHER_NAME = "architecture_curriculum_bridge"
    EVENT_TYPES = ["ARCHITECTURE_WEIGHTS_UPDATED"]

    # Threshold for underperforming architecture (below this triggers boost)
    UNDERPERFORMING_THRESHOLD = 0.15  # 15% allocation = underperforming

    # Weight disparity ratio that triggers exploration boost
    DISPARITY_THRESHOLD = 3.0  # If max/min > 3x, boost exploration

    # Boost factors
    UNDERPERFORMER_BOOST = 0.25  # 25% boost for underperforming arch
    DISPARITY_BOOST = 0.15  # 15% boost when high disparity

    def _compute_weight_multiplier(
        self,
        config_key: str,
        payload: dict[str, Any],
    ) -> float | None:
        """Compute weight multiplier based on architecture performance.

        Boosts curriculum weight when:
        1. Any architecture has weight < UNDERPERFORMING_THRESHOLD
        2. Weight disparity (max/min) > DISPARITY_THRESHOLD

        Returns:
            Weight multiplier (> 1.0 for boost), or None to skip adjustment.
        """
        weights = payload.get("weights", {})
        if not weights or len(weights) < 2:
            return None  # Need at least 2 architectures to compare

        # Filter to architectures with meaningful weights
        valid_weights = {
            arch: w for arch, w in weights.items()
            if isinstance(w, (int, float)) and w > 0.001
        }
        if not valid_weights:
            return None

        # Check for underperforming architectures
        underperforming_archs = [
            arch for arch, w in valid_weights.items()
            if w < self.UNDERPERFORMING_THRESHOLD
        ]

        # Check weight disparity
        weight_values = list(valid_weights.values())
        max_weight = max(weight_values)
        min_weight = min(weight_values)
        disparity = max_weight / min_weight if min_weight > 0.001 else 1.0

        # Track state for logging
        state_key = f"{config_key}:arch_state"
        prev_state = self.get_state(state_key, {"underperformers": [], "disparity": 1.0})

        # Calculate boost
        boost = 0.0

        if underperforming_archs:
            # Boost proportional to number of underperformers
            boost += self.UNDERPERFORMER_BOOST * len(underperforming_archs)
            logger.info(
                f"[{self.WATCHER_NAME}] Config {config_key} has underperforming architectures: "
                f"{underperforming_archs} (weights < {self.UNDERPERFORMING_THRESHOLD:.0%})"
            )

        if disparity > self.DISPARITY_THRESHOLD:
            # Boost when high disparity exists
            boost += self.DISPARITY_BOOST
            logger.info(
                f"[{self.WATCHER_NAME}] Config {config_key} has high weight disparity: "
                f"{disparity:.1f}x (max: {max_weight:.2f}, min: {min_weight:.2f})"
            )

        # Update state
        self.set_state(state_key, {
            "underperformers": underperforming_archs,
            "disparity": disparity,
            "last_weights": valid_weights,
        })

        if boost < 0.01:
            return None  # No significant boost needed

        return 1.0 + min(boost, 0.5)  # Cap at 50% max boost

    def _extract_event_details(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract architecture details for logging and events."""
        weights = payload.get("weights", {})
        return {
            "architecture_count": len(weights),
            "weights": {k: f"{v:.2f}" for k, v in weights.items()} if weights else {},
            "timestamp": payload.get("timestamp", 0),
        }

    def get_architecture_status(self) -> dict[str, Any]:
        """Get current architecture status across all tracked configs."""
        result = {}
        for key, value in self._state.items():
            if key.endswith(":arch_state"):
                config_key = key.rsplit(":", 1)[0]
                result[config_key] = value
        return result

    def get_underperforming_configs(self) -> list[str]:
        """Get configs that currently have underperforming architectures."""
        underperforming = []
        for key, value in self._state.items():
            if key.endswith(":arch_state"):
                config_key = key.rsplit(":", 1)[0]
                if value.get("underperformers"):
                    underperforming.append(config_key)
        return underperforming

class QualityToTemperatureWatcher:
    """Adjusts exploration temperature based on training data quality.

    Low quality data indicates the model may be stuck in a local minimum.
    This watcher increases exploration temperature to generate more diverse
    training data.

    Event flow:
    1. QualityFeedbackWatcher detects low quality
    2. This watcher receives QUALITY_FEEDBACK_ADJUSTED event
    3. Updates temperature schedule exploration_boost

    Thresholds (December 28, 2025 - migrated to coordination_defaults.py):
    - RINGRIFT_EXPLORATION_BOOST_FACTOR (default: 1.3, i.e., +30% exploration)
    - RINGRIFT_LOW_QUALITY_THRESHOLD (default: 0.3)
    """

    # Load from centralized defaults (December 28, 2025)
    try:
        from app.config.coordination_defaults import CurriculumDefaults
        EXPLORATION_BOOST_FACTOR = CurriculumDefaults.EXPLORATION_BOOST_FACTOR
        _LOW_QUALITY_DEFAULT = CurriculumDefaults.LOW_QUALITY_THRESHOLD
    except ImportError:
        # Fallback for standalone testing
        EXPLORATION_BOOST_FACTOR = 1.3
        _LOW_QUALITY_DEFAULT = 0.3

    @property
    def LOW_QUALITY_THRESHOLD(self) -> float:
        """Get low quality threshold from centralized config.

        Note: Still supports thresholds.py for backward compatibility,
        but prefers coordination_defaults.py.
        """
        try:
            from app.config.coordination_defaults import CurriculumDefaults
            return CurriculumDefaults.LOW_QUALITY_THRESHOLD
        except ImportError:
            try:
                from app.config.thresholds import LOW_QUALITY_THRESHOLD
                return LOW_QUALITY_THRESHOLD
            except ImportError:
                return 0.3  # Fallback default

    def __init__(self):
        self._subscribed = False
        self._quality_boosts: dict[str, float] = {}  # config -> boost factor

    def subscribe(self) -> bool:
        """Subscribe to quality events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import DataEventType, subscribe

            subscribe(DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_adjusted)
            subscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_updated)
            self._subscribed = True
            logger.info("[QualityToTemperatureWatcher] Subscribed to quality events")
            return True
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid subscription arguments
            # RuntimeError: subscription failed
            logger.warning(f"[QualityToTemperatureWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, unsubscribe

            unsubscribe(DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_adjusted)
            unsubscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_updated)
            self._subscribed = False
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid unsubscription arguments
            # RuntimeError: unsubscription failed
            pass

    def _on_quality_adjusted(self, event: Any) -> None:
        """Handle QUALITY_FEEDBACK_ADJUSTED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = extract_config_key(payload)
        avg_quality = payload.get("avg_quality", 0.5)

        if not config_key:
            return

        self._update_exploration_boost(config_key, avg_quality)

    def _on_quality_updated(self, event: Any) -> None:
        """Handle QUALITY_SCORE_UPDATED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = extract_config_key(payload)
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
        except (AttributeError, TypeError, KeyError) as e:
            # AttributeError: scheduler method missing
            # TypeError: invalid boost type
            # KeyError: unknown config_key
            logger.debug(f"Failed to apply temperature boost: {e}")

        # Also emit event for downstream systems
        self._emit_exploration_boost(config_key, boost)

    def _emit_exploration_boost(self, config_key: str, boost: float) -> None:
        """Emit EXPLORATION_BOOST event."""
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "EXPLORATION_BOOST",
                {
                    "config": config_key,
                    "boost_factor": boost,
                    "reason": "low_quality_data",
                    "timestamp": time.time(),
                },
                source="quality_temperature_watcher",
            )
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid event arguments
            # RuntimeError: publish failed
            pass

    def get_exploration_boost(self, config_key: str) -> float:
        """Get the current exploration boost for a config."""
        return self._quality_boosts.get(config_key, 1.0)

    def get_all_boosts(self) -> dict[str, float]:
        """Get all current exploration boosts."""
        return dict(self._quality_boosts)

    def health_check(self) -> "HealthCheckResult":
        """Check watcher health for DaemonManager integration."""
        from app.coordination.protocols import HealthCheckResult

        if not self._subscribed:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="QualityToTemperatureWatcher not subscribed",
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Tracking {len(self._quality_boosts)} configs with boosts",
            details={"quality_boosts": dict(self._quality_boosts)},
        )
