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

from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

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

    Event flow (Phase 5 - December 2025):
    1. EVALUATION_COMPLETED triggers FeedbackAccelerator.record_elo_update()
    2. This bridge subscribes to EVALUATION_COMPLETED and syncs immediately
    3. CurriculumFeedback._current_weights updated
    4. CURRICULUM_REBALANCED event emitted

    Note: Converted from 60-second polling to event-driven in Phase 5.
    """

    def __init__(
        self,
        poll_interval_seconds: float = 10.0,  # Dec 2025: Reduced from 60s for faster feedback
        momentum_weight_boost: float = 0.3,
    ):
        self.poll_interval_seconds = poll_interval_seconds
        self.momentum_weight_boost = momentum_weight_boost

        self._running = False
        self._event_subscribed = False
        self._fallback_thread: threading.Thread | None = None
        self._last_weights: dict[str, float] = {}
        # December 2025: Track selfplay allocation shares for curriculum alignment
        self._last_allocation_share: dict[str, float] = {}
        # December 2025: Fix AttributeError in health_check() - initialize missing attrs
        self._last_sync_time: float = 0.0

    def start(self) -> None:
        """Start the momentum-to-curriculum bridge.

        Phase 5: Prefer event-driven, fallback to polling if events unavailable.
        """
        if self._running:
            return

        self._running = True

        # Try event-driven first (Phase 5)
        if self._subscribe_to_events():
            logger.info("[MomentumToCurriculumBridge] Started (event-driven mode)")
        else:
            # Fallback to polling if event subscription fails
            self._fallback_thread = threading.Thread(
                target=self._poll_loop,
                name="MomentumCurriculumBridge",
                daemon=True,
            )
            self._fallback_thread.start()
            logger.info("[MomentumToCurriculumBridge] Started (polling fallback mode)")

    def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        self._unsubscribe_from_events()
        if self._fallback_thread:
            self._fallback_thread.join(timeout=5.0)
            self._fallback_thread = None
        logger.info("[MomentumToCurriculumBridge] Stopped")

    def _subscribe_to_events(self) -> bool:
        """Subscribe to events for reactive weight sync.

        Phase 5 (December 2025): Event-driven replaces polling for sub-second latency.
        Phase 21.2 (December 2025): Also subscribe to SELFPLAY_RATE_CHANGED for Elo momentum sync.
        """
        if self._event_subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            if router is None:
                logger.debug("[MomentumToCurriculumBridge] Event router not available")
                return False

            # Use enum directly (router normalizes both enum and .value)
            router.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)

            # Phase 21.2: Subscribe to SELFPLAY_RATE_CHANGED for Elo momentum → curriculum sync
            if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                router.subscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)

            # December 2025: Subscribe to ELO_SIGNIFICANT_CHANGE for curriculum rebalance triggers
            if hasattr(DataEventType, 'ELO_SIGNIFICANT_CHANGE'):
                router.subscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, self._on_elo_significant_change)

            # December 2025: Subscribe to SELFPLAY_ALLOCATION_UPDATED to track allocation changes
            if hasattr(DataEventType, 'SELFPLAY_ALLOCATION_UPDATED'):
                router.subscribe(DataEventType.SELFPLAY_ALLOCATION_UPDATED, self._on_selfplay_allocation_updated)

            # December 2025 Phase 2: Subscribe to MODEL_PROMOTED to rebalance curriculum
            # when a new model is promoted. This ensures curriculum weights are adjusted
            # based on the latest model strength.
            if hasattr(DataEventType, 'MODEL_PROMOTED'):
                router.subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)

            logger.info("[MomentumToCurriculumBridge] Subscribed to EVALUATION_COMPLETED, SELFPLAY_RATE_CHANGED, ELO_SIGNIFICANT_CHANGE, SELFPLAY_ALLOCATION_UPDATED, MODEL_PROMOTED")
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: modules not available
            # AttributeError: router method missing
            # TypeError: invalid subscription arguments
            # RuntimeError: subscription failed
            logger.debug(f"[MomentumToCurriculumBridge] Event subscription failed: {e}")
            return False
        finally:
            # December 27, 2025: Always set _event_subscribed = True in finally block
            # This ensures cleanup runs even if subscription partially fails
            self._event_subscribed = True

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events."""
        if not self._event_subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            if router:
                router.unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
                if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                    router.unsubscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)
                if hasattr(DataEventType, 'ELO_SIGNIFICANT_CHANGE'):
                    router.unsubscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, self._on_elo_significant_change)
                # December 2025: Unsubscribe from SELFPLAY_ALLOCATION_UPDATED
                if hasattr(DataEventType, 'SELFPLAY_ALLOCATION_UPDATED'):
                    router.unsubscribe(DataEventType.SELFPLAY_ALLOCATION_UPDATED, self._on_selfplay_allocation_updated)
                # December 2025 Phase 2: Unsubscribe from MODEL_PROMOTED
                if hasattr(DataEventType, 'MODEL_PROMOTED'):
                    router.unsubscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            self._event_subscribed = False
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # ImportError: modules not available
            # AttributeError: router method missing
            # TypeError: invalid unsubscription arguments
            # RuntimeError: unsubscription failed
            pass

    def _on_evaluation_completed(self, event) -> None:
        """Handle EVALUATION_COMPLETED event - sync weights immediately.

        Phase 5 (December 2025): Reactive weight sync replaces polling.
        This runs within ~1 second of evaluation completing, vs 60 second polling.
        """
        try:
            self._sync_weights()
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: feedback modules not available
            # AttributeError: method missing
            # TypeError: invalid weight types
            # RuntimeError: sync operation failed
            logger.warning(f"[MomentumToCurriculumBridge] Error syncing on event: {e}")

    def _on_selfplay_rate_changed(self, event) -> None:
        """Handle SELFPLAY_RATE_CHANGED event - sync curriculum weights on Elo momentum.

        Phase 21.2 (December 2025): Close the Elo → Curriculum feedback loop.
        When selfplay rate changes significantly (>20%), it indicates momentum
        shift in training effectiveness. We sync curriculum weights to reallocate
        resources to configs with momentum.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            config_key = payload.get("config", "")
            change_percent = payload.get("change_percent", 0)
            momentum_state = payload.get("momentum_state", "stable")

            if not config_key:
                return

            # Only sync on significant rate changes (>20%)
            if abs(change_percent) < 20:
                return

            logger.info(
                f"[MomentumToCurriculumBridge] Selfplay rate change for {config_key}: "
                f"{change_percent:+.1f}%, momentum={momentum_state} - triggering weight sync"
            )

            # Sync curriculum weights based on momentum state
            self._sync_weights_for_momentum(config_key, momentum_state, change_percent)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # AttributeError: event attribute missing
            # KeyError: missing payload field
            # TypeError: invalid data types
            # ValueError: invalid percentage value
            logger.warning(f"[MomentumToCurriculumBridge] Error handling rate change: {e}")

    def _on_elo_significant_change(self, event) -> None:
        """Handle ELO_SIGNIFICANT_CHANGE event - trigger curriculum rebalancing.

        December 2025: Wire ELO_SIGNIFICANT_CHANGE to curriculum weights.
        When a config's Elo changes significantly (±30 from baseline), we
        rebalance curriculum weights to either capitalize on momentum or
        reduce focus on stalled configs.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            config_key = payload.get("config", payload.get("config_key", ""))
            old_elo = payload.get("old_elo", 0)
            new_elo = payload.get("new_elo", payload.get("elo", 0))
            delta = payload.get("delta", new_elo - old_elo if old_elo else 0)
            significance = payload.get("significance", "unknown")

            if not config_key:
                return

            logger.info(
                f"[MomentumToCurriculumBridge] ELO_SIGNIFICANT_CHANGE for {config_key}: "
                f"Δ={delta:+.1f} ({significance})"
            )

            # Determine momentum direction
            if delta > 30:
                momentum_state = "accelerating"
            elif delta < -30:
                momentum_state = "decelerating"
            else:
                momentum_state = "stable"

            # Sync curriculum weights based on Elo momentum
            self._sync_weights_for_momentum(config_key, momentum_state, delta)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling Elo change: {e}")

    def _on_selfplay_allocation_updated(self, event) -> None:
        """Handle SELFPLAY_ALLOCATION_UPDATED event - track allocation shifts.

        December 2025: Wire SELFPLAY_ALLOCATION_UPDATED to curriculum tracking.
        When SelfplayScheduler allocates games, this event tells us which configs
        are receiving focus. We use this to:
        - Track which configs are currently prioritized by the scheduler
        - Adjust curriculum weights to align with scheduler allocation
        - Detect allocation imbalances that may need curriculum correction
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            trigger = payload.get("trigger", "")
            total_games = payload.get("total_games", 0)
            configs_allocated = payload.get("configs_allocated", [])
            allocation = payload.get("allocation", {})

            if not configs_allocated:
                return

            # Log allocation for tracking
            logger.debug(
                f"[MomentumToCurriculumBridge] SELFPLAY_ALLOCATION_UPDATED: "
                f"trigger={trigger}, games={total_games}, configs={configs_allocated}"
            )

            # Track allocation patterns for curriculum alignment
            # If scheduler is heavily weighting a config, curriculum should align
            if total_games > 0 and allocation:
                total_allocated_games = sum(
                    sum(node_games.values()) if isinstance(node_games, dict) else 0
                    for node_games in allocation.values()
                )

                for config_key, node_allocation in allocation.items():
                    config_games = sum(node_allocation.values()) if isinstance(node_allocation, dict) else 0
                    if config_games > 0 and total_allocated_games > 0:
                        allocation_share = config_games / total_allocated_games
                        # Store allocation share for curriculum weight alignment
                        self._last_allocation_share[config_key] = allocation_share

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"[MomentumToCurriculumBridge] Error handling allocation update: {e}")

    def _on_model_promoted(self, event) -> None:
        """Handle MODEL_PROMOTED event - rebalance curriculum weights.

        December 2025 Phase 2: When a model is promoted, curriculum weights should
        be recalculated based on the new model's strength. This ensures:
        - Curriculum reflects the promoted model's capabilities
        - Exploration/exploitation is rebalanced for the new baseline
        - Other configs' relative weights are adjusted accordingly
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            model_id = payload.get("model_id", "")
            elo_improvement = payload.get("elo_improvement", 0.0)

            if not board_type or not num_players:
                return

            config_key = f"{board_type}_{num_players}p"

            logger.info(
                f"[MomentumToCurriculumBridge] MODEL_PROMOTED for {config_key}: "
                f"model={model_id}, elo_improvement={elo_improvement:+.1f}"
            )

            # Trigger full curriculum weight sync after promotion
            # The new model represents a new baseline for training
            self._sync_weights()

            # Emit curriculum rebalanced event for downstream consumers
            try:
                from app.coordination.event_emitters import emit_curriculum_rebalanced
                import asyncio

                async def _emit():
                    await emit_curriculum_rebalanced(
                        trigger="model_promoted",
                        configs_affected=[config_key],
                        source="curriculum_integration",
                    )

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_emit())
                except RuntimeError:
                    # No running loop - skip async emission
                    pass

            except ImportError:
                pass

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling model promotion: {e}")

    def _sync_weights_for_momentum(
        self,
        config_key: str,
        momentum_state: str,
        change_percent: float,
    ) -> None:
        """Sync curriculum weights based on Elo momentum.

        Phase 21.2: Adjust curriculum weight for specific config based on momentum:
        - accelerating: Boost weight to capitalize on fast learning
        - decelerating: Reduce weight slightly, model may need different data
        - stable: Maintain current weight
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            current_weight = feedback._current_weights.get(config_key, 1.0)

            if momentum_state == "accelerating":
                # Model is learning fast - increase resources to capitalize
                new_weight = min(feedback.weight_max, current_weight * (1 + abs(change_percent) / 200))
            elif momentum_state == "decelerating":
                # Model is slowing down - slightly reduce weight
                new_weight = max(feedback.weight_min, current_weight * (1 - abs(change_percent) / 400))
            else:
                # Stable - no change
                return

            if abs(new_weight - current_weight) > 0.05:
                feedback._current_weights[config_key] = new_weight
                self._last_weights[config_key] = new_weight

                logger.info(
                    f"[MomentumToCurriculumBridge] Curriculum weight adjusted for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (momentum={momentum_state})"
                )

                # Emit CURRICULUM_REBALANCED event
                self._emit_rebalance_event([config_key], {config_key: new_weight})

        except ImportError as e:
            logger.debug(f"[MomentumToCurriculumBridge] curriculum_feedback import error: {e}")
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # AttributeError: feedback method missing
            # TypeError: invalid weight types
            # ValueError: invalid weight values
            # KeyError: unknown config_key
            logger.warning(f"[MomentumToCurriculumBridge] Error syncing momentum weights: {e}")

    def _poll_loop(self) -> None:
        """Fallback poll loop - used only if event subscription fails."""
        while self._running:
            try:
                self._sync_weights()
            except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                # ImportError: feedback modules not available
                # AttributeError: method missing
                # TypeError: invalid weight types
                # RuntimeError: sync operation failed
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
            self._last_sync_time = time.time()  # Track last sync for health_check

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
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid event arguments
            # RuntimeError: publish failed
            logger.debug(f"Failed to emit rebalance event: {e}")

    def force_sync(self) -> dict[str, float]:
        """Force immediate weight sync."""
        self._sync_weights()
        return self._last_weights

    def health_check(self) -> HealthCheckResult:
        """Perform health check for daemon manager integration.

        Returns:
            HealthCheckResult with current status

        December 2025: Fixed AttributeError by using _fallback_thread (not _sync_thread)
        and added exception handling to prevent crash loops.
        """
        try:
            active_configs = sum(1 for w in self._last_weights.values() if w > 0.01)

            # Use _fallback_thread (the actual attribute) or check event subscription
            thread_active = (
                self._fallback_thread is not None and self._fallback_thread.is_alive()
            )
            is_active = thread_active or self._event_subscribed

            if is_active:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.RUNNING,
                    message=f"Sync active, {active_configs} configs with weight",
                    details={
                        "running": self._running,
                        "event_subscribed": self._event_subscribed,
                        "fallback_thread_active": thread_active,
                        "last_sync": self._last_sync_time,
                        "active_configs": active_configs,
                    },
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.READY,  # READY instead of IDLE (IDLE doesn't exist)
                message=f"Sync idle, {active_configs} configs with weight",
                details={
                    "running": self._running,
                    "event_subscribed": self._event_subscribed,
                    "last_sync": self._last_sync_time,
                    "active_configs": active_configs,
                },
            )
        except Exception as e:
            # Prevent health_check crashes from causing daemon restart loops
            logger.warning(f"[MomentumToCurriculumBridge] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )


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
            from app.coordination.event_router import DataEventType

            router = get_router()
            # P1.4 Dec 2025: Use DataEventType enum for type-safe emission
            router.publish_sync(
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


# =============================================================================
# 2.5. QUALITY_PENALTY_APPLIED → Curriculum Weight Reduction
# =============================================================================


class PromotionFailedToCurriculumWatcher:
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
    """

    # Weight increase factor per consecutive failure (cumulative)
    WEIGHT_INCREASE_PER_FAILURE = 0.20  # 20% increase per failure

    def __init__(self):
        self._subscribed = False
        self._failure_counts: dict[str, int] = {}  # config -> consecutive failures

    def subscribe(self) -> bool:
        """Subscribe to PROMOTION_FAILED events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.events.types import RingRiftEventType

            router = get_router()
            if router is None:
                logger.debug("[PromotionFailedToCurriculumWatcher] Event router not available")
                return False

            router.subscribe(RingRiftEventType.PROMOTION_FAILED, self._on_promotion_failed)
            self._subscribed = True
            logger.info("[PromotionFailedToCurriculumWatcher] Subscribed to PROMOTION_FAILED")
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid subscription arguments
            # RuntimeError: subscription failed
            logger.warning(f"[PromotionFailedToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.events.types import RingRiftEventType

            router = get_router()
            if router:
                router.unsubscribe(RingRiftEventType.PROMOTION_FAILED, self._on_promotion_failed)
            self._subscribed = False
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid unsubscription arguments
            # RuntimeError: unsubscription failed
            pass

    def _on_promotion_failed(self, event) -> None:
        """Handle PROMOTION_FAILED event - increase curriculum weight.

        December 2025: Closes the promotion failure → curriculum weight feedback loop.
        When promotion fails, increase selfplay allocation to generate more diverse data.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            config_key = payload.get("config_key", payload.get("config", ""))
            error = payload.get("error", "unknown")
            model_id = payload.get("model_id", "")

            if not config_key:
                return

            # Track consecutive failures
            self._failure_counts[config_key] = self._failure_counts.get(config_key, 0) + 1
            failure_count = self._failure_counts[config_key]

            logger.info(
                f"[PromotionFailedToCurriculumWatcher] Promotion failed for {config_key}: "
                f"model={model_id}, error={error}, consecutive_failures={failure_count}"
            )

            # Increase curriculum weight to generate more diverse training data
            self._increase_curriculum_weight(config_key, failure_count, error)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # AttributeError: event attribute missing
            # KeyError: missing payload field
            # TypeError: invalid data types
            # ValueError: invalid values
            logger.warning(f"[PromotionFailedToCurriculumWatcher] Error handling promotion failure: {e}")

    def _increase_curriculum_weight(
        self,
        config_key: str,
        failure_count: int,
        error: str,
    ) -> None:
        """Increase curriculum weight based on consecutive failures."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            current_weight = feedback._current_weights.get(config_key, 1.0)

            # Increase weight: 20% per failure, up to 2.5x max
            # failure_count=1 -> 1.2x, failure_count=2 -> 1.44x, etc.
            weight_multiplier = min(2.5, 1.0 + (failure_count * self.WEIGHT_INCREASE_PER_FAILURE))
            new_weight = min(feedback.weight_max, current_weight * weight_multiplier)

            if new_weight > current_weight:
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[PromotionFailedToCurriculumWatcher] Increased curriculum weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (failures={failure_count}, error={error})"
                )

                # Emit CURRICULUM_REBALANCED event
                self._emit_rebalance_event(config_key, new_weight, failure_count)

        except ImportError as e:
            logger.debug(f"[PromotionFailedToCurriculumWatcher] curriculum_feedback import error: {e}")
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # AttributeError: feedback method missing
            # TypeError: invalid weight types
            # ValueError: invalid weight values
            # KeyError: unknown config_key
            logger.warning(f"[PromotionFailedToCurriculumWatcher] Error increasing weight: {e}")

    def _emit_rebalance_event(
        self,
        config_key: str,
        new_weight: float,
        failure_count: int,
    ) -> None:
        """Emit CURRICULUM_REBALANCED event for downstream systems."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "promotion_failed",
                    "changed_configs": [config_key],
                    "new_weights": {config_key: new_weight},
                    "failure_count": failure_count,
                    "timestamp": time.time(),
                },
                source="promotion_failed_curriculum_watcher",
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid event arguments
            # RuntimeError: publish failed
            logger.debug(f"Failed to emit rebalance event: {e}")

    def reset_failure_count(self, config_key: str) -> None:
        """Reset failure count for a config (called when promotion succeeds)."""
        if config_key in self._failure_counts:
            del self._failure_counts[config_key]
            logger.info(f"[PromotionFailedToCurriculumWatcher] Reset failure count for {config_key}")

    def get_failure_counts(self) -> dict[str, int]:
        """Get current failure counts."""
        return dict(self._failure_counts)


# =============================================================================
# 2.4.1. REGRESSION_CRITICAL → Curriculum Weight Boost (December 27, 2025)
# =============================================================================


class RegressionCriticalToCurriculumWatcher:
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
    """

    # Weight increase factor per regression severity
    WEIGHT_INCREASE_MODERATE = 0.25  # 25% for moderate regressions
    WEIGHT_INCREASE_SEVERE = 0.50  # 50% for severe regressions

    def __init__(self):
        self._subscribed = False
        self._regression_counts: dict[str, int] = {}  # config -> consecutive regressions

    def subscribe(self) -> bool:
        """Subscribe to REGRESSION_CRITICAL events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            if router is None:
                logger.debug("[RegressionCriticalToCurriculumWatcher] Event router not available")
                return False

            router.subscribe(DataEventType.REGRESSION_CRITICAL, self._on_regression_critical)
            self._subscribed = True
            logger.info("[RegressionCriticalToCurriculumWatcher] Subscribed to REGRESSION_CRITICAL")
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[RegressionCriticalToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            if router:
                router.unsubscribe(DataEventType.REGRESSION_CRITICAL, self._on_regression_critical)
            self._subscribed = False
        except (ImportError, AttributeError, TypeError, RuntimeError):
            pass

    def _on_regression_critical(self, event) -> None:
        """Handle REGRESSION_CRITICAL event - boost curriculum weight.

        December 2025: Closes the regression → curriculum weight feedback loop.
        When model regression is detected, increase selfplay allocation to
        generate more diverse data for recovery training.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            config_key = payload.get("config_key", payload.get("config", ""))
            severity = payload.get("severity", "unknown")
            elo_drop = payload.get("elo_drop", 0)
            consecutive_regressions = payload.get("consecutive_regressions", 1)
            recommendation = payload.get("recommendation", "")

            if not config_key:
                return

            # Track consecutive regressions
            self._regression_counts[config_key] = consecutive_regressions

            logger.warning(
                f"[RegressionCriticalToCurriculumWatcher] Regression detected for {config_key}: "
                f"severity={severity}, elo_drop={elo_drop:.0f}, "
                f"consecutive_regressions={consecutive_regressions}, recommendation={recommendation}"
            )

            # Increase curriculum weight to generate more diverse training data
            self._increase_curriculum_weight(config_key, severity, elo_drop, consecutive_regressions)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[RegressionCriticalToCurriculumWatcher] Error handling regression: {e}")

    def _increase_curriculum_weight(
        self,
        config_key: str,
        severity: str,
        elo_drop: float,
        consecutive_regressions: int,
    ) -> None:
        """Increase curriculum weight based on regression severity."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            current_weight = feedback._current_weights.get(config_key, 1.0)

            # Calculate weight increase based on severity
            if severity == "severe":
                base_increase = self.WEIGHT_INCREASE_SEVERE
            else:
                base_increase = self.WEIGHT_INCREASE_MODERATE

            # Additional increase for consecutive regressions
            weight_multiplier = 1.0 + base_increase + (0.1 * (consecutive_regressions - 1))
            weight_multiplier = min(3.0, weight_multiplier)  # Cap at 3x

            new_weight = min(feedback.weight_max, current_weight * weight_multiplier)

            if new_weight > current_weight:
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[RegressionCriticalToCurriculumWatcher] Increased curriculum weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (severity={severity}, elo_drop={elo_drop:.0f})"
                )

                # Emit CURRICULUM_REBALANCED event
                self._emit_rebalance_event(config_key, new_weight, severity, elo_drop)

        except ImportError as e:
            logger.debug(f"[RegressionCriticalToCurriculumWatcher] curriculum_feedback import error: {e}")
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"[RegressionCriticalToCurriculumWatcher] Error increasing weight: {e}")

    def _emit_rebalance_event(
        self,
        config_key: str,
        new_weight: float,
        severity: str,
        elo_drop: float,
    ) -> None:
        """Emit CURRICULUM_REBALANCED event for downstream systems."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "regression_critical",
                    "changed_configs": [config_key],
                    "new_weights": {config_key: new_weight},
                    "severity": severity,
                    "elo_drop": elo_drop,
                    "timestamp": time.time(),
                },
                source="regression_critical_curriculum_watcher",
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Failed to emit rebalance event: {e}")

    def reset_regression_count(self, config_key: str) -> None:
        """Reset regression count for a config (called when model improves)."""
        if config_key in self._regression_counts:
            del self._regression_counts[config_key]
            logger.info(f"[RegressionCriticalToCurriculumWatcher] Reset regression count for {config_key}")

    def get_regression_counts(self) -> dict[str, int]:
        """Get current regression counts."""
        return dict(self._regression_counts)


# =============================================================================
# 2.5. QUALITY_PENALTY_APPLIED → Curriculum Weight Reduction
# =============================================================================


class QualityPenaltyToCurriculumWatcher:
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
    """

    # Weight reduction factor per penalty unit (cumulative with penalties)
    WEIGHT_REDUCTION_PER_PENALTY = 0.15  # 15% reduction per penalty unit

    def __init__(self):
        self._subscribed = False
        self._penalty_weights: dict[str, float] = {}  # config -> weight multiplier

    def subscribe(self) -> bool:
        """Subscribe to QUALITY_PENALTY_APPLIED events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            if router is None:
                logger.debug("[QualityPenaltyToCurriculumWatcher] Event router not available")
                return False

            router.subscribe(DataEventType.QUALITY_PENALTY_APPLIED, self._on_quality_penalty)
            self._subscribed = True
            logger.info("[QualityPenaltyToCurriculumWatcher] Subscribed to QUALITY_PENALTY_APPLIED")
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid subscription arguments
            # RuntimeError: subscription failed
            logger.warning(f"[QualityPenaltyToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            if router:
                router.unsubscribe(DataEventType.QUALITY_PENALTY_APPLIED, self._on_quality_penalty)
            self._subscribed = False
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid unsubscription arguments
            # RuntimeError: unsubscription failed
            pass

    def _on_quality_penalty(self, event) -> None:
        """Handle QUALITY_PENALTY_APPLIED event - reduce curriculum weight.

        December 2025: Closes the quality → curriculum weight feedback loop.
        When quality penalty is applied, proportionally reduce selfplay allocation.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            config_key = payload.get("config_key", payload.get("config", ""))
            new_penalty = payload.get("new_penalty", 0.0)
            rate_multiplier = payload.get("rate_multiplier", 1.0)
            reason = payload.get("reason", "")

            if not config_key:
                return

            # Calculate weight reduction based on penalty severity
            # penalty=0 → weight=1.0, penalty=1 → weight=0.85, penalty=2 → weight=0.70
            weight_factor = max(0.3, 1.0 - (new_penalty * self.WEIGHT_REDUCTION_PER_PENALTY))
            old_weight = self._penalty_weights.get(config_key, 1.0)

            if abs(weight_factor - old_weight) > 0.02:
                self._penalty_weights[config_key] = weight_factor
                self._apply_curriculum_weight(config_key, weight_factor, new_penalty, reason)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # AttributeError: event attribute missing
            # KeyError: missing payload field
            # TypeError: invalid data types
            # ValueError: invalid penalty values
            logger.warning(f"[QualityPenaltyToCurriculumWatcher] Error handling penalty: {e}")

    def _apply_curriculum_weight(
        self,
        config_key: str,
        weight_factor: float,
        penalty: float,
        reason: str,
    ) -> None:
        """Apply weight reduction to CurriculumFeedback."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            current_weight = feedback._current_weights.get(config_key, 1.0)

            # Apply the quality-based weight reduction
            new_weight = max(feedback.weight_min, current_weight * weight_factor)

            if new_weight < current_weight:
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[QualityPenaltyToCurriculumWatcher] Reduced curriculum weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (penalty={penalty:.2f}, reason={reason})"
                )

                # Emit CURRICULUM_REBALANCED event
                self._emit_rebalance_event(config_key, new_weight, penalty)

        except ImportError as e:
            logger.debug(f"[QualityPenaltyToCurriculumWatcher] curriculum_feedback import error: {e}")
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # AttributeError: feedback method missing
            # TypeError: invalid weight types
            # ValueError: invalid weight values
            # KeyError: unknown config_key
            logger.warning(f"[QualityPenaltyToCurriculumWatcher] Error applying weight: {e}")

    def _emit_rebalance_event(
        self,
        config_key: str,
        new_weight: float,
        penalty: float,
    ) -> None:
        """Emit CURRICULUM_REBALANCED event for downstream systems."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "quality_penalty",
                    "changed_configs": [config_key],
                    "new_weights": {config_key: new_weight},
                    "penalty": penalty,
                    "timestamp": time.time(),
                },
                source="quality_penalty_curriculum_watcher",
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: event_router not available
            # AttributeError: router method missing
            # TypeError: invalid event arguments
            # RuntimeError: publish failed
            logger.debug(f"Failed to emit rebalance event: {e}")

    def get_penalty_weights(self) -> dict[str, float]:
        """Get current penalty-based weight factors."""
        return dict(self._penalty_weights)

    def reset_penalty(self, config_key: str) -> None:
        """Reset penalty weight for a config (called when quality recovers)."""
        if config_key in self._penalty_weights:
            del self._penalty_weights[config_key]
            logger.info(f"[QualityPenaltyToCurriculumWatcher] Reset penalty weight for {config_key}")


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

    EXPLORATION_BOOST_FACTOR = 1.3  # Increase temperature by 30% on low quality

    @property
    def LOW_QUALITY_THRESHOLD(self) -> float:
        """Get low quality threshold from centralized config."""
        try:
            from app.config.thresholds import LOW_QUALITY_THRESHOLD
            return LOW_QUALITY_THRESHOLD
        except ImportError:
            return 0.4  # Fallback default

    def __init__(self):
        self._subscribed = False
        self._quality_boosts: dict[str, float] = {}  # config -> boost factor

    def subscribe(self) -> bool:
        """Subscribe to quality events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            # P0.6 Dec 2025: Use DataEventType enum for type-safe subscriptions
            router.subscribe(DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_adjusted)
            router.subscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_updated)
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
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()
            # P0.6 Dec 2025: Use DataEventType enum for type-safe unsubscriptions
            router.unsubscribe(DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_adjusted)
            router.unsubscribe(DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_updated)
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


# =============================================================================
# Unified Wiring
# =============================================================================


def wire_all_feedback_loops(
    enable_momentum_bridge: bool = True,
    enable_pfsp_weakness: bool = True,
    enable_promotion_failed: bool = True,
    enable_quality_penalty: bool = True,
    enable_quality_temperature: bool = True,
    enable_curriculum_feedback: bool = True,
) -> dict[str, Any]:
    """Wire all feedback loop connections at once.

    This is the main entry point for connecting all feedback systems.
    Call this at startup to enable the full self-improvement loop.

    Args:
        enable_momentum_bridge: Enable FeedbackAccelerator → CurriculumFeedback
        enable_pfsp_weakness: Enable PFSP weak opponent → CurriculumFeedback
        enable_promotion_failed: Enable PROMOTION_FAILED → CurriculumFeedback
        enable_quality_penalty: Enable QUALITY_PENALTY_APPLIED → CurriculumFeedback
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


__all__ = [
    # Main wiring functions
    "wire_all_feedback_loops",
    "unwire_all_feedback_loops",
    "get_integration_status",
    # Individual components
    "MomentumToCurriculumBridge",
    "PFSPWeaknessWatcher",
    "PromotionFailedToCurriculumWatcher",
    "QualityPenaltyToCurriculumWatcher",
    "QualityToTemperatureWatcher",
    # Convenience functions
    "get_exploration_boost",
    "get_mastered_opponents",
    "force_momentum_sync",
    "get_quality_penalty_weights",
    "reset_quality_penalty",
    "get_promotion_failure_counts",
    "reset_promotion_failure_count",
]
