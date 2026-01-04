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
import re
import threading
import time
from typing import Any

from app.coordination.curriculum_router import CurriculumSignalBridge
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key
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

            # December 2025: Subscribe to TIER_PROMOTION to adjust curriculum when
            # difficulty tier changes (e.g., advancing from D4 to D5)
            if hasattr(DataEventType, 'TIER_PROMOTION'):
                router.subscribe(DataEventType.TIER_PROMOTION, self._on_tier_promotion)

            # December 29, 2025: Subscribe to CROSSBOARD_PROMOTION to adjust curriculum when
            # a model achieves multi-config promotion (high Elo across multiple configurations)
            if hasattr(DataEventType, 'CROSSBOARD_PROMOTION'):
                router.subscribe(DataEventType.CROSSBOARD_PROMOTION, self._on_crossboard_promotion)

            # December 29, 2025: Subscribe to CURRICULUM_ADVANCEMENT_NEEDED to handle
            # stagnant configs (3+ evaluations with minimal Elo improvement).
            # Emitted by TrainingTriggerDaemon._signal_curriculum_advancement().
            if hasattr(DataEventType, 'CURRICULUM_ADVANCEMENT_NEEDED'):
                router.subscribe(DataEventType.CURRICULUM_ADVANCEMENT_NEEDED, self._on_curriculum_advancement_needed)

            # January 2026 Sprint 10: Subscribe to ELO_VELOCITY_CHANGED for
            # velocity-based curriculum acceleration (+15-25 Elo improvement).
            # When learning is fast (high velocity), accelerate curriculum to capitalize.
            if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                router.subscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed)

            # January 2026 Sprint 10: Subscribe to CURRICULUM_ADVANCED for cross-board
            # propagation (+5-15 Elo). When one config advances, similar configs can benefit.
            if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                router.subscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced)

            # January 2026 Sprint 10: Subscribe to CURRICULUM_PROPAGATE to receive
            # curriculum advancements propagated from similar configs.
            if hasattr(DataEventType, 'CURRICULUM_PROPAGATE'):
                router.subscribe(DataEventType.CURRICULUM_PROPAGATE, self._on_curriculum_propagate)

            # January 2026 Sprint 10: Subscribe to REGRESSION_DETECTED for direct
            # curriculum response (+12-18 Elo). Previously took 2-3 cycles through
            # intermediate handlers. Direct subscription enables immediate difficulty
            # reduction when regression is detected.
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                router.subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)

            # January 2026 Sprint 12: Subscribe to TRAINING_LOSS_ANOMALY for direct
            # curriculum response (+10-15 Elo). Loss anomalies indicate training data
            # quality issues. Reducing curriculum weight for affected configs prevents
            # learning from bad data.
            if hasattr(DataEventType, 'TRAINING_LOSS_ANOMALY'):
                router.subscribe(DataEventType.TRAINING_LOSS_ANOMALY, self._on_loss_anomaly)

            logger.info("[MomentumToCurriculumBridge] Subscribed to EVALUATION_COMPLETED, SELFPLAY_RATE_CHANGED, ELO_SIGNIFICANT_CHANGE, SELFPLAY_ALLOCATION_UPDATED, MODEL_PROMOTED, TIER_PROMOTION, CROSSBOARD_PROMOTION, CURRICULUM_ADVANCEMENT_NEEDED, ELO_VELOCITY_CHANGED, CURRICULUM_ADVANCED, CURRICULUM_PROPAGATE, REGRESSION_DETECTED, TRAINING_LOSS_ANOMALY")

            # December 29, 2025: Only set _event_subscribed = True after successful subscription
            # Previously this was in finally block which caused race condition:
            # - If subscription failed, _event_subscribed was still True
            # - Next call would skip re-subscription, events silently missed
            self._event_subscribed = True
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            # ImportError: modules not available
            # AttributeError: router method missing
            # TypeError: invalid subscription arguments
            # RuntimeError: subscription failed
            logger.debug(f"[MomentumToCurriculumBridge] Event subscription failed: {e}")
            # Note: _event_subscribed stays False on failure, allowing retry
            return False

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
                # December 2025: Unsubscribe from TIER_PROMOTION
                if hasattr(DataEventType, 'TIER_PROMOTION'):
                    router.unsubscribe(DataEventType.TIER_PROMOTION, self._on_tier_promotion)
                # December 29, 2025: Unsubscribe from CROSSBOARD_PROMOTION
                if hasattr(DataEventType, 'CROSSBOARD_PROMOTION'):
                    router.unsubscribe(DataEventType.CROSSBOARD_PROMOTION, self._on_crossboard_promotion)
                # December 29, 2025: Unsubscribe from CURRICULUM_ADVANCEMENT_NEEDED
                if hasattr(DataEventType, 'CURRICULUM_ADVANCEMENT_NEEDED'):
                    router.unsubscribe(DataEventType.CURRICULUM_ADVANCEMENT_NEEDED, self._on_curriculum_advancement_needed)
                # January 2026 Sprint 10: Unsubscribe from ELO_VELOCITY_CHANGED
                if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                    router.unsubscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed)
                # January 2026 Sprint 10: Unsubscribe from CURRICULUM_ADVANCED
                if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                    router.unsubscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced)
                # January 2026 Sprint 10: Unsubscribe from CURRICULUM_PROPAGATE
                if hasattr(DataEventType, 'CURRICULUM_PROPAGATE'):
                    router.unsubscribe(DataEventType.CURRICULUM_PROPAGATE, self._on_curriculum_propagate)
                # January 2026 Sprint 10: Unsubscribe from REGRESSION_DETECTED
                if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                    router.unsubscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
                # January 2026 Sprint 12: Unsubscribe from TRAINING_LOSS_ANOMALY
                if hasattr(DataEventType, 'TRAINING_LOSS_ANOMALY'):
                    router.unsubscribe(DataEventType.TRAINING_LOSS_ANOMALY, self._on_loss_anomaly)
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

            config_key = extract_config_key(payload)
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

            config_key = extract_config_key(payload)
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

        Note: This handler only tracks allocation patterns, it does NOT emit
        CURRICULUM_REBALANCED events, so there is no loop risk.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            # Note: Loop guard not needed here since this handler doesn't emit events
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

            config_key = make_config_key(board_type, num_players)

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

    def _on_tier_promotion(self, event) -> None:
        """Handle TIER_PROMOTION event - adjust curriculum for difficulty tier changes.

        December 2025: When a model advances to a new difficulty tier, curriculum
        weights should be adjusted to:
        - Increase exploration for the newly promoted tier
        - Reduce focus on mastered lower tiers
        - Maintain training diversity across all tiers
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            config_key = extract_config_key(payload)
            old_tier = payload.get("old_tier", "")
            new_tier = payload.get("new_tier", "")
            elo = payload.get("elo", 0.0)
            win_rate = payload.get("win_rate", 0.0)

            if not config_key or not new_tier:
                return

            logger.info(
                f"[MomentumToCurriculumBridge] TIER_PROMOTION: {config_key} "
                f"{old_tier} -> {new_tier}, elo={elo:.0f}, win_rate={win_rate:.1%}"
            )

            # Trigger full curriculum weight sync after tier promotion
            # The new tier represents higher skill level and needs rebalanced training
            self._sync_weights()

            # Emit curriculum advanced event for downstream consumers
            try:
                from app.coordination.event_emitters import emit_curriculum_advanced
                import asyncio

                async def _emit():
                    await emit_curriculum_advanced(
                        config_key=config_key,
                        old_tier=old_tier,
                        new_tier=new_tier,
                        trigger="tier_promotion",
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
            logger.warning(f"[MomentumToCurriculumBridge] Error handling tier promotion: {e}")

    def _on_crossboard_promotion(self, event) -> None:
        """Handle CROSSBOARD_PROMOTION event - adjust curriculum for multi-config achievements.

        December 29, 2025: When a model achieves high Elo across multiple configurations
        (crossboard promotion), curriculum should be adjusted to:
        - Celebrate the milestone with reduced training intensity
        - Shift focus to configurations that haven't achieved crossboard status
        - Balance exploration across all configurations
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else {}

            model_id = payload.get("model_id", payload.get("model", ""))
            configs = payload.get("configs", [])
            avg_elo = payload.get("avg_elo", 0.0)
            min_elo = payload.get("min_elo", 0.0)
            timestamp = payload.get("timestamp", 0.0)

            if not model_id:
                return

            logger.info(
                f"[MomentumToCurriculumBridge] CROSSBOARD_PROMOTION: {model_id} "
                f"achieved Elo >= {min_elo:.0f} across {len(configs)} configs, "
                f"avg_elo={avg_elo:.0f}"
            )

            # Trigger full curriculum weight sync after crossboard promotion
            # This ensures training resources are rebalanced across all configs
            self._sync_weights()

            # Emit curriculum advanced event for downstream consumers
            try:
                from app.coordination.event_emitters import emit_curriculum_advanced
                import asyncio

                async def _emit():
                    await emit_curriculum_advanced(
                        config_key=",".join(configs) if configs else "crossboard",
                        old_tier="pre_crossboard",
                        new_tier="crossboard_achieved",
                        trigger="crossboard_promotion",
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
            logger.warning(f"[MomentumToCurriculumBridge] Error handling crossboard promotion: {e}")

    def _on_curriculum_advancement_needed(self, event) -> None:
        """Handle CURRICULUM_ADVANCEMENT_NEEDED event - advance curriculum for stagnant configs.

        December 29, 2025: When a config has 3+ evaluations with minimal Elo improvement,
        TrainingTriggerDaemon signals that curriculum should advance. This handler:
        1. Increases opponent difficulty for the config
        2. Boosts exploration temperature to encourage novelty
        3. Adjusts curriculum weight to prioritize the stagnant config
        4. Emits CURRICULUM_ADVANCED event for downstream consumers

        This closes the feedback loop: stagnant Elo → harder curriculum → fresh training signal.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = extract_config_key(payload)
            reason = payload.get("reason", "unknown")
            timestamp = payload.get("timestamp", 0.0)

            if not config_key:
                logger.debug("[MomentumToCurriculumBridge] CURRICULUM_ADVANCEMENT_NEEDED without config_key")
                return

            logger.info(
                f"[MomentumToCurriculumBridge] CURRICULUM_ADVANCEMENT_NEEDED: {config_key}, "
                f"reason={reason}"
            )

            # 1. Increase opponent difficulty by boosting curriculum weight
            # This prioritizes training against stronger opponents for the stagnant config
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                curriculum = get_curriculum_feedback()
                if curriculum:
                    # Increase weight for stagnant config to force harder training
                    current_weights = curriculum.get_curriculum_weights()
                    old_weight = current_weights.get(config_key, 1.0)

                    # Boost by 30% to prioritize this config
                    new_weight = min(old_weight * 1.3, 2.0)  # Cap at 2.0x

                    curriculum.update_weight(
                        config_key=config_key,
                        new_weight=new_weight,
                        source="curriculum_advancement",
                    )
                    logger.info(
                        f"[MomentumToCurriculumBridge] Boosted curriculum weight for {config_key}: "
                        f"{old_weight:.2f} -> {new_weight:.2f}"
                    )
            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not update curriculum weight: {e}")

            # 2. Boost exploration temperature to encourage novel game states
            try:
                from app.training.temperature_scheduling import boost_exploration_for_config

                boost_exploration_for_config(config_key, boost_factor=1.2, duration_games=500)
                logger.info(
                    f"[MomentumToCurriculumBridge] Boosted exploration temperature for {config_key}"
                )
            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not boost exploration: {e}")

            # 3. Trigger full weight sync to propagate changes
            self._sync_weights()

            # 4. Emit CURRICULUM_ADVANCED to signal downstream consumers
            try:
                from app.coordination.event_emitters import emit_curriculum_advanced
                import asyncio

                async def _emit():
                    await emit_curriculum_advanced(
                        config_key=config_key,
                        old_tier="stagnant",
                        new_tier="advancing",
                        trigger=f"curriculum_advancement_{reason}",
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

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling curriculum advancement: {e}")

    def _on_elo_velocity_changed(self, event) -> None:
        """Handle ELO_VELOCITY_CHANGED event - accelerate curriculum on high velocity.

        January 2026 Sprint 10: Velocity-based curriculum acceleration (+15-25 Elo).
        When learning is fast (high Elo velocity), accelerate curriculum to capitalize:
        - High velocity (>10 Elo/hr): Boost curriculum weight by 20%
        - Very high velocity (>20 Elo/hr): Boost by 35%, emit fast-track event

        This closes the feedback loop: Fast learning → Accelerated curriculum → More challenge
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = extract_config_key(payload)
            velocity = payload.get("velocity", 0.0)
            previous_velocity = payload.get("previous_velocity", 0.0)
            trend = payload.get("trend", "stable")

            if not config_key:
                return

            # Velocity thresholds for curriculum acceleration
            HIGH_VELOCITY_THRESHOLD = 10.0  # Elo/hour
            VERY_HIGH_VELOCITY_THRESHOLD = 20.0  # Elo/hour
            BOOST_HIGH = 1.20  # 20% boost
            BOOST_VERY_HIGH = 1.35  # 35% boost

            # Only accelerate on positive velocity with accelerating trend
            if velocity <= 0 or trend == "decelerating":
                return

            # Determine boost level
            if velocity >= VERY_HIGH_VELOCITY_THRESHOLD:
                boost_multiplier = BOOST_VERY_HIGH
                acceleration_level = "fast_track"
            elif velocity >= HIGH_VELOCITY_THRESHOLD:
                boost_multiplier = BOOST_HIGH
                acceleration_level = "accelerated"
            else:
                # Below threshold - no acceleration needed
                return

            logger.info(
                f"[MomentumToCurriculumBridge] Velocity acceleration for {config_key}: "
                f"velocity={velocity:.1f} Elo/hr, level={acceleration_level}"
            )

            # Apply curriculum weight boost
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                curriculum = get_curriculum_feedback()
                if curriculum:
                    current_weights = curriculum.get_curriculum_weights()
                    old_weight = current_weights.get(config_key, 1.0)

                    # Apply boost, cap at 2.5x
                    new_weight = min(old_weight * boost_multiplier, 2.5)

                    if new_weight > old_weight:
                        curriculum.update_weight(
                            config_key=config_key,
                            new_weight=new_weight,
                            source=f"velocity_acceleration_{acceleration_level}",
                        )
                        self._last_weights[config_key] = new_weight

                        logger.info(
                            f"[MomentumToCurriculumBridge] Boosted curriculum weight for {config_key}: "
                            f"{old_weight:.2f} -> {new_weight:.2f} (velocity={velocity:.1f})"
                        )

                        # Emit curriculum rebalanced event
                        self._emit_rebalance_event([config_key], {config_key: new_weight})

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not update curriculum weight: {e}")

            # For very high velocity, also emit curriculum advanced event
            if acceleration_level == "fast_track":
                try:
                    from app.coordination.event_emitters import emit_curriculum_advanced
                    import asyncio

                    async def _emit():
                        await emit_curriculum_advanced(
                            config_key=config_key,
                            old_tier="standard",
                            new_tier="fast_track",
                            trigger=f"velocity_acceleration_{velocity:.0f}_elo_hr",
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

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling Elo velocity: {e}")

    def _on_curriculum_advanced(self, event) -> None:
        """Handle CURRICULUM_ADVANCED event - propagate to similar configs.

        January 2026 Sprint 10: Cross-board curriculum propagation (+5-15 Elo).
        When one config (e.g., square8_2p) achieves a curriculum advancement,
        similar configs (e.g., hex8_2p, hexagonal_2p) can benefit from reduced
        exploration and accelerated progression.

        Similarity is based on:
        - Same number of players (required)
        - Similar board types (hex->hex, square->square preferred)
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = extract_config_key(payload)
            old_tier = payload.get("old_tier", "")
            new_tier = payload.get("new_tier", "")
            trigger = payload.get("trigger", "unknown")
            source = payload.get("source", "")

            if not config_key:
                return

            # Don't propagate propagated events (prevent infinite loops)
            if source == "curriculum_propagation":
                return

            # Don't propagate crossboard_promotion events (already affect multiple configs)
            if "crossboard" in trigger.lower() or "crossboard" in new_tier.lower():
                return

            logger.info(
                f"[MomentumToCurriculumBridge] CURRICULUM_ADVANCED: {config_key} "
                f"{old_tier} -> {new_tier} (trigger={trigger}), checking for propagation"
            )

            # Find similar configs to propagate to
            similar_configs = self._get_similar_configs(config_key)

            if not similar_configs:
                logger.debug(f"[MomentumToCurriculumBridge] No similar configs for {config_key}")
                return

            # Emit CURRICULUM_PROPAGATE for each similar config
            for target_config in similar_configs:
                self._emit_curriculum_propagate(
                    source_config=config_key,
                    target_config=target_config,
                    advancement_tier=new_tier,
                    original_trigger=trigger,
                )

            logger.info(
                f"[MomentumToCurriculumBridge] Propagated curriculum advancement from "
                f"{config_key} to {len(similar_configs)} similar configs: {similar_configs}"
            )

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling curriculum advanced: {e}")

    def _on_curriculum_propagate(self, event) -> None:
        """Handle CURRICULUM_PROPAGATE event - apply propagated advancement.

        January 2026 Sprint 10: Receive curriculum advancement from similar config.
        Apply a reduced boost (50% of normal) to benefit from cross-board learning.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            source_config = payload.get("source_config", "")
            target_config = payload.get("target_config", "")
            advancement_tier = payload.get("advancement_tier", "")
            propagation_weight = payload.get("propagation_weight", 0.5)

            if not target_config:
                return

            logger.info(
                f"[MomentumToCurriculumBridge] Received CURRICULUM_PROPAGATE: "
                f"{source_config} -> {target_config}, tier={advancement_tier}"
            )

            # Apply curriculum weight boost (reduced compared to direct advancement)
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                curriculum = get_curriculum_feedback()
                if curriculum:
                    current_weights = curriculum.get_curriculum_weights()
                    old_weight = current_weights.get(target_config, 1.0)

                    # Apply propagation boost (50% of normal boost = 10% weight increase)
                    boost_multiplier = 1.0 + (0.20 * propagation_weight)  # 10% boost at 0.5 weight
                    new_weight = min(old_weight * boost_multiplier, 2.0)  # Cap at 2.0

                    if new_weight > old_weight:
                        curriculum.update_weight(
                            config_key=target_config,
                            new_weight=new_weight,
                            source=f"propagation_from_{source_config}",
                        )
                        self._last_weights[target_config] = new_weight

                        logger.info(
                            f"[MomentumToCurriculumBridge] Applied propagated boost to {target_config}: "
                            f"{old_weight:.2f} -> {new_weight:.2f} (from {source_config})"
                        )

                        # Emit curriculum rebalanced event
                        self._emit_rebalance_event([target_config], {target_config: new_weight})

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not apply propagation: {e}")

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling curriculum propagate: {e}")

    def _on_regression_detected(self, event) -> None:
        """Handle REGRESSION_DETECTED event - immediately reduce curriculum difficulty.

        January 2026 Sprint 10: Direct subscription to REGRESSION_DETECTED enables
        immediate curriculum response (+12-18 Elo improvement). Previously the signal
        flowed through 2-3 intermediate handlers (FeedbackLoop → TrainingCoordinator
        → CurriculumFeedback), causing 2-3 cycle delays before curriculum adjustment.

        Action: Reduce curriculum weight by 30-50% depending on regression severity.
        This reduces difficulty level so the model can recover from the regression.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = payload.get("config_key", "")
            elo_delta = payload.get("elo_delta", 0)
            current_elo = payload.get("current_elo", 0)
            previous_elo = payload.get("previous_elo", 0)

            if not config_key:
                logger.debug("[MomentumToCurriculumBridge] REGRESSION_DETECTED missing config_key")
                return

            # Calculate regression severity: larger drop = more aggressive weight reduction
            # -50 Elo: 30% reduction, -100 Elo: 50% reduction, -150+ Elo: 60% reduction
            severity = min(abs(elo_delta), 150) / 150.0  # Normalize to 0-1
            reduction_factor = 0.70 - (severity * 0.30)  # 0.70 to 0.40 based on severity

            logger.warning(
                f"[MomentumToCurriculumBridge] REGRESSION_DETECTED for {config_key}: "
                f"Elo {previous_elo} -> {current_elo} (delta={elo_delta}), "
                f"reducing curriculum weight by {(1-reduction_factor)*100:.0f}%"
            )

            # Apply curriculum weight reduction
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                curriculum = get_curriculum_feedback()
                if curriculum:
                    current_weights = curriculum.get_curriculum_weights()
                    old_weight = current_weights.get(config_key, 1.0)

                    # Apply reduction, but maintain minimum weight of 0.3
                    new_weight = max(old_weight * reduction_factor, 0.3)

                    if new_weight < old_weight:
                        curriculum.update_weight(
                            config_key=config_key,
                            new_weight=new_weight,
                            source=f"regression_detected_elo_delta_{elo_delta}",
                        )
                        self._last_weights[config_key] = new_weight

                        logger.info(
                            f"[MomentumToCurriculumBridge] Reduced curriculum weight for {config_key}: "
                            f"{old_weight:.2f} -> {new_weight:.2f} (regression recovery)"
                        )

                        # Emit curriculum rebalanced event
                        self._emit_rebalance_event([config_key], {config_key: new_weight})

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not apply regression adjustment: {e}")

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling regression detected: {e}")

    def _on_loss_anomaly(self, event) -> None:
        """Handle TRAINING_LOSS_ANOMALY event - reduce curriculum weight for affected config.

        January 2026 Sprint 12: Direct subscription to TRAINING_LOSS_ANOMALY enables
        immediate curriculum response (+10-15 Elo improvement). Loss anomalies indicate
        training data quality issues. Reducing curriculum weight for affected configs
        prevents learning from bad data and redirects resources to healthier configs.

        Action: Reduce curriculum weight by 20-40% depending on anomaly severity.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = payload.get("config_key", "")
            severity = payload.get("severity", "moderate")  # mild, moderate, severe
            loss_value = payload.get("loss_value", 0)
            expected_loss = payload.get("expected_loss", 0)
            anomaly_type = payload.get("type", "spike")  # spike, drop, nan

            if not config_key:
                logger.debug("[MomentumToCurriculumBridge] TRAINING_LOSS_ANOMALY missing config_key")
                return

            # Calculate reduction factor based on severity
            # mild: 15% reduction, moderate: 25% reduction, severe: 40% reduction
            severity_map = {
                "mild": 0.85,
                "moderate": 0.75,
                "severe": 0.60,
            }
            reduction_factor = severity_map.get(severity, 0.75)

            logger.warning(
                f"[MomentumToCurriculumBridge] TRAINING_LOSS_ANOMALY for {config_key}: "
                f"severity={severity}, type={anomaly_type}, loss={loss_value:.4f} (expected={expected_loss:.4f}), "
                f"reducing curriculum weight by {(1-reduction_factor)*100:.0f}%"
            )

            # Apply curriculum weight reduction
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                curriculum = get_curriculum_feedback()
                if curriculum:
                    current_weights = curriculum.get_curriculum_weights()
                    old_weight = current_weights.get(config_key, 1.0)

                    # Apply reduction, but maintain minimum weight of 0.25
                    # (lower floor than regression since data quality issues are more severe)
                    new_weight = max(old_weight * reduction_factor, 0.25)

                    if new_weight < old_weight:
                        curriculum.update_weight(
                            config_key=config_key,
                            new_weight=new_weight,
                            source=f"loss_anomaly_{severity}_{anomaly_type}",
                        )
                        self._last_weights[config_key] = new_weight

                        logger.info(
                            f"[MomentumToCurriculumBridge] Reduced curriculum weight for {config_key}: "
                            f"{old_weight:.2f} -> {new_weight:.2f} (loss anomaly recovery)"
                        )

                        # Emit curriculum rebalanced event
                        self._emit_rebalance_event([config_key], {config_key: new_weight})

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not apply loss anomaly adjustment: {e}")

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling loss anomaly: {e}")

    def _get_similar_configs(self, config_key: str) -> list[str]:
        """Get similar configs for cross-board curriculum propagation.

        January 2026 Sprint 10: Identifies configs that can benefit from
        curriculum advancements in a source config.

        Similarity criteria:
        1. Same number of players (required)
        2. Same board family preferred (hex->hex, square->square)
        3. Excludes the source config itself

        Args:
            config_key: Source config key (e.g., "square8_2p")

        Returns:
            List of similar config keys to propagate to
        """
        parsed = parse_config_key(config_key)
        if not parsed or not parsed.num_players:
            return []

        source_board = parsed.board_type or ""
        source_players = parsed.num_players

        # Determine board family
        if source_board.startswith("hex"):
            source_family = "hex"
        elif source_board.startswith("square"):
            source_family = "square"
        else:
            source_family = source_board

        # All known board configurations
        ALL_BOARDS = ["hex8", "hexagonal", "square8", "square19"]
        ALL_PLAYERS = [2, 3, 4]

        similar_configs = []
        for board in ALL_BOARDS:
            for players in ALL_PLAYERS:
                candidate_key = f"{board}_{players}p"

                # Skip source config
                if candidate_key == config_key:
                    continue

                # Require same number of players
                if players != source_players:
                    continue

                # Prefer same board family, but include all same-player configs
                if board.startswith("hex"):
                    candidate_family = "hex"
                elif board.startswith("square"):
                    candidate_family = "square"
                else:
                    candidate_family = board

                # Include same-family configs with higher priority
                # (for now, include all same-player configs)
                similar_configs.append(candidate_key)

        return similar_configs

    def _emit_curriculum_propagate(
        self,
        source_config: str,
        target_config: str,
        advancement_tier: str,
        original_trigger: str,
    ) -> None:
        """Emit CURRICULUM_PROPAGATE event for cross-board propagation.

        January 2026 Sprint 10: Signals that a curriculum advancement
        should be propagated to a similar config.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                router.publish_sync(
                    "CURRICULUM_PROPAGATE",
                    {
                        "source_config": source_config,
                        "target_config": target_config,
                        "advancement_tier": advancement_tier,
                        "original_trigger": original_trigger,
                        "propagation_weight": 0.5,  # 50% weight for propagated boosts
                        "timestamp": time.time(),
                    },
                    source="curriculum_integration",
                )
                logger.debug(
                    f"[MomentumToCurriculumBridge] Emitted CURRICULUM_PROPAGATE: "
                    f"{source_config} -> {target_config}"
                )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Failed to emit curriculum propagate event: {e}")

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
        """Sync weights from FeedbackAccelerator to CurriculumFeedback.

        January 2026 Sprint 10: Enhanced with quality-weighted curriculum adjustment.
        Combines quality scores from QualityMonitorDaemon with momentum weights
        from FeedbackAccelerator to produce final curriculum weights.

        Quality adjustment:
        - High quality (>0.8): +15% weight boost (capitalize on good data)
        - Medium quality (0.5-0.8): no change
        - Low quality (<0.5): -20% weight reduction (focus elsewhere)

        Expected improvement: +12-18 Elo from better quality/curriculum alignment.
        """
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator
            from app.training.curriculum_feedback import get_curriculum_feedback

            accelerator = get_feedback_accelerator()
            feedback = get_curriculum_feedback()

            # Get momentum-based weights from accelerator
            accelerator_weights = accelerator.get_curriculum_weights()

            if not accelerator_weights:
                return

            # Sprint 10: Apply quality-weighted adjustment to momentum weights
            quality_adjusted_weights = self._apply_quality_adjustment(accelerator_weights)

            # Check for significant changes
            changed_configs = []
            for config_key, new_weight in quality_adjusted_weights.items():
                old_weight = self._last_weights.get(config_key, 1.0)
                if abs(new_weight - old_weight) > 0.1:
                    changed_configs.append(config_key)

            if not changed_configs:
                return

            # Update CurriculumFeedback weights with quality-adjusted values
            for config_key, weight in quality_adjusted_weights.items():
                feedback._current_weights[config_key] = weight

            self._last_weights = dict(quality_adjusted_weights)
            self._last_sync_time = time.time()  # Track last sync for health_check

            # Emit event
            self._emit_rebalance_event(changed_configs, quality_adjusted_weights)

            logger.info(
                f"[MomentumToCurriculumBridge] Synced {len(changed_configs)} weight changes: "
                f"{', '.join(changed_configs)}"
            )

        except ImportError as e:
            logger.debug(f"[MomentumToCurriculumBridge] Import error: {e}")

    def _apply_quality_adjustment(
        self, momentum_weights: dict[str, float]
    ) -> dict[str, float]:
        """Apply quality-based adjustment to momentum weights.

        January 2026 Sprint 10: Combines quality + Elo momentum for curriculum.
        High-quality configs get boosted, low-quality configs get reduced.

        Args:
            momentum_weights: Momentum-based weights from FeedbackAccelerator

        Returns:
            Quality-adjusted weights
        """
        try:
            from app.coordination.quality_monitor_daemon import get_quality_monitor

            quality_monitor = get_quality_monitor()
            adjusted_weights: dict[str, float] = {}

            for config_key, momentum_weight in momentum_weights.items():
                quality = quality_monitor.get_quality_for_config(config_key)

                if quality is None:
                    # No quality data available, use momentum weight as-is
                    adjusted_weights[config_key] = momentum_weight
                    continue

                # Apply quality multiplier
                quality_multiplier = self._get_quality_multiplier(quality)
                adjusted_weight = momentum_weight * quality_multiplier

                # Clamp to reasonable bounds (0.1 to 3.0)
                adjusted_weight = max(0.1, min(3.0, adjusted_weight))
                adjusted_weights[config_key] = adjusted_weight

                # Log significant adjustments
                if abs(quality_multiplier - 1.0) > 0.05:
                    logger.debug(
                        f"[MomentumToCurriculumBridge] Quality adjustment for {config_key}: "
                        f"quality={quality:.2f}, multiplier={quality_multiplier:.2f}, "
                        f"weight {momentum_weight:.2f} → {adjusted_weight:.2f}"
                    )

            return adjusted_weights

        except ImportError:
            # QualityMonitorDaemon not available, return original weights
            return dict(momentum_weights)
        except Exception as e:
            logger.debug(f"[MomentumToCurriculumBridge] Quality adjustment error: {e}")
            return dict(momentum_weights)

    def _get_quality_multiplier(self, quality: float) -> float:
        """Get weight multiplier based on quality score.

        January 2026 Sprint 10: Quality-to-multiplier mapping.

        Args:
            quality: Quality score (0.0 to 1.0)

        Returns:
            Weight multiplier:
            - Quality >= 0.8: 1.0 to 1.15 (linear boost)
            - Quality 0.5-0.8: 1.0 (no change)
            - Quality < 0.5: 0.8 to 1.0 (linear reduction)
        """
        if quality >= 0.8:
            # High quality: boost weight by up to 15%
            # quality 0.8 → 1.0, quality 1.0 → 1.15
            return 1.0 + (quality - 0.8) * 0.75  # (0.2 * 0.75 = 0.15 max boost)
        elif quality >= 0.5:
            # Medium quality: no change
            return 1.0
        else:
            # Low quality: reduce weight by up to 20%
            # quality 0.5 → 1.0, quality 0.0 → 0.8
            return 0.8 + quality * 0.4  # (0.5 * 0.4 = 0.2 when quality = 0.5)

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

    def health_check(self) -> "HealthCheckResult":
        """Check watcher health for DaemonManager integration.

        December 2025: Added for unified health monitoring.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

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


# =============================================================================
# 2.5. QUALITY_PENALTY_APPLIED → Curriculum Weight Reduction
# =============================================================================


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


# =============================================================================
# 2.5.1. PROMOTION_COMPLETED → Curriculum Advancement/Regression (December 29, 2025)
# =============================================================================


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
            from app.coordination.event_router import get_router

            router = get_router()
            if router is None:
                logger.debug("[PromotionCompletedToCurriculumWatcher] Event router not available")
                return False

            # Subscribe to PROMOTION_COMPLETED (string type, as emitted by auto_promotion_daemon)
            router.subscribe("PROMOTION_COMPLETED", self._on_promotion_completed)
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
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                router.unsubscribe("PROMOTION_COMPLETED", self._on_promotion_completed)
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
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
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
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

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


# =============================================================================
# 2.4.1. REGRESSION_CRITICAL → Curriculum Weight Boost (December 27, 2025)
# =============================================================================


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


# =============================================================================
# 2.5. QUALITY_PENALTY_APPLIED → Curriculum Weight Reduction
# =============================================================================


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

    def health_check(self) -> "HealthCheckResult":
        """Check watcher health for DaemonManager integration."""
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

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
]
