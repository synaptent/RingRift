"""Momentum-to-curriculum bridge extracted from curriculum_integration."""

from __future__ import annotations

import logging
import threading
import time

from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


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
        # Session 17.25: Track weights BEFORE boost for rollback capability
        self._pre_boost_weights: dict[str, float] = {}

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
            from app.coordination.event_router import DataEventType, subscribe

            # Use enum directly (router normalizes both enum and .value)
            subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)

            # Phase 21.2: Subscribe to SELFPLAY_RATE_CHANGED for Elo momentum → curriculum sync
            if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                subscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)

            # December 2025: Subscribe to ELO_SIGNIFICANT_CHANGE for curriculum rebalance triggers
            if hasattr(DataEventType, 'ELO_SIGNIFICANT_CHANGE'):
                subscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, self._on_elo_significant_change)

            # December 2025: Subscribe to SELFPLAY_ALLOCATION_UPDATED to track allocation changes
            if hasattr(DataEventType, 'SELFPLAY_ALLOCATION_UPDATED'):
                subscribe(DataEventType.SELFPLAY_ALLOCATION_UPDATED, self._on_selfplay_allocation_updated)

            # December 2025 Phase 2: Subscribe to MODEL_PROMOTED to rebalance curriculum
            # when a new model is promoted. This ensures curriculum weights are adjusted
            # based on the latest model strength.
            if hasattr(DataEventType, 'MODEL_PROMOTED'):
                subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)

            # December 2025: Subscribe to TIER_PROMOTION to adjust curriculum when
            # difficulty tier changes (e.g., advancing from D4 to D5)
            if hasattr(DataEventType, 'TIER_PROMOTION'):
                subscribe(DataEventType.TIER_PROMOTION, self._on_tier_promotion)

            # December 29, 2025: Subscribe to CROSSBOARD_PROMOTION to adjust curriculum when
            # a model achieves multi-config promotion (high Elo across multiple configurations)
            if hasattr(DataEventType, 'CROSSBOARD_PROMOTION'):
                subscribe(DataEventType.CROSSBOARD_PROMOTION, self._on_crossboard_promotion)

            # December 29, 2025: Subscribe to CURRICULUM_ADVANCEMENT_NEEDED to handle
            # stagnant configs (3+ evaluations with minimal Elo improvement).
            # Emitted by TrainingTriggerDaemon._signal_curriculum_advancement().
            if hasattr(DataEventType, 'CURRICULUM_ADVANCEMENT_NEEDED'):
                subscribe(DataEventType.CURRICULUM_ADVANCEMENT_NEEDED, self._on_curriculum_advancement_needed)

            # January 2026 Sprint 10: Subscribe to ELO_VELOCITY_CHANGED for
            # velocity-based curriculum acceleration (+15-25 Elo improvement).
            # When learning is fast (high velocity), accelerate curriculum to capitalize.
            if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                subscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed)

            # January 2026 Sprint 10: Subscribe to CURRICULUM_ADVANCED for cross-board
            # propagation (+5-15 Elo). When one config advances, similar configs can benefit.
            if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                subscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced)

            # January 2026 Sprint 10: Subscribe to CURRICULUM_PROPAGATE to receive
            # curriculum advancements propagated from similar configs.
            if hasattr(DataEventType, 'CURRICULUM_PROPAGATE'):
                subscribe(DataEventType.CURRICULUM_PROPAGATE, self._on_curriculum_propagate)

            # January 2026 Sprint 10: Subscribe to REGRESSION_DETECTED for direct
            # curriculum response (+12-18 Elo). Previously took 2-3 cycles through
            # intermediate handlers. Direct subscription enables immediate difficulty
            # reduction when regression is detected.
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                subscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)

            # January 2026 Sprint 12: Subscribe to TRAINING_LOSS_ANOMALY for direct
            # curriculum response (+10-15 Elo). Loss anomalies indicate training data
            # quality issues. Reducing curriculum weight for affected configs prevents
            # learning from bad data.
            if hasattr(DataEventType, 'TRAINING_LOSS_ANOMALY'):
                subscribe(DataEventType.TRAINING_LOSS_ANOMALY, self._on_loss_anomaly)

            # January 2026 Sprint 12: Subscribe to QUORUM_RECOVERY_STARTED for
            # curriculum adjustment during quorum recovery. When quorum is lost and
            # recovery starts, boost selfplay priority for affected configs to help
            # the cluster recover faster with fresh training data.
            if hasattr(DataEventType, 'QUORUM_RECOVERY_STARTED'):
                subscribe(DataEventType.QUORUM_RECOVERY_STARTED, self._on_quorum_recovery)

            # Session 17.25: Subscribe to CURRICULUM_ROLLBACK to restore prior weights
            # after regression is detected. This undoes curriculum boosts that led to
            # regressions, allowing the model to recover.
            if hasattr(DataEventType, 'CURRICULUM_ROLLBACK'):
                subscribe(DataEventType.CURRICULUM_ROLLBACK, self._on_curriculum_rollback)

            # Jan 2026 P1: Subscribe to OPPONENT_DIVERSITY_NEEDED for stall escalation.
            # When a config has been stalled for 48+ hours, inject opponent diversity
            # to help break through local optimum.
            try:
                subscribe("OPPONENT_DIVERSITY_NEEDED", self._on_opponent_diversity_needed)
            except (AttributeError, TypeError):
                logger.debug("[MomentumToCurriculumBridge] OPPONENT_DIVERSITY_NEEDED not available")

            logger.info("[MomentumToCurriculumBridge] Subscribed to EVALUATION_COMPLETED, SELFPLAY_RATE_CHANGED, ELO_SIGNIFICANT_CHANGE, SELFPLAY_ALLOCATION_UPDATED, MODEL_PROMOTED, TIER_PROMOTION, CROSSBOARD_PROMOTION, CURRICULUM_ADVANCEMENT_NEEDED, ELO_VELOCITY_CHANGED, CURRICULUM_ADVANCED, CURRICULUM_PROPAGATE, REGRESSION_DETECTED, TRAINING_LOSS_ANOMALY, QUORUM_RECOVERY_STARTED, CURRICULUM_ROLLBACK, OPPONENT_DIVERSITY_NEEDED")

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
            from app.coordination.event_router import DataEventType, unsubscribe

            unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
            if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                unsubscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)
            if hasattr(DataEventType, 'ELO_SIGNIFICANT_CHANGE'):
                unsubscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, self._on_elo_significant_change)
            if hasattr(DataEventType, 'SELFPLAY_ALLOCATION_UPDATED'):
                unsubscribe(DataEventType.SELFPLAY_ALLOCATION_UPDATED, self._on_selfplay_allocation_updated)
            if hasattr(DataEventType, 'MODEL_PROMOTED'):
                unsubscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            if hasattr(DataEventType, 'TIER_PROMOTION'):
                unsubscribe(DataEventType.TIER_PROMOTION, self._on_tier_promotion)
            if hasattr(DataEventType, 'CROSSBOARD_PROMOTION'):
                unsubscribe(DataEventType.CROSSBOARD_PROMOTION, self._on_crossboard_promotion)
            if hasattr(DataEventType, 'CURRICULUM_ADVANCEMENT_NEEDED'):
                unsubscribe(DataEventType.CURRICULUM_ADVANCEMENT_NEEDED, self._on_curriculum_advancement_needed)
            if hasattr(DataEventType, 'ELO_VELOCITY_CHANGED'):
                unsubscribe(DataEventType.ELO_VELOCITY_CHANGED, self._on_elo_velocity_changed)
            if hasattr(DataEventType, 'CURRICULUM_ADVANCED'):
                unsubscribe(DataEventType.CURRICULUM_ADVANCED, self._on_curriculum_advanced)
            if hasattr(DataEventType, 'CURRICULUM_PROPAGATE'):
                unsubscribe(DataEventType.CURRICULUM_PROPAGATE, self._on_curriculum_propagate)
            if hasattr(DataEventType, 'REGRESSION_DETECTED'):
                unsubscribe(DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
            if hasattr(DataEventType, 'TRAINING_LOSS_ANOMALY'):
                unsubscribe(DataEventType.TRAINING_LOSS_ANOMALY, self._on_loss_anomaly)
            if hasattr(DataEventType, 'QUORUM_RECOVERY_STARTED'):
                unsubscribe(DataEventType.QUORUM_RECOVERY_STARTED, self._on_quorum_recovery)
            if hasattr(DataEventType, 'CURRICULUM_ROLLBACK'):
                unsubscribe(DataEventType.CURRICULUM_ROLLBACK, self._on_curriculum_rollback)
            try:
                unsubscribe("OPPONENT_DIVERSITY_NEEDED", self._on_opponent_diversity_needed)
            except (AttributeError, TypeError):
                pass
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

            # Emit curriculum rebalanced event for downstream consumers (Jan 2026 - migrated to event_router)
            from app.coordination.event_emission_helpers import safe_emit_event
            safe_emit_event(
                "CURRICULUM_REBALANCED",
                {
                    "trigger": "model_promoted",
                    "configs_affected": [config_key],
                    "source": "curriculum_integration",
                },
                context="curriculum_integration",
            )

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

            # Emit curriculum advanced event for downstream consumers (Jan 2026 - migrated to event_router)
            from app.coordination.event_emission_helpers import safe_emit_event
            safe_emit_event(
                "CURRICULUM_ADVANCED",
                {
                    "config_key": config_key,
                    "old_tier": old_tier,
                    "new_tier": new_tier,
                    "trigger": "tier_promotion",
                    "source": "curriculum_integration",
                },
                context="curriculum_integration",
            )

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

            # Emit curriculum advanced event for downstream consumers (Jan 2026 - migrated to event_router)
            from app.coordination.event_emission_helpers import safe_emit_event
            safe_emit_event(
                "CURRICULUM_ADVANCED",
                {
                    "config_key": ",".join(configs) if configs else "crossboard",
                    "old_tier": "pre_crossboard",
                    "new_tier": "crossboard_achieved",
                    "trigger": "crossboard_promotion",
                    "source": "curriculum_integration",
                },
                context="curriculum_integration",
            )

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

                    if new_weight > old_weight:
                        # Session 17.25: Save pre-boost weight for rollback capability
                        # Only save if not already tracked (keep original pre-boost weight)
                        if config_key not in self._pre_boost_weights:
                            self._pre_boost_weights[config_key] = old_weight

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

            # 4. Emit CURRICULUM_ADVANCED to signal downstream consumers (Jan 2026 - migrated to event_router)
            from app.coordination.event_emission_helpers import safe_emit_event
            safe_emit_event(
                "CURRICULUM_ADVANCED",
                {
                    "config_key": config_key,
                    "old_tier": "stagnant",
                    "new_tier": "advancing",
                    "trigger": f"curriculum_advancement_{reason}",
                    "source": "curriculum_integration",
                },
                context="curriculum_integration",
            )

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
                        # Session 17.25: Save pre-boost weight for rollback capability
                        # Only save if not already tracked (keep original pre-boost weight)
                        if config_key not in self._pre_boost_weights:
                            self._pre_boost_weights[config_key] = old_weight

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
                safe_emit_event(
                    "CURRICULUM_ADVANCED",
                    {
                        "config_key": config_key,
                        "old_tier": "standard",
                        "new_tier": "fast_track",
                        "trigger": f"velocity_acceleration_{velocity:.0f}_elo_hr",
                        "source": "curriculum_integration",
                    },
                    context="curriculum_integration",
                )

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling Elo velocity: {e}")

    def _on_curriculum_advanced(self, event) -> None:
        """Handle CURRICULUM_ADVANCED event - propagate to similar configs.

        January 2026 Sprint 12: Enhanced curriculum hierarchy with sibling propagation.
        When one config achieves a curriculum advancement:
        - Same board family (hex→hex): 80% strength propagation
        - Cross-board (hex→square): 40% strength with Elo guard
        - Sibling player count (2p→3p/4p on same board): 60% strength

        Expected Elo improvement: +12-18 from better knowledge transfer.

        Similarity is based on:
        - Same number of players (required for same-family propagation)
        - Same board (required for sibling propagation)
        - Similar board types (hex->hex, square->square preferred)
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = extract_config_key(payload)
            old_tier = payload.get("old_tier", "")
            new_tier = payload.get("new_tier", "")
            trigger = payload.get("trigger", "unknown")
            source = payload.get("source", "")
            source_elo = payload.get("elo", payload.get("source_elo"))

            if not config_key:
                return

            # Don't propagate propagated events (prevent infinite loops)
            if source == "curriculum_propagation":
                return

            # Don't propagate crossboard_promotion events (already affect multiple configs)
            if "crossboard" in trigger.lower() or "crossboard" in new_tier.lower():
                return

            # Get source Elo for Elo guards (cross-board propagation)
            if source_elo is None:
                try:
                    from app.coordination.elo_manager import get_elo_manager
                    elo_manager = get_elo_manager()
                    source_elo = elo_manager.get_elo(config_key)
                except (ImportError, AttributeError, RuntimeError):
                    pass  # Elo manager not available

            logger.info(
                f"[MomentumToCurriculumBridge] CURRICULUM_ADVANCED: {config_key} "
                f"{old_tier} -> {new_tier} (trigger={trigger}, elo={source_elo}), "
                f"checking for hierarchy propagation"
            )

            # Find similar configs with propagation weights (January 2026 Sprint 12)
            weighted_configs = self._get_similar_configs_with_weights(config_key, source_elo)

            if not weighted_configs:
                logger.debug(f"[MomentumToCurriculumBridge] No similar configs for {config_key}")
                return

            # Emit CURRICULUM_PROPAGATE for each similar config with appropriate weight
            for target_config, weight in weighted_configs.items():
                self._emit_curriculum_propagate(
                    source_config=config_key,
                    target_config=target_config,
                    advancement_tier=new_tier,
                    original_trigger=trigger,
                    propagation_weight=weight,
                )

            # Log with weight breakdown
            weight_summary = ", ".join(f"{k}@{v:.0%}" for k, v in weighted_configs.items())
            logger.info(
                f"[MomentumToCurriculumBridge] Propagated curriculum advancement from "
                f"{config_key} to {len(weighted_configs)} configs: {weight_summary}"
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

                        # Sprint 16.1 (Jan 3, 2026): Emit rollback confirmation for observability
                        self._emit_rollback_completed(
                            config_key=config_key,
                            old_weight=old_weight,
                            new_weight=new_weight,
                            elo_delta=elo_delta,
                            trigger_reason="regression_detected",
                        )

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not apply regression adjustment: {e}")

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling regression detected: {e}")

    def _on_curriculum_rollback(self, event) -> None:
        """Handle CURRICULUM_ROLLBACK event - restore weights to pre-boost values.

        Session 17.25: When regression is detected after a curriculum boost,
        this handler restores the weights to what they were BEFORE the boost,
        undoing the boost that may have contributed to the regression.

        This is different from _on_regression_detected which applies an
        emergency weight reduction. CURRICULUM_ROLLBACK specifically undoes
        prior weight increases.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = payload.get("config_key", "")
            trigger = payload.get("trigger", "unknown")
            elo_drop = payload.get("elo_drop", 0)

            if not config_key:
                logger.debug("[MomentumToCurriculumBridge] CURRICULUM_ROLLBACK missing config_key")
                return

            # Check if we have a pre-boost weight to restore
            if config_key not in self._pre_boost_weights:
                logger.debug(
                    f"[MomentumToCurriculumBridge] No pre-boost weight for {config_key}, "
                    "skipping rollback"
                )
                return

            pre_boost_weight = self._pre_boost_weights[config_key]

            logger.warning(
                f"[MomentumToCurriculumBridge] CURRICULUM_ROLLBACK for {config_key}: "
                f"restoring to pre-boost weight {pre_boost_weight:.2f} "
                f"(trigger={trigger}, elo_drop={elo_drop})"
            )

            # Restore pre-boost weight
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                curriculum = get_curriculum_feedback()
                if curriculum:
                    current_weights = curriculum.get_curriculum_weights()
                    current_weight = current_weights.get(config_key, 1.0)

                    # Only restore if pre-boost weight is lower (undoing a boost)
                    if pre_boost_weight < current_weight:
                        curriculum.update_weight(
                            config_key=config_key,
                            new_weight=pre_boost_weight,
                            source=f"curriculum_rollback_elo_drop_{elo_drop}",
                        )
                        self._last_weights[config_key] = pre_boost_weight

                        logger.info(
                            f"[MomentumToCurriculumBridge] Restored curriculum weight for {config_key}: "
                            f"{current_weight:.2f} -> {pre_boost_weight:.2f} (rollback)"
                        )

                        # Clear the pre-boost weight since we've rolled back
                        del self._pre_boost_weights[config_key]

                        # Emit curriculum rebalanced event
                        self._emit_rebalance_event([config_key], {config_key: pre_boost_weight})

                        # Emit rollback completed for observability
                        self._emit_rollback_completed(
                            config_key=config_key,
                            old_weight=current_weight,
                            new_weight=pre_boost_weight,
                            elo_delta=-elo_drop if elo_drop else 0,
                            trigger_reason="curriculum_rollback",
                        )
                    else:
                        logger.debug(
                            f"[MomentumToCurriculumBridge] Pre-boost weight {pre_boost_weight:.2f} >= "
                            f"current {current_weight:.2f} for {config_key}, no rollback needed"
                        )

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not apply rollback: {e}")

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling curriculum rollback: {e}")

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

            # Sprint 16.1 (Jan 3, 2026): Add severity-weighted exploration boost
            # Higher severity anomalies get larger exploration boosts to help escape bad regions
            exploration_boost = self._compute_exploration_boost_from_anomaly(payload)
            if exploration_boost > 0.1:  # Only apply meaningful boosts
                self._apply_exploration_boost_for_anomaly(config_key, exploration_boost)

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling loss anomaly: {e}")

    def _compute_exploration_boost_from_anomaly(self, anomaly: dict) -> float:
        """Compute exploration boost based on anomaly severity.

        Sprint 16.1 (Jan 3, 2026): Scale exploration boost by how far the loss
        exceeds the expected threshold. Larger anomalies indicate worse data quality
        issues that require more exploration to escape.

        Args:
            anomaly: Event payload with magnitude and threshold info

        Returns:
            Boost factor to apply (0.1 to 0.3 range)
        """
        try:
            # Extract magnitude (actual loss value or deviation)
            magnitude = anomaly.get("magnitude") or anomaly.get("loss_value", 0)
            threshold = anomaly.get("threshold") or anomaly.get("expected_loss", 1.0)

            if threshold <= 0:
                threshold = 1.0  # Avoid division by zero

            # Severity = how far above threshold (normalized)
            severity = max(0, (magnitude - threshold) / threshold)

            # Base boost 0.1, scaled by severity (capped at 0.3)
            # severity 0 -> 0.1, severity 1 -> 0.2, severity 2+ -> 0.3
            boost = min(0.1 + (severity * 0.1), 0.3)

            logger.debug(
                f"[MomentumToCurriculumBridge] Anomaly exploration boost: "
                f"magnitude={magnitude:.4f}, threshold={threshold:.4f}, "
                f"severity={severity:.2f}, boost={boost:.2f}"
            )

            return boost

        except (ValueError, TypeError) as e:
            logger.debug(f"[MomentumToCurriculumBridge] Error computing exploration boost: {e}")
            return 0.1  # Default minimum boost

    def _apply_exploration_boost_for_anomaly(self, config_key: str, boost_factor: float) -> None:
        """Apply exploration boost for loss anomaly.

        Sprint 16.1 (Jan 3, 2026): Temporarily boost exploration to help escape
        training loss anomaly regions.

        Args:
            config_key: Config identifier
            boost_factor: Amount to boost exploration (0.1-0.3)
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "EXPLORATION_BOOST",
                {
                    "config_key": config_key,
                    "boost_factor": 1.0 + boost_factor,  # Convert to multiplier
                    "source": "loss_anomaly",
                    "duration_games": 200,  # Shorter duration for anomaly recovery
                },
                source="MomentumToCurriculumBridge",
            )
            logger.info(
                f"[MomentumToCurriculumBridge] Emitted exploration boost for {config_key}: "
                f"+{boost_factor*100:.0f}% (loss anomaly recovery)"
            )
        except (ImportError, AttributeError) as e:
            logger.debug(f"[MomentumToCurriculumBridge] Could not emit exploration boost: {e}")

    def _on_quorum_recovery(self, event) -> None:
        """Handle QUORUM_RECOVERY_STARTED event - boost selfplay priority during recovery.

        January 2026 Sprint 12: When quorum is lost and recovery starts, we need
        to generate fresh training data quickly to help the cluster recover.
        This handler boosts curriculum weights for all active configs to increase
        selfplay allocation during recovery.

        Action: Temporarily boost curriculum weights by 20% for all configs.
        This effect decays naturally over time as the cluster recovers.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            severity = payload.get("severity", "unknown")
            source_node = payload.get("source_node", "unknown")
            recovery_reason = payload.get("reason", "quorum_recovery")

            logger.info(
                f"[MomentumToCurriculumBridge] QUORUM_RECOVERY_STARTED: severity={severity}, "
                f"source={source_node}, reason={recovery_reason}"
            )

            # Boost curriculum weights for all active configs to accelerate recovery
            # This is a temporary boost that helps generate fresh training data
            boost_factor = 1.20  # 20% boost during recovery

            try:
                from app.coordination.curriculum import get_curriculum
                curriculum = get_curriculum()
                if curriculum:
                    # Get all active configs
                    config_keys = list(curriculum.get_all_weights().keys())

                    for config_key in config_keys:
                        old_weight = curriculum.get_weight(config_key)
                        if old_weight is None:
                            continue

                        # Apply temporary boost (capped at 1.0)
                        new_weight = min(old_weight * boost_factor, 1.0)

                        # Only apply if there's meaningful change
                        if new_weight > old_weight + 0.01:
                            curriculum.update_weight(
                                config_key=config_key,
                                weight=new_weight,
                                source="quorum_recovery_boost",
                            )
                            self._last_weights[config_key] = new_weight

                    logger.info(
                        f"[MomentumToCurriculumBridge] Applied {boost_factor:.0%} curriculum boost "
                        f"to {len(config_keys)} configs during quorum recovery"
                    )

                    # Emit curriculum rebalanced event
                    self._emit_rebalance_event(config_keys, {k: boost_factor for k in config_keys})

            except (ImportError, AttributeError) as e:
                logger.debug(f"[MomentumToCurriculumBridge] Could not apply quorum recovery boost: {e}")

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling quorum recovery: {e}")

    def _on_opponent_diversity_needed(self, event) -> None:
        """Handle OPPONENT_DIVERSITY_NEEDED - inject varied opponents to break local optimum.

        Jan 2026 P1: At escalation level 2 (48h+ stall), this handler receives
        a signal to diversify the opponent mix for a stalled config. This helps
        break through local optima by introducing varied training opponents.

        Action:
        1. Emit CURRICULUM_DIVERSITY_BOOST with suggested opponent mix
        2. Temporarily increase exploration in selfplay
        3. Reduce weight of "best" opponent in favor of varied opponents

        Suggested opponent mix at escalation level 2:
        - best: 30% (reduced from typical 60-70%)
        - previous: 40% (play against previous versions)
        - heuristic: 20% (play against heuristic AI)
        - random: 10% (pure exploration)
        """
        try:
            from app.coordination.safe_event_emit import safe_emit_event

            payload = event.payload if hasattr(event, 'payload') else event if isinstance(event, dict) else {}

            config_key = payload.get("config_key", "")
            stall_duration = payload.get("stall_duration_hours", 48.0)
            escalation_level = payload.get("escalation_level", 2)
            suggested_mix = payload.get("suggested_mix", {
                "best": 0.30,
                "previous": 0.40,
                "heuristic": 0.20,
                "random": 0.10,
            })

            if not config_key:
                logger.debug("[MomentumToCurriculumBridge] OPPONENT_DIVERSITY_NEEDED missing config_key")
                return

            logger.warning(
                f"[MomentumToCurriculumBridge] OPPONENT_DIVERSITY_NEEDED for {config_key}: "
                f"stalled {stall_duration:.1f}h, escalation level {escalation_level}. "
                f"Injecting opponent diversity: best={suggested_mix.get('best', 0.3):.0%}, "
                f"previous={suggested_mix.get('previous', 0.4):.0%}, "
                f"heuristic={suggested_mix.get('heuristic', 0.2):.0%}, "
                f"random={suggested_mix.get('random', 0.1):.0%}"
            )

            # Emit CURRICULUM_DIVERSITY_BOOST to trigger opponent mix change
            safe_emit_event(
                "CURRICULUM_DIVERSITY_BOOST",
                {
                    "config_key": config_key,
                    "stall_duration_hours": stall_duration,
                    "escalation_level": escalation_level,
                    "opponent_mix": suggested_mix,
                    "source": "stall_escalation",
                    "duration_games": 500,  # Apply diversity for next 500 games
                },
            )

            # Also emit exploration boost to increase temperature/noise
            safe_emit_event(
                "EXPLORATION_BOOST",
                {
                    "config_key": config_key,
                    "boost_factor": 1.3,  # 30% exploration boost
                    "source": "opponent_diversity_injection",
                    "duration_games": 500,
                },
            )

            logger.info(
                f"[MomentumToCurriculumBridge] Emitted CURRICULUM_DIVERSITY_BOOST and "
                f"EXPLORATION_BOOST for {config_key} (stall recovery)"
            )

            self._last_sync_time = time.time()

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[MomentumToCurriculumBridge] Error handling opponent diversity: {e}")

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
        # Use the new weighted version and extract just the keys for backward compatibility
        weighted = self._get_similar_configs_with_weights(config_key)
        return list(weighted.keys())

    def _get_similar_configs_with_weights(
        self, config_key: str, source_elo: float | None = None
    ) -> dict[str, float]:
        """Get similar configs with propagation weights for curriculum hierarchy.

        January 2026 Sprint 12: Enhanced curriculum hierarchy with sibling propagation.
        Returns weighted config mappings for smarter propagation:

        - Same board family (hex→hex, square→square): 80% weight
        - Cross-board (hex→square): 40% weight with Elo guard
        - Same player count (2p→3p, 2p→4p): 60% weight for sibling configs

        Expected Elo improvement: +12-18 from better knowledge transfer.

        Args:
            config_key: Source config key (e.g., "square8_2p")
            source_elo: Optional Elo of source config for Elo guards

        Returns:
            Dict of {config_key: propagation_weight} for similar configs
        """
        parsed = parse_config_key(config_key)
        if not parsed or not parsed.num_players:
            return {}

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

        # Propagation weights based on similarity
        SAME_BOARD_FAMILY_WEIGHT = 0.80    # hex→hex, square→square
        CROSS_BOARD_FAMILY_WEIGHT = 0.40   # hex→square
        SIBLING_PLAYER_WEIGHT = 0.60       # 2p→3p/4p on same board

        # Get target Elos for Elo guard (cross-board only propagates to lower Elo)
        target_elos: dict[str, float] = {}
        if source_elo is not None:
            try:
                from app.coordination.elo_manager import get_elo_manager
                elo_manager = get_elo_manager()
                for board in ALL_BOARDS:
                    for players in ALL_PLAYERS:
                        candidate_key = f"{board}_{players}p"
                        target_elos[candidate_key] = elo_manager.get_elo(candidate_key) or 1000.0
            except (ImportError, AttributeError, RuntimeError):
                pass  # Elo manager not available, skip guards

        similar_configs: dict[str, float] = {}

        for board in ALL_BOARDS:
            for players in ALL_PLAYERS:
                candidate_key = f"{board}_{players}p"

                # Skip source config
                if candidate_key == config_key:
                    continue

                # Determine candidate's board family
                if board.startswith("hex"):
                    candidate_family = "hex"
                elif board.startswith("square"):
                    candidate_family = "square"
                else:
                    candidate_family = board

                # Calculate propagation weight based on relationship
                weight = 0.0

                if players == source_players:
                    # Same player count
                    if candidate_family == source_family:
                        # Same family (hex→hex, square→square): highest weight
                        weight = SAME_BOARD_FAMILY_WEIGHT
                    else:
                        # Cross-board (hex→square): lower weight with Elo guard
                        # Only propagate to lower-Elo configs to avoid hurting strong configs
                        if source_elo is not None and candidate_key in target_elos:
                            target_elo = target_elos[candidate_key]
                            # Guard: only propagate if target Elo is at least 50 lower
                            if target_elo < source_elo - 50:
                                weight = CROSS_BOARD_FAMILY_WEIGHT
                            else:
                                # Target is at similar or higher Elo, skip propagation
                                logger.debug(
                                    f"[CurriculumHierarchy] Skipping cross-board propagation "
                                    f"{config_key} → {candidate_key}: target Elo ({target_elo:.0f}) "
                                    f"not significantly lower than source ({source_elo:.0f})"
                                )
                                continue
                        else:
                            # No Elo info, use full cross-board weight
                            weight = CROSS_BOARD_FAMILY_WEIGHT

                elif board == source_board:
                    # Same board, different player count (sibling)
                    # hex8_2p → hex8_3p, hex8_4p
                    weight = SIBLING_PLAYER_WEIGHT

                # Only include configs with positive weight
                if weight > 0:
                    similar_configs[candidate_key] = weight

        return similar_configs

    def _emit_curriculum_propagate(
        self,
        source_config: str,
        target_config: str,
        advancement_tier: str,
        original_trigger: str,
        propagation_weight: float = 0.5,
    ) -> None:
        """Emit CURRICULUM_PROPAGATE event for curriculum hierarchy propagation.

        January 2026 Sprint 12: Enhanced with weighted propagation:
        - Same board family (hex→hex): 80% weight
        - Cross-board (hex→square): 40% weight
        - Sibling player count: 60% weight

        Args:
            source_config: Config that achieved advancement
            target_config: Config to propagate to
            advancement_tier: The tier achieved
            original_trigger: What triggered the advancement
            propagation_weight: Weight for this propagation (0.0-1.0)
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "CURRICULUM_PROPAGATE",
                {
                    "source_config": source_config,
                    "target_config": target_config,
                    "advancement_tier": advancement_tier,
                    "original_trigger": original_trigger,
                    "propagation_weight": propagation_weight,
                    "timestamp": time.time(),
                },
                source="curriculum_integration",
            )
            logger.debug(
                f"[MomentumToCurriculumBridge] Emitted CURRICULUM_PROPAGATE: "
                f"{source_config} -> {target_config} @ {propagation_weight:.0%}"
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
            from app.coordination.event_router import publish_sync

            publish_sync(
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

    def _emit_rollback_completed(
        self,
        config_key: str,
        old_weight: float,
        new_weight: float,
        elo_delta: float,
        trigger_reason: str = "regression_detected",
    ) -> None:
        """Emit CURRICULUM_ROLLBACK_COMPLETED event for observability.

        Sprint 16.1 (Jan 3, 2026): Confirmation event when curriculum weight is
        rolled back due to regression. Enables monitoring dashboards and alerts.
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync(
                "CURRICULUM_ROLLBACK_COMPLETED",
                {
                    "config_key": config_key,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "elo_delta": elo_delta,
                    "trigger_reason": trigger_reason,
                    "weight_reduction_pct": (1 - new_weight / old_weight) * 100 if old_weight > 0 else 0,
                    "timestamp": time.time(),
                },
                source="momentum_curriculum_bridge",
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"Failed to emit rollback completed event: {e}")

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
