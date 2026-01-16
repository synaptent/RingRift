"""Feedback Loop Controller - Central orchestration of training feedback signals.

This is the "central nervous system" of the training improvement loop.
It orchestrates feedback between:
- Selfplay quality → Training intensity
- Training metrics → Curriculum adjustment
- Evaluation results → Promotion decisions
- Promotion outcomes → Selfplay exploration

The goal is to create a positive feedback loop where:
1. Good models train faster (accelerated intensity)
2. Struggling models get more exploration
3. Failed promotions trigger more aggressive training
4. Successful promotions shift curriculum to harder tasks

Usage:
    from app.coordination.feedback_loop_controller import (
        FeedbackLoopController,
        get_feedback_loop_controller,
    )

    # Get singleton instance
    controller = get_feedback_loop_controller()

    # Start the feedback loop (as daemon)
    await controller.start()

    # Manual signals (usually automatic via events)
    controller.signal_selfplay_quality("square8_2p", quality_score=0.85)
    controller.signal_training_complete("square8_2p", policy_accuracy=0.78)

December 2025: Created as part of Phase 7 feedback loop integration.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_utils import make_config_key, parse_config_key
from app.training.regression_detector import RegressionSeverity
from app.coordination.feedback.cluster_health_mixin import FeedbackClusterHealthMixin
from app.coordination.feedback.exploration_boost import ExplorationBoostMixin
from app.coordination.feedback.quality_feedback import QualityFeedbackMixin
from app.coordination.feedback.elo_velocity_mixin import EloVelocityAdaptationMixin
from app.coordination.feedback.training_curriculum_mixin import TrainingCurriculumFeedbackMixin
from app.coordination.handler_base import HandlerBase
from app.coordination.protocols import HealthCheckResult

logger = logging.getLogger(__name__)

# Jan 12, 2026: Task error tracking for escalation
# Previously, task errors were only logged and never escalated, causing silent failures.
_task_error_count = 0
_task_error_threshold = 3  # Escalate after this many consecutive errors
_last_error_context: str = ""


def _handle_task_error(task: asyncio.Task, context: str = "") -> None:
    """Handle errors from fire-and-forget tasks with escalation.

    Call this via: task.add_done_callback(lambda t: _handle_task_error(t, "context"))

    Jan 12, 2026: Added error tracking and escalation to prevent silent failures.
    After 3 consecutive errors, logs a CRITICAL message for alerting.
    """
    global _task_error_count, _last_error_context
    try:
        exc = task.exception()
        if exc is not None:
            _task_error_count += 1
            _last_error_context = context
            logger.error(f"[FeedbackLoopController] Task error{' in ' + context if context else ''}: {exc}")

            # Escalate after threshold
            if _task_error_count >= _task_error_threshold:
                logger.critical(
                    f"[FeedbackLoopController] DEGRADED: {_task_error_count} consecutive task failures. "
                    f"Last context: {_last_error_context}. Feedback loop may be malfunctioning."
                )
                # Try to emit event for monitoring (best effort)
                try:
                    from app.coordination.event_emission_helpers import safe_emit_event
                    safe_emit_event("FEEDBACK_LOOP_DEGRADED", {
                        "consecutive_errors": _task_error_count,
                        "last_context": _last_error_context,
                        "last_error": str(exc),
                    })
                except Exception:
                    pass  # Event emission is best-effort
        else:
            # Task succeeded, reset error count
            _task_error_count = 0
    except asyncio.CancelledError:
        pass  # Task was cancelled, not an error
    except asyncio.InvalidStateError:
        pass  # Task not done yet


def _safe_create_task(coro, context: str = "") -> asyncio.Task | None:
    """Create a task with error handling callback.

    Returns the task if created successfully, None otherwise.
    """
    try:
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: _handle_task_error(t, context))
        return task
    except RuntimeError as e:
        logger.debug(f"[FeedbackLoopController] Could not create task for {context}: {e}")
        return None


# Event emission for selfplay feedback loops (Phase 21.2 - Dec 2025)
# December 2025: Use canonical event_router imports
from app.config.thresholds import (
    PROMOTION_WIN_RATE_THRESHOLD,
    # December 28, 2025: Extracted constants from magic numbers
    POLICY_ACCURACY_EVALUATION_THRESHOLD,
    FAILURE_EXPLORATION_BOOST,
    SUCCESS_INTENSITY_REDUCTION,
    ELO_SIGNIFICANT_CHANGE,
    PLATEAU_COOLDOWN_SECONDS,
    TOURNAMENT_COOLDOWN_SECONDS,
    EXPLORATION_BOOST_MAX,
    EXPLORATION_BOOST_MULTIPLIER,
    EXPLORATION_BOOST_RECOVERY,
    EXPLORATION_BOOST_PER_ANOMALY,
    EXPLORATION_BOOST_PER_STALL_GROUP,
    EXPLORATION_BOOST_STALL_MAX,
    EXPLORATION_BOOST_DECAY,
    EXPLORATION_BOOST_BASE,
    EXPLORATION_BOOST_SUCCESS_DECREMENT,
    EXPLORATION_BOOST_FAILURE_INCREMENT,
    RATE_CHANGE_SIGNIFICANT_PERCENT,
    CURRICULUM_WEIGHT_ADJUSTMENT_DOWN,
    CURRICULUM_WEIGHT_ADJUSTMENT_UP,
    POLICY_LOW_THRESHOLD,
    POLICY_HIGH_THRESHOLD,
    ELO_PLATEAU_PER_HOUR,
    ELO_FAST_IMPROVEMENT_PER_HOUR,
    ELO_TARGET_ALL_CONFIGS,
    LOSS_ANOMALY_SEVERE_COUNT,
    TREND_DURATION_MODERATE,
    TREND_DURATION_SEVERE,
    MEDIUM_QUALITY_THRESHOLD,
)

try:
    from app.coordination.event_router import emit_selfplay_target_updated
    HAS_SELFPLAY_EVENTS = True
except ImportError:
    emit_selfplay_target_updated = None
    HAS_SELFPLAY_EVENTS = False


@dataclass
class FeedbackState:
    """State tracking for a single config's feedback loop.

    DEPRECATION NOTE (December 28, 2025):
    This class will be consolidated with MonitoringFeedbackState in
    app.coordination.feedback_state in Q1 2026. New code should prefer:

        from app.coordination.feedback_state import MonitoringFeedbackState

    Field mapping for migration:
        last_selfplay_quality -> quality_score
        last_training_accuracy -> training_accuracy
        last_evaluation_win_rate -> win_rate
        last_elo -> elo_current
        current_curriculum_weight -> curriculum_weight
        current_training_intensity -> training_intensity
        current_exploration_boost -> exploration_boost
    """

    config_key: str

    # Quality metrics
    last_selfplay_quality: float = 0.0
    last_training_accuracy: float = 0.0
    last_evaluation_win_rate: float = 0.0

    # Elo tracking for velocity calculation (Dec 28 2025)
    last_elo: float = 1500.0
    elo_history: list = None  # List of (timestamp, elo) tuples
    elo_velocity: float = 0.0  # Elo points per hour

    # Status
    last_promotion_success: bool | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Timing
    last_selfplay_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0

    # Dec 29 2025: Engine bandit tracking
    last_selfplay_engine: str = "gumbel-mcts"  # Last engine mode used for selfplay
    last_selfplay_games: int = 0  # Games from last selfplay batch
    elo_before_training: float = 1500.0  # Elo before the most recent training

    # Dec 29 2025: Curriculum tier tracking for velocity-based advancement
    curriculum_tier: int = 0  # Current curriculum tier (0=beginner, 1=intermediate, 2=advanced, 3=expert)
    curriculum_last_advanced: float = 0.0  # Timestamp of last curriculum advancement

    # Work queue metrics (December 2025)
    work_completed_count: int = 0  # Total work items completed
    last_work_completion_time: float = 0.0

    # Feedback signals applied
    current_training_intensity: str = "normal"  # normal, accelerated, hot_path, reduced
    current_exploration_boost: float = 1.0  # 1.0 = normal, >1.0 = more exploration
    current_curriculum_weight: float = 1.0  # curriculum priority multiplier
    current_search_budget: int = 400  # Gumbel MCTS budget (Dec 28 2025)

    def __post_init__(self):
        if self.elo_history is None:
            self.elo_history = []

    def update_elo(self, elo: float, timestamp: float = None) -> float:
        """Update Elo and calculate velocity.

        Dec 28 2025: Track Elo over time to detect plateaus.
        Returns: Elo velocity (points per hour)
        """
        if timestamp is None:
            timestamp = time.time()

        self.last_elo = elo
        self.elo_history.append((timestamp, elo))

        # Keep last 10 data points
        if len(self.elo_history) > 10:
            self.elo_history = self.elo_history[-10:]

        # Calculate velocity from last 3+ data points
        if len(self.elo_history) >= 3:
            oldest_time, oldest_elo = self.elo_history[0]
            newest_time, newest_elo = self.elo_history[-1]
            hours_elapsed = (newest_time - oldest_time) / 3600.0
            if hours_elapsed > 0.01:  # At least ~36 seconds
                self.elo_velocity = (newest_elo - oldest_elo) / hours_elapsed

        return self.elo_velocity


@dataclass
class AdaptiveTrainingSignal:
    """Adaptive training parameters based on evaluation results.

    December 29, 2025: Phase 6 - Training parameters adapt based on evaluation results.
    Consumed by training system to adjust LR, batch size, epochs, and gradient clipping.

    Fields:
        learning_rate_multiplier: Multiplier for base LR (0.2-1.0)
        batch_size_multiplier: Multiplier for base batch size (1.0-2.0)
        epochs_extension: Additional epochs to train (0-20)
        gradient_clip_enabled: Whether to enable gradient clipping
        reason: Human-readable explanation for the signal
    """

    learning_rate_multiplier: float = 1.0
    batch_size_multiplier: float = 1.0
    epochs_extension: int = 0
    gradient_clip_enabled: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for event emission."""
        return {
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "batch_size_multiplier": self.batch_size_multiplier,
            "epochs_extension": self.epochs_extension,
            "gradient_clip_enabled": self.gradient_clip_enabled,
            "reason": self.reason,
        }


class FeedbackLoopController(TrainingCurriculumFeedbackMixin, EloVelocityAdaptationMixin, QualityFeedbackMixin, ExplorationBoostMixin, FeedbackClusterHealthMixin, HandlerBase):
    """Central controller orchestrating all feedback signals.

    Subscribes to:
    - SELFPLAY_COMPLETE: Assess data quality, adjust training intensity
    - TRAINING_COMPLETE: Trigger evaluation, adjust curriculum
    - EVALUATION_COMPLETED: Consider promotion, record results
    - PROMOTION_COMPLETE: Adjust curriculum and exploration based on outcome
    - REGRESSION_DETECTED: Boost exploration, request more selfplay (Dec 2025)

    Emits decisions that are consumed by:
    - FeedbackAccelerator: Training intensity signals
    - CurriculumFeedback: Curriculum weight adjustments
    - TemperatureScheduler: Exploration rate adjustments
    - SelfplayScheduler: Target games and priorities (via SELFPLAY_TARGET_UPDATED)

    January 2026 Sprint 17.9: Cluster health handlers extracted to FeedbackClusterHealthMixin.
    December 30, 2025: Now inherits from HandlerBase for unified singleton
    management, health checks, and error tracking. Event subscriptions remain
    in _subscribe_to_events() due to complex conditional logic (coexistence
    with UnifiedFeedbackOrchestrator).
    """

    def __init__(self):
        # December 30, 2025: Initialize HandlerBase for unified patterns
        # Note: Event subscriptions handled in _subscribe_to_events() due to
        # complex conditional logic (coexistence with UnifiedFeedbackOrchestrator)
        # January 5, 2026 (Phase 7.8): Reduced from 60s to 30s for faster feedback.
        # Expected improvement: +3-7 Elo from faster training iteration cycles.
        super().__init__(
            name="feedback_loop_controller",
            cycle_interval=30.0,  # Feedback loop check interval (was 60s)
            dedup_enabled=True,
        )

        self._states: dict[str, FeedbackState] = {}
        # Note: _running and _subscribed inherited from HandlerBase but kept here
        # for backward compatibility with existing code checking these attributes
        self._running = False
        self._subscribed = False
        # January 2026: Track actually subscribed events for safe unsubscription
        # Prevents trying to unsubscribe from events that were skipped due to
        # coexistence with UnifiedFeedbackOrchestrator
        self._subscribed_events: set[tuple[Any, Any]] = set()  # (event_type, handler) pairs
        self._lock = threading.Lock()

        # Phase 23.1: Track selfplay rate changes for monitoring
        self._rate_history: dict[str, list[dict[str, Any]]] = {}

        # Gap 1 fix (Dec 2025): Initialize cluster health flag
        self._cluster_healthy = True

        # Configuration (Dec 2025: thresholds from app.config.thresholds)
        self.policy_accuracy_threshold = POLICY_ACCURACY_EVALUATION_THRESHOLD
        self.promotion_threshold = PROMOTION_WIN_RATE_THRESHOLD  # Win rate for promotion
        self.failure_exploration_boost = FAILURE_EXPLORATION_BOOST
        self.success_intensity_reduction = SUCCESS_INTENSITY_REDUCTION

    def _get_or_create_state(self, config_key: str) -> FeedbackState:
        """Get or create state for a config."""
        with self._lock:
            if config_key not in self._states:
                self._states[config_key] = FeedbackState(config_key=config_key)
            return self._states[config_key]

    # =========================================================================
    # HandlerBase Abstract Methods
    # =========================================================================

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions (HandlerBase abstract method).

        December 30, 2025: Returns empty dict because FeedbackLoopController
        handles its own event subscriptions in _subscribe_to_events() due to
        complex conditional logic (coexistence with UnifiedFeedbackOrchestrator).

        The actual subscriptions happen in start() -> _subscribe_to_events().
        """
        return {}

    async def _run_cycle(self) -> None:
        """Main work loop iteration (HandlerBase abstract method).

        December 30, 2025: FeedbackLoopController is primarily event-driven,
        so this cycle performs periodic health monitoring and state cleanup.
        """
        # Periodic state maintenance
        if not self._running:
            return

        # Record cycle for health tracking
        self._stats.cycles_completed += 1

        # Clean up stale states (configs not updated in 24+ hours)
        now = time.time()
        stale_threshold = 24 * 3600  # 24 hours

        stale_configs = [
            key for key, state in self._states.items()
            if now - max(
                state.last_selfplay_time,
                state.last_training_time,
                state.last_evaluation_time,
            ) > stale_threshold
        ]

        # Log stale configs but don't remove (may be intentionally inactive)
        if stale_configs:
            logger.debug(
                f"[FeedbackLoopController] {len(stale_configs)} configs inactive >24h: "
                f"{stale_configs[:5]}..."
            )

    async def start(self) -> None:
        """Start the feedback loop controller.

        Subscribes to all relevant events and begins orchestration.
        """
        if self._running:
            logger.warning("FeedbackLoopController already running")
            return

        self._running = True

        # Subscribe to events
        self._subscribe_to_events()

        # Wire all curriculum feedback integrations
        self._wire_curriculum_feedback()

        # Wire exploration boost to active temperature schedulers
        self._wire_exploration_boost()

        # Subscribe to lazy scheduler registration for schedulers created after startup
        self._subscribe_to_lazy_scheduler_registration()

        logger.info("[FeedbackLoopController] Started feedback loop orchestration")

    async def stop(self) -> None:
        """Stop the feedback loop controller."""
        self._running = False
        self._unsubscribe_from_events()
        logger.info("[FeedbackLoopController] Stopped")

    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    def _safe_subscribe(self, bus: Any, event_type: Any, handler: Any) -> None:
        """Subscribe to an event and track it for safe unsubscription.

        January 2026: Helper to track which events were actually subscribed,
        enabling safe unsubscription without errors.
        """
        bus.subscribe(event_type, handler)
        self._subscribed_events.add((event_type, handler))

    def _subscribe_to_events(self) -> None:
        """Subscribe to all relevant training events.

        December 2025: Added coexistence guard with UnifiedFeedbackOrchestrator.
        If the unified orchestrator is already running, we skip subscribing to
        overlapping events to prevent duplicate processing. FeedbackLoopController
        still handles events that the unified orchestrator doesn't cover.

        January 2026: Uses _safe_subscribe() to track subscriptions for clean
        unsubscription.
        """
        if self._subscribed:
            return

        # Check if UnifiedFeedbackOrchestrator is already running (December 2025)
        # If so, skip overlapping subscriptions to prevent duplicate event processing
        unified_running = False
        try:
            from app.coordination.unified_feedback import get_unified_feedback

            unified = get_unified_feedback()
            if unified._running and unified._subscribed:
                unified_running = True
                logger.info(
                    "[FeedbackLoopController] UnifiedFeedbackOrchestrator is active, "
                    "deferring core feedback events to it"
                )
        except ImportError:
            pass  # Unified module not available, proceed normally
        except (AttributeError, TypeError):
            pass  # Unified orchestrator not initialized yet

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()

            # Core feedback events - skip if unified orchestrator is handling them
            if not unified_running:
                self._safe_subscribe(bus, DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
                self._safe_subscribe(bus, DataEventType.TRAINING_COMPLETED, self._on_training_complete)
                self._safe_subscribe(bus, DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
                self._safe_subscribe(bus, DataEventType.MODEL_PROMOTED, self._on_promotion_complete)

            # Work queue events - FeedbackLoopController only (not in unified)
            self._safe_subscribe(bus, DataEventType.WORK_COMPLETED, self._on_work_completed)

            # Phase 27: Subscribe to work failure events (December 2025)
            # Closes gap: WORK_FAILED/WORK_TIMEOUT events were orphaned
            if hasattr(DataEventType, 'WORK_FAILED'):
                self._safe_subscribe(bus, DataEventType.WORK_FAILED, self._on_work_failed)
            if hasattr(DataEventType, 'WORK_TIMEOUT'):
                self._safe_subscribe(bus, DataEventType.WORK_TIMEOUT, self._on_work_timeout)

            # Phase 23.1: Subscribe to selfplay rate change events for monitoring
            if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                self._safe_subscribe(bus, DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)

            # Subscribe to DATABASE_CREATED for data awareness (December 2025 - Phase 4A.3)
            # This provides early visibility into new databases before DataPipelineOrchestrator
            if hasattr(DataEventType, 'DATABASE_CREATED'):
                self._safe_subscribe(bus, DataEventType.DATABASE_CREATED, self._on_database_created)

            # Phase 8: Subscribe to training loss anomaly events (December 2025)
            # Closes critical feedback loop: training loss anomaly → quality check/exploration boost
            # Note: event_count starts at 4 if not unified_running (core events), 0 otherwise
            event_count = 4 if not unified_running else 0
            event_count += 1  # WORK_COMPLETED is always subscribed

            # TRAINING_LOSS_ANOMALY overlaps with unified orchestrator
            if not unified_running and hasattr(DataEventType, 'TRAINING_LOSS_ANOMALY'):
                self._safe_subscribe(bus, DataEventType.TRAINING_LOSS_ANOMALY, self._on_training_loss_anomaly)
                event_count += 1
            # TRAINING_LOSS_TREND is FeedbackLoopController-only (unified doesn't handle it)
            if hasattr(DataEventType, 'TRAINING_LOSS_TREND'):
                self._safe_subscribe(bus, DataEventType.TRAINING_LOSS_TREND, self._on_training_loss_trend)
                event_count += 1

            # QUALITY_DEGRADED overlaps with unified orchestrator (P1.1 Dec 2025)
            if not unified_running and hasattr(DataEventType, 'QUALITY_DEGRADED'):
                self._safe_subscribe(bus, DataEventType.QUALITY_DEGRADED, self._on_quality_degraded_for_training)
                event_count += 1

            # P10-LOOP-2 (Dec 2025): EVALUATION_FAILED is FeedbackLoopController-only
            if hasattr(DataEventType, 'EVALUATION_FAILED'):
                self._safe_subscribe(bus, DataEventType.EVALUATION_FAILED, self._on_evaluation_failed)
                event_count += 1

            # REGRESSION_DETECTED overlaps with unified orchestrator (Dec 2025)
            if not unified_running and hasattr(DataEventType, 'REGRESSION_DETECTED'):
                self._safe_subscribe(bus, DataEventType.REGRESSION_DETECTED, self._on_regression_detected)
                event_count += 1

            # Dec 2025: Subscribe to DAEMON_STATUS_CHANGED for health monitoring
            if hasattr(DataEventType, 'DAEMON_STATUS_CHANGED'):
                self._safe_subscribe(bus, DataEventType.DAEMON_STATUS_CHANGED, self._on_daemon_status_changed)
                event_count += 1

            # Dec 2025: Subscribe to P2P_CLUSTER_UNHEALTHY for cluster health feedback
            if hasattr(DataEventType, 'P2P_CLUSTER_UNHEALTHY'):
                self._safe_subscribe(bus, DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_p2p_cluster_unhealthy)
                event_count += 1

            # Dec 2025: Subscribe to TRAINING_ROLLBACK_NEEDED for rollback coordination
            if hasattr(DataEventType, 'TRAINING_ROLLBACK_NEEDED'):
                self._safe_subscribe(bus, DataEventType.TRAINING_ROLLBACK_NEEDED, self._on_training_rollback_needed)
                event_count += 1

            # Dec 28, 2025: Subscribe to TRAINING_ROLLBACK_COMPLETED for post-rollback recovery
            # Closes feedback loop: rollback complete → update state, resume training with adjustments
            if hasattr(DataEventType, 'TRAINING_ROLLBACK_COMPLETED'):
                self._safe_subscribe(bus, DataEventType.TRAINING_ROLLBACK_COMPLETED, self._on_training_rollback_completed)
                event_count += 1

            # Dec 2025: Subscribe to QUALITY_CHECK_FAILED for quality feedback
            if hasattr(DataEventType, 'QUALITY_CHECK_FAILED'):
                self._safe_subscribe(bus, DataEventType.QUALITY_CHECK_FAILED, self._on_quality_check_failed)
                event_count += 1

            # Dec 2025: Subscribe to QUALITY_FEEDBACK_ADJUSTED for dynamic training adjustments
            # Closes feedback loop: quality assessment → training intensity/exploration adjustments
            if hasattr(DataEventType, 'QUALITY_FEEDBACK_ADJUSTED'):
                self._safe_subscribe(bus, DataEventType.QUALITY_FEEDBACK_ADJUSTED, self._on_quality_feedback_adjusted)
                event_count += 1

            # Dec 2025: Subscribe to CPU_PIPELINE_JOB_COMPLETED for Vast.ai CPU selfplay jobs
            # Closes integration gap: CPU selfplay completions now trigger downstream pipeline
            if hasattr(DataEventType, 'CPU_PIPELINE_JOB_COMPLETED'):
                self._safe_subscribe(bus, DataEventType.CPU_PIPELINE_JOB_COMPLETED, self._on_cpu_pipeline_job_completed)
                event_count += 1

            # Dec 2025: Subscribe to HIGH_QUALITY_DATA_AVAILABLE for quality recovery
            # Closes feedback loop: quality recovery → reduce exploration, resume normal training
            if hasattr(DataEventType, 'HIGH_QUALITY_DATA_AVAILABLE'):
                self._safe_subscribe(bus, DataEventType.HIGH_QUALITY_DATA_AVAILABLE, self._on_high_quality_data_available)
                event_count += 1

            # Dec 27, 2025: Subscribe to QUALITY_SCORE_UPDATED for quality tracking
            # Closes feedback loop: quality monitoring → training intensity adjustments
            if hasattr(DataEventType, 'QUALITY_SCORE_UPDATED'):
                self._safe_subscribe(bus, DataEventType.QUALITY_SCORE_UPDATED, self._on_quality_score_updated)
                event_count += 1

            # Dec 27, 2025: Subscribe to CLUSTER_CAPACITY_CHANGED for resource adjustments
            # Closes feedback loop: cluster capacity changes → selfplay/training rate adjustments
            if hasattr(DataEventType, 'CLUSTER_CAPACITY_CHANGED'):
                self._safe_subscribe(bus, DataEventType.CLUSTER_CAPACITY_CHANGED, self._on_cluster_capacity_changed)
                event_count += 1

            # Dec 27, 2025: Subscribe to HEALTH_CHECK_PASSED/FAILED for node health tracking
            # Closes feedback loop: node health changes → training scheduling adjustments
            if hasattr(DataEventType, 'HEALTH_CHECK_PASSED'):
                self._safe_subscribe(bus, DataEventType.HEALTH_CHECK_PASSED, self._on_health_check_passed)
                event_count += 1
            if hasattr(DataEventType, 'HEALTH_CHECK_FAILED'):
                self._safe_subscribe(bus, DataEventType.HEALTH_CHECK_FAILED, self._on_health_check_failed)
                event_count += 1

            # Dec 29, 2025: Subscribe to PLATEAU_DETECTED for exploration boost
            # Closes feedback loop: plateau detection → exploration boost → break out of plateau
            if hasattr(DataEventType, 'PLATEAU_DETECTED'):
                self._safe_subscribe(bus, DataEventType.PLATEAU_DETECTED, self._on_plateau_detected)
                event_count += 1

            # Jan 3, 2026: Subscribe to TRAINING_TIMEOUT_REACHED for timeout recovery
            # Closes feedback loop: training timeout → exploration boost, selfplay priority bump
            # Critical gap fix: event was emitted at training_trigger_daemon.py:3314 but had no handler
            if hasattr(DataEventType, 'TRAINING_TIMEOUT_REACHED'):
                self._safe_subscribe(bus, DataEventType.TRAINING_TIMEOUT_REACHED, self._on_training_timeout_reached)
                event_count += 1

            logger.info(f"[FeedbackLoopController] Subscribed to {event_count} event types")

            # December 29, 2025: Only set _subscribed = True after successful subscription
            # Previously this was in finally block which caused race condition:
            # - If subscription failed, _subscribed was still True
            # - Next start() call would skip re-subscription entirely
            # - Events would be silently missed
            self._subscribed = True

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Failed to subscribe: {e}")
            # Note: _subscribed stays False on failure, allowing retry on next start()

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events.

        January 2026: Now uses _subscribed_events tracking to only unsubscribe
        from events that were actually subscribed. Prevents errors when events
        were skipped due to UnifiedFeedbackOrchestrator coexistence.
        """
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()

            # Only unsubscribe from events we actually subscribed to
            for event_type, handler in list(self._subscribed_events):
                try:
                    bus.unsubscribe(event_type, handler)
                except (ValueError, KeyError) as e:
                    # Handler not found - may have been unsubscribed elsewhere
                    logger.debug(f"[FeedbackLoopController] Handler not found during unsubscribe: {e}")

            # Clear tracking set
            self._subscribed_events.clear()

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Error unsubscribing: {e}")
        finally:
            # Dec 2025: Always reset _subscribed to allow re-subscription after restart
            self._subscribed = False

    def _wire_curriculum_feedback(self) -> None:
        """Wire all curriculum feedback integrations."""
        try:
            from app.training.curriculum_feedback import wire_all_curriculum_feedback

            watchers = wire_all_curriculum_feedback(
                elo_significant_change=ELO_SIGNIFICANT_CHANGE,
                plateau_cooldown_seconds=PLATEAU_COOLDOWN_SECONDS,
                tournament_cooldown_seconds=TOURNAMENT_COOLDOWN_SECONDS,
                promotion_failure_urgency="high",
                auto_export=True,
            )

            logger.info(
                f"[FeedbackLoopController] Wired {len(watchers)} curriculum feedback integrations"
            )
        except ImportError:
            logger.warning(
                "[FeedbackLoopController] curriculum_feedback module not available"
            )
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Failed to wire curriculum feedback: {e}")

    def _wire_exploration_boost(self) -> None:
        """Wire exploration boost events to active temperature schedulers.

        December 2025: Connects MODEL_PROMOTED and PROMOTION_FAILED events
        to the temperature scheduler's set_exploration_boost() method.

        When promotion succeeds: exploration boost resets to 1.0
        When promotion fails: exploration boost increases by 1.2x (up to 2.0)

        This closes the feedback loop: poor performance → more exploration → diverse data
        """
        try:
            from app.training.temperature_scheduling import (
                get_active_schedulers,
                wire_exploration_boost,
            )

            schedulers = get_active_schedulers()
            if not schedulers:
                logger.debug(
                    "[FeedbackLoopController] No active schedulers to wire "
                    "(normal if no selfplay running)"
                )
                return

            wired_count = 0
            for config_key, scheduler in schedulers.items():
                if wire_exploration_boost(scheduler, config_key):
                    wired_count += 1
                    logger.info(
                        f"[FeedbackLoopController] Wired exploration boost for {config_key}"
                    )

            if wired_count > 0:
                logger.info(
                    f"[FeedbackLoopController] Wired exploration boost to "
                    f"{wired_count} scheduler(s)"
                )

        except ImportError:
            logger.debug(
                "[FeedbackLoopController] Temperature scheduling not available"
            )
    def _subscribe_to_lazy_scheduler_registration(self) -> None:
        """Subscribe to SCHEDULER_REGISTERED for lazy exploration boost wiring.

        December 2025: This enables wiring exploration boost to schedulers that
        register AFTER FeedbackLoopController has started. Without this, schedulers
        created by new selfplay processes would miss feedback events.
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus is None:
                return

            def on_scheduler_registered(event) -> None:
                """Wire exploration boost when new scheduler registers."""
                payload = event.payload if hasattr(event, "payload") else {}
                config_key = extract_config_key(payload)

                if not config_key:
                    return

                try:
                    from app.training.temperature_scheduling import (
                        get_active_schedulers,
                        wire_exploration_boost,
                    )

                    schedulers = get_active_schedulers()
                    if config_key in schedulers:
                        if wire_exploration_boost(schedulers[config_key], config_key):
                            logger.info(
                                f"[FeedbackLoopController] Lazily wired exploration "
                                f"boost for {config_key}"
                            )
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"[FeedbackLoopController] Lazy wiring failed: {e}")

            # P0.6 Dec 2025: Use DataEventType enum for type-safe subscription
            # January 2026: Track subscription for safe unsubscription
            self._safe_subscribe(bus, DataEventType.SCHEDULER_REGISTERED, on_scheduler_registered)
            logger.debug("[FeedbackLoopController] Subscribed to SCHEDULER_REGISTERED")

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Failed to subscribe to lazy scheduler: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion.

        Actions:
        1. Assess data quality
        2. Update training intensity based on quality
        3. Signal training readiness if quality is sufficient
        4. Track engine mode for bandit feedback (Dec 29 2025)

        Sprint 17.9: Converted to async to avoid blocking event loop during
        SQLite quality assessment.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            games_count = payload.get("games_count", 0)
            db_path = payload.get("db_path", "")
            engine_mode = payload.get("engine_mode", "gumbel-mcts")  # Dec 29 2025

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            # Dec 29 2025: Track engine mode for bandit feedback
            state.last_selfplay_engine = engine_mode
            state.last_selfplay_games += games_count  # Accumulate across batches

            # Assess data quality (Sprint 17.9: now async to avoid blocking event loop)
            previous_quality = state.last_selfplay_quality
            quality_score = await self._assess_selfplay_quality_async(db_path, games_count)
            state.last_selfplay_quality = quality_score

            logger.info(
                f"[FeedbackLoopController] Selfplay complete for {config_key}: "
                f"{games_count} games, quality={quality_score:.2f}"
            )

            # Phase 5 (Dec 2025): Emit QUALITY_DEGRADED event when quality drops below threshold
            # December 2025: Use centralized threshold from config
            try:
                from app.config.thresholds import MEDIUM_QUALITY_THRESHOLD
                quality_threshold = MEDIUM_QUALITY_THRESHOLD
            except ImportError:
                quality_threshold = 0.6  # Fallback default
            if quality_score < quality_threshold:
                self._emit_quality_degraded(config_key, quality_score, quality_threshold, previous_quality)

            # Update training intensity based on quality
            self._update_training_intensity(config_key, quality_score)

            # Gap 4 fix (Dec 2025): Update curriculum weight based on selfplay quality
            self._update_curriculum_weight_from_selfplay(config_key, quality_score)

            # Signal training readiness if quality is good
            if quality_score >= quality_threshold:
                self._signal_training_ready(config_key, quality_score)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling selfplay complete: {e}")

    def _on_selfplay_rate_changed(self, event: Any) -> None:
        """Handle selfplay rate change events (Phase 23.1).

        Tracks rate changes for monitoring and logs significant adjustments.
        This enables visibility into how Elo momentum affects selfplay rates.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            old_rate = payload.get("old_rate", 1.0)
            new_rate = payload.get("new_rate", 1.0)
            change_percent = payload.get("change_percent", 0.0)
            momentum_state = payload.get("momentum_state", "unknown")
            timestamp = payload.get("timestamp", time.time())

            if not config_key:
                return

            # Track in rate history
            if config_key not in self._rate_history:
                self._rate_history[config_key] = []

            self._rate_history[config_key].append({
                "rate": new_rate,
                "old_rate": old_rate,
                "change_percent": change_percent,
                "momentum_state": momentum_state,
                "timestamp": timestamp,
            })

            # Keep bounded history (last 100 changes per config)
            if len(self._rate_history[config_key]) > 100:
                self._rate_history[config_key] = self._rate_history[config_key][-100:]

            logger.info(
                f"[FeedbackLoopController] Selfplay rate for {config_key}: "
                f"{old_rate:.2f}x → {new_rate:.2f}x ({change_percent:+.0f}%), "
                f"momentum={momentum_state}"
            )

            # Gap 2 fix (Dec 2025): Sync curriculum weight when rate changes significantly
            # When Elo momentum drives big rate changes, adjust curriculum priority accordingly
            if abs(change_percent) >= RATE_CHANGE_SIGNIFICANT_PERCENT:
                try:
                    from app.training.curriculum_feedback import get_curriculum_feedback

                    feedback = get_curriculum_feedback()
                    state = self._get_or_create_state(config_key)

                    # Increasing rate = model improving = can reduce curriculum weight slightly
                    # Decreasing rate = model struggling = increase curriculum weight for more focus
                    if change_percent > 0:  # Rate increased
                        adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_DOWN
                    else:  # Rate decreased
                        adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_UP

                    current_weight = feedback._current_weights.get(config_key, 1.0)
                    new_weight = max(
                        feedback.weight_min,
                        min(feedback.weight_max, current_weight + adjustment)
                    )

                    if abs(new_weight - current_weight) > 0.01:
                        feedback._current_weights[config_key] = new_weight
                        state.current_curriculum_weight = new_weight

                        logger.info(
                            f"[FeedbackLoopController] Curriculum weight adjusted for {config_key}: "
                            f"{current_weight:.2f} → {new_weight:.2f} (rate change {change_percent:+.0f}%)"
                        )
                except ImportError:
                    logger.debug("[FeedbackLoopController] curriculum_feedback not available")
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"[FeedbackLoopController] Failed to adjust curriculum: {e}")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Error handling rate change: {e}")

    async def _on_cpu_pipeline_job_completed(self, event: Any) -> None:
        """Handle CPU_PIPELINE_JOB_COMPLETED from Vast.ai CPU selfplay jobs.

        December 2025: Closes integration gap - CPU selfplay completions now trigger
        downstream pipeline actions (training readiness, quality assessment, etc.).

        This event is emitted by VastCpuPipelineDaemon when CPU-based selfplay jobs
        complete on Vast.ai nodes. We treat these like GPU selfplay completions.

        Sprint 17.9: Converted to async to avoid blocking event loop during
        SQLite quality assessment.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            games_count = payload.get("games_count", 0) or payload.get("games_generated", 0)
            db_path = payload.get("db_path", "")
            node_id = payload.get("node_id", "")
            job_id = payload.get("job_id", "")

            if not config_key:
                logger.debug("[FeedbackLoopController] CPU pipeline job missing config_key")
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            logger.info(
                f"[FeedbackLoopController] CPU pipeline job complete for {config_key}: "
                f"{games_count} games from node={node_id}, job={job_id}"
            )

            # Assess data quality (Sprint 17.9: now async to avoid blocking event loop)
            quality_score = await self._assess_selfplay_quality_async(db_path, games_count)
            state.last_selfplay_quality = quality_score

            # Update training intensity based on quality
            self._update_training_intensity(config_key, quality_score)

            # Signal training readiness if quality is good
            try:
                from app.config.thresholds import MEDIUM_QUALITY_THRESHOLD
                quality_threshold = MEDIUM_QUALITY_THRESHOLD
            except ImportError:
                quality_threshold = 0.6

            if quality_score >= quality_threshold:
                self._signal_training_ready(config_key, quality_score)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling CPU pipeline complete: {e}")

    def _on_database_created(self, event: Any) -> None:
        """Handle DATABASE_CREATED event (December 2025 - Phase 4A.3).

        Provides early awareness of new databases for feedback loop coordination.
        This handler primarily logs and tracks database creation for monitoring.
        The main processing happens in DataPipelineOrchestrator.

        Actions:
        1. Log database creation for visibility
        2. Track creation timestamps for freshness monitoring
        3. Update state for potential training triggers
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            db_path = payload.get("db_path", "")
            config_key = extract_config_key(payload)
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            node_id = payload.get("node_id", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            # Track database creation time for freshness awareness
            if not hasattr(state, 'last_database_created'):
                state.last_database_created = 0.0
            state.last_database_created = time.time()

            logger.info(
                f"[FeedbackLoopController] New database created for {config_key}: "
                f"{db_path} on {node_id}"
            )

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Error handling database_created: {e}")

    def _on_training_loss_anomaly(self, event: Any) -> None:
        """Handle training loss anomaly events (Phase 8).

        Closes the critical feedback loop: training loss spike detected →
        trigger quality check and/or exploration boost.

        Uses adaptive severe count thresholds (January 2026):
        - Early training (epochs 0-4): 5 consecutive anomalies before escalation
        - Mid training (epochs 5-14): 3 consecutive anomalies
        - Late training (epochs 15+): 2 consecutive anomalies (catch early)

        Actions:
        1. Log the anomaly for monitoring
        2. Trigger quality check on training data
        3. Optionally boost exploration if anomaly is severe
        4. Track consecutive anomalies for escalation
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            loss_value = payload.get("loss", 0.0)
            expected_loss = payload.get("expected_loss", 0.0)
            deviation = payload.get("deviation", 0.0)
            epoch = payload.get("epoch", 0)
            severity = payload.get("severity", "unknown")

            if not config_key:
                return

            # Get adaptive severe count threshold based on epoch
            from app.config.thresholds import get_severe_anomaly_count
            severe_count_threshold = get_severe_anomaly_count(epoch)

            logger.warning(
                f"[FeedbackLoopController] Training loss anomaly for {config_key}: "
                f"loss={loss_value:.4f} (expected={expected_loss:.4f}, "
                f"deviation={deviation:.2f}σ), epoch={epoch}, severity={severity}, "
                f"severe_threshold={severe_count_threshold}"
            )

            # Track anomaly count for escalation
            state = self._get_or_create_state(config_key)
            if not hasattr(state, 'loss_anomaly_count'):
                state.loss_anomaly_count = 0
            state.loss_anomaly_count += 1
            state.last_loss_anomaly_time = time.time()

            # Trigger quality check on training data
            self._trigger_quality_check(config_key, reason="training_loss_anomaly")

            # If severe or consecutive anomalies exceed adaptive threshold, boost exploration
            # Early training is more permissive (5 anomalies), late training is strict (2 anomalies)
            if severity == "severe" or state.loss_anomaly_count >= severe_count_threshold:
                self._boost_exploration_for_anomaly(config_key, state.loss_anomaly_count)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling loss anomaly: {e}")

    def _on_training_loss_trend(self, event: Any) -> None:
        """Handle training loss trend events (Phase 8).

        Responds to ongoing trends in training loss (improving/stalled/degrading).

        Actions:
        - Stalled: Increase exploration diversity, consider curriculum rebalance
        - Degrading: Trigger quality check, potential pause
        - Improving: Reset anomaly counters, reduce exploration boost
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            trend = payload.get("trend", "unknown")  # improving, stalled, degrading
            current_loss = payload.get("current_loss", 0.0)
            trend_duration_epochs = payload.get("trend_duration_epochs", 0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            logger.info(
                f"[FeedbackLoopController] Training loss trend for {config_key}: "
                f"trend={trend}, loss={current_loss:.4f}, duration={trend_duration_epochs} epochs"
            )

            if trend == "stalled":
                # Training has stagnated - boost exploration to generate diverse data
                if trend_duration_epochs >= TREND_DURATION_SEVERE:
                    self._boost_exploration_for_stall(config_key, trend_duration_epochs)

            elif trend == "plateau":
                # Dec 29, 2025: Plateau detected - boost exploration to escape
                # The PLATEAU_DETECTED event handler provides more detailed response
                self._boost_exploration_for_stall(config_key, trend_duration_epochs or 10)
                logger.info(
                    f"[FeedbackLoopController] Plateau trend for {config_key}, "
                    f"boosting exploration"
                )

            elif trend == "degrading":
                # Loss is getting worse - check data quality, consider rollback
                self._trigger_quality_check(config_key, reason="training_loss_degrading")
                if trend_duration_epochs >= TREND_DURATION_MODERATE:
                    logger.warning(
                        f"[FeedbackLoopController] Sustained loss degradation for {config_key}, "
                        f"consider training pause or rollback"
                    )

            elif trend == "improving":
                # Good news - reset anomaly tracking
                if hasattr(state, 'loss_anomaly_count'):
                    state.loss_anomaly_count = 0
                # Optionally reduce exploration boost since training is on track
                self._reduce_exploration_after_improvement(config_key)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling loss trend: {e}")

    def _on_plateau_detected(self, event: Any) -> None:
        """Handle training plateau by boosting exploration aggressively.

        Dec 29, 2025: Implements exploration boost based on plateau type.
        Jan 2026 Sprint 10: AGGRESSIVE plateau breaking for faster Elo gains.
        - Overfitting: 2.0x exploration boost + temperature increase + quality boost
        - Data limitation: 1.8x exploration boost + quality boost + request more games

        Closes feedback loop: PLATEAU_DETECTED → exploration boost → SelfplayScheduler
        Expected improvement: +5-10 Elo per config from faster plateau recovery.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event

            config_key = extract_config_key(payload)
            plateau_type = payload.get("plateau_type", "data_limitation")
            # Jan 2026 Sprint 10: More aggressive default boost
            exploration_boost = payload.get("exploration_boost", 1.8)
            train_val_gap = payload.get("train_val_gap", 0.0)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)

            # Track plateau count for escalation
            if not hasattr(state, "plateau_count"):
                state.plateau_count = 0
            state.plateau_count += 1
            state.last_plateau_time = time.time()

            # Jan 2026 Sprint 10: Scale boost with plateau count for persistent plateaus
            plateau_multiplier = 1.0 + (state.plateau_count * 0.2)  # 1.2x, 1.4x, 1.6x...
            plateau_multiplier = min(plateau_multiplier, 2.0)  # Cap at 2x

            # Apply exploration boost (scaled by plateau count)
            final_exploration_boost = exploration_boost * plateau_multiplier
            state.exploration_boost = final_exploration_boost
            state.exploration_boost_expires_at = time.time() + 3600  # 1 hour

            if plateau_type == "overfitting":
                # High train/val gap indicates overfitting - aggressive diversity boost
                state.selfplay_temperature_boost = 1.3 + (state.plateau_count * 0.1)
                state.selfplay_temperature_boost = min(state.selfplay_temperature_boost, 1.8)
                logger.info(
                    f"[FeedbackLoopController] Plateau (overfitting) for {config_key}: "
                    f"exploration_boost={final_exploration_boost:.2f}, "
                    f"temp_boost={state.selfplay_temperature_boost:.2f}, "
                    f"train_val_gap={train_val_gap:.4f}, plateau_count={state.plateau_count}"
                )
            else:
                # Data-limited plateau - request more high-quality games
                state.games_multiplier = 1.5 + (state.plateau_count * 0.2)
                state.games_multiplier = min(state.games_multiplier, 2.5)
                logger.info(
                    f"[FeedbackLoopController] Plateau (data limited) for {config_key}: "
                    f"exploration_boost={final_exploration_boost:.2f}, "
                    f"games_multiplier={state.games_multiplier:.2f}, "
                    f"plateau_count={state.plateau_count}"
                )

            # Emit EXPLORATION_BOOST event for SelfplayScheduler
            try:
                from app.coordination.event_router import emit_exploration_boost

                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config_key,
                        boost_factor=final_exploration_boost,
                        reason="plateau",
                        anomaly_count=state.plateau_count,  # Signal plateau severity
                        source="FeedbackLoopController",
                    ),
                    context=f"emit_exploration_boost:plateau:{config_key}",
                )
                logger.debug(
                    f"[FeedbackLoopController] Emitted EXPLORATION_BOOST event for {config_key}"
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f"[FeedbackLoopController] Failed to emit EXPLORATION_BOOST: {e}")

            # Jan 2026 Sprint 10: Emit TRAINING_BLOCKED_BY_QUALITY to trigger quality boost
            # This increases Gumbel budget in SelfplayScheduler for higher quality games
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "TRAINING_BLOCKED_BY_QUALITY",
                {
                    "config_key": config_key,
                    "quality_score": 0.5 - (state.plateau_count * 0.1),  # Lower score with more plateaus
                    "threshold": 0.7,
                    "reason": f"plateau_{plateau_type}",
                },
                context="FeedbackLoopController",
                log_after=f"Triggered quality boost for {config_key} (plateau)",
            )

            # If repeated plateaus, consider triggering hyperparameter search
            if state.plateau_count >= 3:
                logger.warning(
                    f"[FeedbackLoopController] Repeated plateaus ({state.plateau_count}) "
                    f"for {config_key}, triggering aggressive hyperparameter search"
                )
                # Jan 2026 Sprint 10: Emit hyperparameter update request
                self._trigger_hyperparameter_search(config_key, state)

            # Jan 2026 Sprint 10: Start curriculum advancement earlier (after 1 plateau)
            # to provide harder training data and break out of the plateau faster
            if state.plateau_count >= 1:
                self._advance_curriculum_on_velocity_plateau(config_key, state)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Error handling plateau: {e}")

    def _advance_curriculum_on_velocity_plateau(
        self, config_key: str, state: FeedbackState
    ) -> None:
        """Advance curriculum tier when velocity indicates persistent plateau.

        Dec 29, 2025: Implements velocity-based curriculum advancement to break
        out of training plateaus. When Elo velocity is low and we've had
        repeated plateaus, we advance to a harder curriculum tier.

        Curriculum tiers:
        - 0: Beginner (basic positions, weaker opponents)
        - 1: Intermediate (moderate complexity)
        - 2: Advanced (complex positions, stronger opponents)
        - 3: Expert (most challenging positions)

        Benefits:
        - Harder training data forces model to learn new strategies
        - Breaks out of local optima caused by repetitive training data
        - Provides progressive difficulty as model improves
        """
        try:
            # Check velocity - only advance if we're truly plateauing
            velocity = state.elo_velocity
            is_low_velocity = velocity < ELO_PLATEAU_PER_HOUR and len(state.elo_history or []) >= 3

            if not is_low_velocity:
                logger.debug(
                    f"[FeedbackLoopController] Velocity {velocity:.1f} Elo/hr not low enough "
                    f"for curriculum advancement ({config_key})"
                )
                return

            # Check cooldown - don't advance too frequently (min 2 hours between advances)
            cooldown_seconds = 7200  # 2 hours
            time_since_advance = time.time() - state.curriculum_last_advanced
            if time_since_advance < cooldown_seconds:
                logger.debug(
                    f"[FeedbackLoopController] Curriculum advancement on cooldown "
                    f"({time_since_advance:.0f}s < {cooldown_seconds}s) for {config_key}"
                )
                return

            # Check max tier - don't exceed expert level
            max_tier = 3
            if state.curriculum_tier >= max_tier:
                logger.debug(
                    f"[FeedbackLoopController] Already at max curriculum tier "
                    f"({state.curriculum_tier}) for {config_key}"
                )
                return

            # Advance the curriculum tier
            old_tier = state.curriculum_tier
            new_tier = old_tier + 1
            state.curriculum_tier = new_tier
            state.curriculum_last_advanced = time.time()

            # Reset plateau count after advancement
            state.plateau_count = 0

            tier_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
            logger.info(
                f"[FeedbackLoopController] Curriculum advancement for {config_key}: "
                f"{tier_names[old_tier]} -> {tier_names[new_tier]} "
                f"(velocity={velocity:.1f} Elo/hr, plateaus={state.plateau_count})"
            )

            # Emit CURRICULUM_ADVANCED event for downstream consumers
            from app.coordination.event_router import safe_emit_event

            safe_emit_event(
                "CURRICULUM_ADVANCED",
                {
                    "config_key": config_key,
                    "old_tier": old_tier,
                    "new_tier": new_tier,
                    "trigger": "velocity_plateau",
                    "velocity": velocity,
                    "plateau_count": state.plateau_count,
                    "source": "FeedbackLoopController",
                },
                log_after=f"[FeedbackLoopController] Emitted CURRICULUM_ADVANCED for {config_key}",
                log_level=logging.DEBUG,
                context="FeedbackLoopController._check_velocity_plateau",
            )

            # Also notify CurriculumFeedback to adjust weights
            try:
                from app.training.curriculum_feedback import get_curriculum_feedback

                feedback = get_curriculum_feedback()
                if feedback and hasattr(feedback, "set_difficulty_tier"):
                    feedback.set_difficulty_tier(config_key, new_tier)
                    logger.debug(
                        f"[FeedbackLoopController] Updated CurriculumFeedback tier for {config_key}"
                    )
            except ImportError:
                logger.debug("[FeedbackLoopController] curriculum_feedback not available")
            except (AttributeError, TypeError) as cf_err:
                logger.debug(f"[FeedbackLoopController] CurriculumFeedback error: {cf_err}")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.warning(
                f"[FeedbackLoopController] Error advancing curriculum for {config_key}: {e}"
            )

    def _trigger_hyperparameter_search(self, config_key: str, state: FeedbackState) -> None:
        """Trigger hyperparameter search for persistently plateaued configs.

        Jan 2026 Sprint 10: When a config has 3+ consecutive plateaus, trigger
        aggressive hyperparameter adjustments to break out of local minima.

        Adjustments:
        - Increase learning rate by 50% temporarily (shake out of local minimum)
        - Reduce batch size by 25% (more gradient updates)
        - Enable cosine annealing if not already active
        - Emit HYPERPARAMETER_UPDATED event for downstream consumers

        Args:
            config_key: Config key (e.g., "hex8_2p")
            state: FeedbackState for the config
        """
        try:
            # Calculate hyperparameter adjustments based on plateau severity
            plateau_count = getattr(state, "plateau_count", 1)
            lr_boost = 1.0 + (plateau_count * 0.15)  # 1.15x, 1.30x, 1.45x per plateau
            lr_boost = min(lr_boost, 2.0)  # Cap at 2x

            batch_reduction = max(0.5, 1.0 - (plateau_count * 0.08))  # 0.92, 0.84, 0.76...

            logger.info(
                f"[FeedbackLoopController] Hyperparameter search for {config_key}: "
                f"lr_boost={lr_boost:.2f}x, batch_reduction={batch_reduction:.2f}x, "
                f"plateau_count={plateau_count}"
            )

            # Emit HYPERPARAMETER_UPDATED event for training to pick up
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "HYPERPARAMETER_UPDATED",
                {
                    "config_key": config_key,
                    "learning_rate_multiplier": lr_boost,
                    "batch_size_multiplier": batch_reduction,
                    "enable_cosine_annealing": True,
                    "reason": f"plateau_count_{plateau_count}",
                    "source": "FeedbackLoopController",
                },
                context="FeedbackLoopController",
                log_after=f"Emitted HYPERPARAMETER_UPDATED for {config_key}",
            )

            # Also update state to track that we triggered hyperparam search
            state.last_hyperparam_search = time.time()
            state.hyperparam_search_count = getattr(state, "hyperparam_search_count", 0) + 1

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(
                f"[FeedbackLoopController] Error triggering hyperparameter search: {e}"
            )

    def _on_quality_degraded_for_training(self, event: Any) -> None:
        """Handle QUALITY_DEGRADED events to adjust training thresholds (P1.1).

        When quality degrades, we want to train MORE to fix the problem.
        This reduces the training threshold via ImprovementOptimizer.

        Actions:
        - Record low data quality score in ImprovementOptimizer
        - This triggers faster training cycles (lower threshold)
        - Also boost exploration to gather more diverse data
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            # Dec 2025: Source tracking loop guard - skip events we emitted
            source = payload.get("source", "")
            if source in ("feedback_loop_controller", "FeedbackLoopController"):
                logger.debug("[FeedbackLoopController] Skipping self-emitted QUALITY_DEGRADED event")
                return

            config_key = extract_config_key(payload)
            quality_score = payload.get("quality_score", 0.5)
            threshold = payload.get("threshold", MEDIUM_QUALITY_THRESHOLD)

            if not config_key:
                return

            logger.info(
                f"[FeedbackLoopController] Quality degraded for {config_key}: "
                f"score={quality_score:.2f} < threshold={threshold:.2f}, "
                f"triggering training acceleration"
            )

            # Update ImprovementOptimizer to reduce training threshold
            try:
                from app.training.improvement_optimizer import ImprovementOptimizer

                optimizer = ImprovementOptimizer.get_instance()
                # Record low data quality - this reduces threshold_multiplier
                optimizer.record_data_quality(
                    config_key=config_key,
                    data_quality_score=quality_score,
                    parity_success_rate=quality_score,  # Use quality as proxy
                )
                logger.info(
                    f"[FeedbackLoopController] Updated ImprovementOptimizer for {config_key}: "
                    f"quality={quality_score:.2f}"
                )
            except ImportError:
                logger.debug("[FeedbackLoopController] ImprovementOptimizer not available")

            # Also boost exploration for this config
            self._boost_exploration_for_stall(config_key, trend_duration_epochs=3)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling quality degraded: {e}")

    def _trigger_quality_check(self, config_key: str, reason: str) -> None:
        """Trigger a quality check for the given config.

        Phase 9 (Dec 2025): Emits QUALITY_CHECK_REQUESTED event to be handled
        by QualityMonitorDaemon, completing the feedback loop from training
        loss anomalies to data quality verification.
        """
        try:
            import asyncio
            from app.coordination.event_router import emit_quality_check_requested

            logger.info(
                f"[FeedbackLoopController] Triggering quality check for {config_key}: {reason}"
            )

            # Determine priority based on reason
            priority = "high" if reason in ("training_loss_anomaly", "training_loss_degrading") else "normal"

            # Emit the event (handle both sync and async contexts)
            try:
                _safe_create_task(
                    emit_quality_check_requested(
                        config_key=config_key,
                        reason=reason,
                        source="FeedbackLoopController",
                        priority=priority,
                    ),
                    context=f"emit_quality_check_requested:{config_key}",
                )
            except RuntimeError:
                # No running event loop, run synchronously
                asyncio.run(emit_quality_check_requested(
                    config_key=config_key,
                    reason=reason,
                    source="FeedbackLoopController",
                    priority=priority,
                ))

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Error triggering quality check: {e}")

    # Sprint 17.9 (Jan 16, 2026): Exploration boost methods extracted to ExplorationBoostMixin
    # - _boost_exploration_for_anomaly()
    # - _boost_exploration_for_stall()
    # - _reduce_exploration_after_improvement()
    # See: app/coordination/feedback/exploration_boost.py

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion.

        Actions:
        1. Record training metrics
        2. Trigger evaluation if accuracy threshold met
        3. Adjust curriculum based on metrics
        4. Snapshot Elo for bandit feedback (Dec 29 2025)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            policy_accuracy = payload.get("policy_accuracy", 0.0)
            value_accuracy = payload.get("value_accuracy", 0.0)
            model_path = payload.get("model_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_training_time = time.time()
            state.last_training_accuracy = policy_accuracy

            # Dec 29 2025: Snapshot current Elo before evaluation
            # This allows computing Elo gain after evaluation for bandit feedback
            state.elo_before_training = state.last_elo

            logger.info(
                f"[FeedbackLoopController] Training complete for {config_key}: "
                f"policy_acc={policy_accuracy:.2%}, value_acc={value_accuracy:.2%}"
            )

            # Trigger evaluation if accuracy is good enough
            if policy_accuracy >= self.policy_accuracy_threshold:
                self._trigger_evaluation(config_key, model_path)

            # Record training in curriculum
            self._record_training_in_curriculum(config_key)

            # Emit curriculum adjustment event based on training metrics
            # Low accuracy = model needs more training data (boost weight)
            # High accuracy = model is learning well (maintain or reduce weight)
            self._emit_curriculum_training_feedback(
                config_key, policy_accuracy, value_accuracy
            )

            # Jan 2026: Record training feedback to Elo database
            # This populates the training_feedback table for Elo velocity tracking
            self._record_training_feedback_to_db(config_key, state)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling training complete: {e}")

    # Sprint 17.9 (Jan 16, 2026): Training curriculum methods extracted to TrainingCurriculumFeedbackMixin
    # Extracted methods (~122 LOC):
    # - _emit_curriculum_training_feedback(config_key, policy_accuracy, value_accuracy)
    # - _record_training_feedback_to_db(config_key, state)
    # See: app/coordination/feedback/training_curriculum_mixin.py

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle evaluation completion.

        Actions:
        1. Record evaluation results
        2. Track Elo velocity (Dec 28 2025)
        3. Adjust selfplay intensity based on velocity
        4. Compute and emit adaptive training signal (Dec 29 2025 - Phase 6)
        5. Report feedback to engine bandit (Dec 29 2025)
        6. Consider promotion if win rate threshold met
        7. Update training_history.final_elo (Jan 6 2026 - P1 improvement)
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            win_rate = payload.get("win_rate", 0.0)
            elo = payload.get("elo", 1500.0)
            model_path = payload.get("model_path", "")

            # Jan 6, 2026: P1 - Update training_history.final_elo
            # This closes the feedback loop between training and evaluation
            self._update_training_final_elo(config_key, elo)

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_evaluation_time = time.time()
            state.last_evaluation_win_rate = win_rate

            # Dec 28 2025: Track Elo velocity for plateau detection
            velocity = state.update_elo(elo)

            logger.info(
                f"[FeedbackLoopController] Evaluation complete for {config_key}: "
                f"win_rate={win_rate:.2%}, elo={elo:.0f}, velocity={velocity:.1f} Elo/hr"
            )

            # Dec 28 2025: Adjust selfplay based on velocity and Elo gap
            self._adjust_selfplay_for_velocity(config_key, state, elo, velocity)

            # Dec 29 2025: Phase 6 - Compute and emit adaptive training parameters
            eval_result = {"elo": elo, "win_rate": win_rate, "velocity": velocity}
            adaptive_signal = self._compute_adaptive_signal(config_key, state, eval_result)
            self._emit_adaptive_training_signal(config_key, adaptive_signal)

            # Dec 29 2025: Report feedback to selfplay engine bandit
            # This helps the bandit learn which engines produce best training data
            self._report_engine_bandit_feedback(config_key, state, elo)

            # Dec 29 2025: Consider promotion with Elo-adaptive threshold
            # Higher Elo models need to show stronger performance to be promoted
            adaptive_threshold = self._get_adaptive_promotion_threshold(elo, state)
            if win_rate >= adaptive_threshold:
                self._consider_promotion(config_key, model_path, win_rate, elo)
            else:
                # Log near-miss promotions for debugging
                margin = adaptive_threshold - win_rate
                if margin < 0.05:  # Within 5% of threshold
                    logger.info(
                        f"[FeedbackLoopController] Near-miss promotion for {config_key}: "
                        f"win_rate={win_rate:.2%} < threshold={adaptive_threshold:.2%} "
                        f"(margin={margin:.2%}, elo={elo:.0f})"
                    )

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling evaluation complete: {e}")

    # Sprint 17.9 (Jan 16, 2026): Elo velocity methods extracted to EloVelocityAdaptationMixin
    # Extracted methods (~166 LOC):
    # - _adjust_selfplay_for_velocity(config_key, state, elo, velocity)
    # - _emit_selfplay_adjustment(config_key, state, elo_gap, velocity)
    # - _compute_adaptive_signal(config_key, state, eval_result) -> AdaptiveTrainingSignal
    # - _emit_adaptive_training_signal(config_key, signal)
    # See: app/coordination/feedback/elo_velocity_mixin.py

    def _on_evaluation_failed(self, event: Any) -> None:
        """Handle evaluation failure.

        P10-LOOP-2 (Dec 2025): When evaluation fails after built-in retries,
        attempt secondary recovery:
        1. Track failure count per config
        2. If under threshold, re-queue evaluation with different settings
        3. If over threshold, signal need for fresh training data

        This closes the feedback loop:
        EVALUATION_FAILED → Retry with modifications → Eventual success OR data refresh
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = extract_config_key(payload)
            model_path = payload.get("model_path", "")
            error = payload.get("error", "unknown")
            retry_count = payload.get("retry_count", 0)

            if not config_key:
                logger.debug("[FeedbackLoopController] EVALUATION_FAILED missing config_key")
                return

            state = self._get_or_create_state(config_key)
            state.consecutive_failures += 1

            logger.warning(
                f"[FeedbackLoopController] Evaluation failed for {config_key}: "
                f"error={error}, retry_count={retry_count}, "
                f"consecutive_failures={state.consecutive_failures}"
            )

            # Decide recovery strategy based on failure count
            max_secondary_retries = 3

            if state.consecutive_failures <= max_secondary_retries:
                # Attempt secondary retry with modified parameters
                logger.info(
                    f"[FeedbackLoopController] Attempting secondary evaluation retry "
                    f"for {config_key} (attempt {state.consecutive_failures}/{max_secondary_retries})"
                )
                self._retry_evaluation(config_key, model_path, state.consecutive_failures)
            else:
                # Too many failures - signal need for fresh data
                logger.warning(
                    f"[FeedbackLoopController] Max retries exceeded for {config_key}. "
                    f"Signaling need for fresh training data."
                )

                # Boost exploration to generate more diverse data
                old_boost = state.current_exploration_boost
                state.current_exploration_boost = min(EXPLORATION_BOOST_MAX, old_boost * FAILURE_EXPLORATION_BOOST)

                # Emit event for selfplay boost
                from app.coordination.event_router import safe_emit_event

                safe_emit_event(
                    "SELFPLAY_TARGET_UPDATED",
                    {
                        "config_key": config_key,
                        "priority": "urgent",
                        "reason": "evaluation_failures_exceeded",
                        "exploration_boost": state.current_exploration_boost,
                    },
                    context="FeedbackLoopController._on_evaluation_failed",
                )

                # Reset failure counter after signaling
                state.consecutive_failures = 0

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling evaluation failed: {e}")

    def _calculate_regression_amplitude(
        self, elo_drop: float, consecutive_count: int, severity_str: str = "minor"
    ) -> tuple[float, int]:
        """Calculate response amplitude based on regression severity.

        January 3, 2026: Implements proportional response to regression severity.
        Larger Elo drops and repeated regressions trigger stronger responses.

        January 2026: Now uses RegressionDetector's severity directly instead of
        recalculating from elo_drop. This consolidates severity logic in one place.

        The amplitude formula:
        - Base boost: 1.5x (EXPLORATION_BOOST_RECOVERY)
        - Severity scaling: MINOR +0.0, MODERATE +0.2, SEVERE +0.35, CRITICAL +0.5
        - Consecutive scaling: +0.15x per consecutive regression

        Args:
            elo_drop: Magnitude of Elo regression (positive value)
            consecutive_count: Number of consecutive regression events (from RegressionDetector)
            severity_str: Severity level from RegressionDetector ("minor", "moderate", "severe", "critical")

        Returns:
            Tuple of (exploration_boost, target_games):
            - exploration_boost: Scaled boost factor (1.5x to 2.5x range)
            - target_games: Number of selfplay games to request (500 to 2000 range)
        """
        # Base values
        base_boost = EXPLORATION_BOOST_RECOVERY  # 1.5x
        base_games = 500

        # January 2026: Use severity from RegressionDetector instead of recalculating
        severity_boost_map = {
            "minor": 0.0,
            "moderate": 0.2,
            "severe": 0.35,
            "critical": 0.5,
        }
        severity_boost = severity_boost_map.get(severity_str, 0.0)

        # Scale by consecutive count (+0.15x each, capped at +0.5x)
        consecutive_boost = min(0.5, consecutive_count * 0.15)

        # Combined boost, capped at EXPLORATION_BOOST_MAX (2.0)
        exploration_boost = min(
            EXPLORATION_BOOST_MAX,
            base_boost + severity_boost + consecutive_boost
        )

        # Scale target games based on severity
        # Use severity-based multiplier instead of raw elo_drop calculation
        severity_game_multiplier = {
            "minor": 1.0,
            "moderate": 1.5,
            "severe": 2.0,
            "critical": 2.5,
        }
        severity_factor = severity_game_multiplier.get(severity_str, 1.0) + consecutive_count * 0.3
        target_games = min(2000, int(base_games * severity_factor))

        return exploration_boost, target_games

    def _on_regression_detected(self, event: Any) -> None:
        """Handle REGRESSION_DETECTED event.

        Dec 2025: When regression is detected, trigger:
        1. Exploration boost (1.5x to generate more diverse data)
        2. SELFPLAY_TARGET_UPDATED event to request additional games
        3. Log the action for monitoring

        January 3, 2026: Added amplitude scaling - response intensity is now
        proportional to regression severity (Elo drop magnitude and consecutive count).

        January 3, 2026: Added deduplication to prevent duplicate handling when
        both FeedbackLoopController and UnifiedFeedbackOrchestrator are running.

        This closes the feedback loop: REGRESSION_DETECTED → exploration boost →
        more diverse selfplay → better training data → recovery.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event if isinstance(event, dict) else {}

            # January 3, 2026: Check for duplicate event (via HandlerBase)
            if self._is_duplicate_event(payload):
                logger.debug("[FeedbackLoopController] Skipping duplicate REGRESSION_DETECTED")
                return

            config_key = extract_config_key(payload)
            elo_drop = payload.get("elo_drop", 0.0)
            # January 2026: Use consecutive_count from RegressionDetector (single source of truth)
            consecutive_count = payload.get("consecutive_count", payload.get("consecutive_regressions", 1))
            severity_str = payload.get("severity", "minor")

            if not config_key:
                logger.debug("[FeedbackLoopController] REGRESSION_DETECTED missing config_key")
                return

            state = self._get_or_create_state(config_key)
            # Sync with RegressionDetector's consecutive count instead of maintaining separate tracking
            state.consecutive_failures = consecutive_count

            # Jan 3, 2026: Calculate amplitude-scaled response based on severity
            # January 2026: Pass severity for severity-aware amplitude calculation
            exploration_boost, target_games = self._calculate_regression_amplitude(
                elo_drop, consecutive_count, severity_str
            )

            logger.warning(
                f"[FeedbackLoopController] Regression detected for {config_key}: "
                f"elo_drop={elo_drop:.0f}, consecutive={consecutive_count}, severity={severity_str}, "
                f"total_failures={state.consecutive_failures}, "
                f"amplitude_boost={exploration_boost:.2f}x, target_games={target_games}"
            )

            # Set exploration boost to generate more diverse data
            # Use max to preserve any higher existing boost
            old_boost = state.current_exploration_boost
            new_boost = max(state.current_exploration_boost, exploration_boost)
            state.current_exploration_boost = new_boost

            logger.info(
                f"[FeedbackLoopController] Increased exploration boost for {config_key}: "
                f"{old_boost:.2f}x → {new_boost:.2f}x (regression amplitude response)"
            )

            # Emit SELFPLAY_TARGET_UPDATED to request more diverse selfplay games
            if HAS_SELFPLAY_EVENTS and emit_selfplay_target_updated:

                try:
                    # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                    loop = asyncio.get_running_loop()
                    task = _safe_create_task(
                        emit_selfplay_target_updated(
                            config_key=config_key,
                            target_games=target_games,
                            reason=f"regression_detected_elo_drop_{elo_drop:.0f}",
                            priority=2,  # High priority
                            source="feedback_loop_controller.py",
                        ),
                        f"emit_selfplay_target_updated_regression({config_key})",
                    )
                    if task:
                        logger.info(
                            f"[FeedbackLoopController] Emitted SELFPLAY_TARGET_UPDATED for {config_key}: "
                            f"{target_games} games (exploration_boost={exploration_boost:.1f}x, priority=2)"
                        )
                except RuntimeError:
                    logger.debug("[FeedbackLoopController] No event loop for SELFPLAY_TARGET_UPDATED")

            # Emit EXPLORATION_BOOST event for temperature schedulers
            try:
                from app.coordination.event_router import emit_exploration_boost

                _safe_create_task(emit_exploration_boost(
                    config_key=config_key,
                    boost_factor=new_boost,
                    reason="regression_detected",
                    anomaly_count=consecutive_count,
                    source="FeedbackLoopController",
                ), f"emit_exploration_boost_regression({config_key})")
                logger.debug(
                    f"[FeedbackLoopController] Emitted EXPLORATION_BOOST event for {config_key}"
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug(f"[FeedbackLoopController] Failed to emit EXPLORATION_BOOST: {e}")

            # January 3, 2026 (Sprint 12 P1): Curriculum tier rollback for major regressions
            # When Elo drops significantly, roll back to a previous curriculum tier to allow
            # the model to relearn from simpler positions before advancing again.
            if elo_drop > 50 and state.curriculum_tier > 0:
                old_tier = state.curriculum_tier
                new_tier = max(0, old_tier - 1)  # Roll back one tier
                state.curriculum_tier = new_tier

                tier_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
                logger.warning(
                    f"[FeedbackLoopController] Curriculum rollback for {config_key}: "
                    f"{tier_names[old_tier]} → {tier_names[new_tier]} "
                    f"(elo_drop={elo_drop:.0f} > 50 threshold)"
                )

                # Emit CURRICULUM_ROLLBACK event for downstream consumers
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        from app.coordination.event_router import DataEvent

                        event = DataEvent(
                            event_type=DataEventType.CURRICULUM_ROLLBACK
                            if hasattr(DataEventType, "CURRICULUM_ROLLBACK")
                            else DataEventType.CURRICULUM_ADVANCED,
                            payload={
                                "config_key": config_key,
                                "old_tier": old_tier,
                                "new_tier": new_tier,
                                "trigger": "regression_detected",
                                "elo_drop": elo_drop,
                                "consecutive_regressions": consecutive_regressions,
                                "direction": "rollback",
                                "source": "FeedbackLoopController",
                            },
                            source="FeedbackLoopController",
                        )
                        _safe_create_task(
                            bus.publish(event),
                            context=f"emit_curriculum_rollback:{config_key}",
                        )
                        logger.info(
                            f"[FeedbackLoopController] Emitted CURRICULUM_ROLLBACK for {config_key}"
                        )
                except (AttributeError, TypeError, ImportError, RuntimeError) as emit_err:
                    logger.debug(
                        f"[FeedbackLoopController] Failed to emit curriculum event: {emit_err}"
                    )

                # Trigger gauntlet evaluation against all baselines to reassess model strength
                # This ensures we have fresh Elo data after the regression
                try:
                    self._trigger_gauntlet_all_baselines(config_key)
                except (AttributeError, TypeError, RuntimeError) as e:
                    logger.debug(f"[FeedbackLoopController] Failed to trigger gauntlet: {e}")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling regression detected: {e}")

    def _retry_evaluation(self, config_key: str, model_path: str, attempt: int) -> None:
        """Retry evaluation with modified parameters.

        P10-LOOP-2 (Dec 2025): Secondary retry logic for failed evaluations.
        Modifies parameters based on attempt number:
        - Attempt 1: Increase num_games by 50%
        - Attempt 2: Add delay and increase games by 100%
        - Attempt 3: Maximum games, longer delay
        """
        try:
            # Parse config key using canonical utility
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.debug(f"[FeedbackLoopController] Invalid config_key format: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            # Adjust parameters based on attempt
            base_games = 100
            delay = 0.0
            if attempt == 1:
                num_games = int(base_games * 1.5)  # 150 games
                delay = 2.0
            elif attempt == 2:
                num_games = int(base_games * 2.0)  # 200 games
                delay = 5.0
            else:
                num_games = int(base_games * 2.5)  # 250 games
                delay = 10.0

            async def _do_retry():
                import asyncio
                if delay > 0:
                    await asyncio.sleep(delay)

                from app.coordination.pipeline_actions import trigger_evaluation

                logger.info(
                    f"[FeedbackLoopController] Retrying evaluation for {config_key}: "
                    f"num_games={num_games}, delay={delay}s"
                )

                result = await trigger_evaluation(
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=num_games,
                    max_retries=2,  # Reduced retries for secondary attempt
                )

                if result.success:
                    logger.info(f"[FeedbackLoopController] Secondary eval succeeded for {config_key}")
                else:
                    logger.warning(f"[FeedbackLoopController] Secondary eval failed for {config_key}")

            _safe_create_task(_do_retry(), f"retry_evaluation:{config_key}")

        except (AttributeError, TypeError, RuntimeError, asyncio.CancelledError) as e:
            logger.error(f"[FeedbackLoopController] Error setting up evaluation retry: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion.

        Actions:
        1. Record promotion outcome
        2. Adjust exploration rate based on outcome
        3. Update training intensity based on outcome
        4. Signal curriculum adjustments
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}
            metadata = payload.get("metadata", payload)

            config_key = metadata.get("config") or metadata.get("config_key", "")
            promoted = metadata.get("promoted", False)
            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_promotion_time = time.time()
            state.last_promotion_success = promoted

            if promoted:
                state.consecutive_successes += 1
                state.consecutive_failures = 0
                logger.info(
                    f"[FeedbackLoopController] Promotion SUCCESS for {config_key} "
                    f"(streak: {state.consecutive_successes})"
                )

                # December 2025 - Phase 2C.4: HOT_PATH after 3+ consecutive successes
                # The model is performing well consistently - maximize training speed
                if state.consecutive_successes >= 3:
                    state.current_training_intensity = "hot_path"
                    logger.info(
                        f"[FeedbackLoopController] HOT_PATH activated for {config_key} "
                        f"(3+ consecutive successes)"
                    )
                else:
                    # Still improving - use accelerated training
                    state.current_training_intensity = "accelerated"

                state.current_exploration_boost = max(EXPLORATION_BOOST_BASE, state.current_exploration_boost - EXPLORATION_BOOST_SUCCESS_DECREMENT)

            else:
                state.consecutive_failures += 1
                state.consecutive_successes = 0
                logger.info(
                    f"[FeedbackLoopController] Promotion FAILED for {config_key} "
                    f"(streak: {state.consecutive_failures})"
                )

                # Boost exploration and intensity on failure
                state.current_exploration_boost = min(EXPLORATION_BOOST_MAX, state.current_exploration_boost + EXPLORATION_BOOST_FAILURE_INCREMENT)

                if state.consecutive_failures >= 3:
                    state.current_training_intensity = "hot_path"
                else:
                    state.current_training_intensity = "accelerated"

                # Signal urgent training needed
                self._signal_urgent_training(config_key, state.consecutive_failures)

            # Apply feedback to FeedbackAccelerator
            self._apply_intensity_feedback(config_key, state)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling promotion complete: {e}")

    def _on_work_completed(self, event: Any) -> None:
        """Handle work completion events.

        Tracks work queue completion metrics for monitoring and statistics.
        This prevents silent failures when work completes but no handler processes it.

        Actions:
        1. Log work completion for visibility
        2. Update completion statistics per config
        3. Track timing for throughput monitoring
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            work_id = payload.get("work_id", "")
            work_type = payload.get("work_type", "unknown")
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            claimed_by = payload.get("claimed_by", "")

            # Build config key if we have board type and num players
            config_key = ""
            if board_type and num_players:
                config_key = make_config_key(board_type, num_players)

            logger.info(
                f"[FeedbackLoopController] Work completed: "
                f"id={work_id}, type={work_type}, config={config_key or 'N/A'}, node={claimed_by}"
            )

            # Update metrics if we have a config
            if config_key:
                state = self._get_or_create_state(config_key)
                state.work_completed_count += 1
                state.last_work_completion_time = time.time()

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Error handling work completed: {e}")

    def _on_work_failed(self, event: Any) -> None:
        """Handle work failure events (Phase 27 - December 2025).

        Tracks work queue failure metrics for monitoring and alerting.
        Enables:
        1. Failure rate monitoring per config
        2. Node health correlation
        3. Automatic selfplay rate adjustment on high failure rates
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            work_id = payload.get("work_id", "")
            work_type = payload.get("work_type", "unknown")
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            node_id = payload.get("node_id", "")
            reason = payload.get("reason", "unknown")

            config_key = ""
            if board_type and num_players:
                config_key = make_config_key(board_type, num_players)

            logger.warning(
                f"[FeedbackLoopController] Work failed: "
                f"id={work_id}, type={work_type}, config={config_key or 'N/A'}, "
                f"node={node_id}, reason={reason}"
            )

            # Update metrics if we have a config
            if config_key:
                state = self._get_or_create_state(config_key)
                if not hasattr(state, 'work_failed_count'):
                    state.work_failed_count = 0
                state.work_failed_count += 1

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Error handling work failed: {e}")

    def _on_work_timeout(self, event: Any) -> None:
        """Handle work timeout events (Phase 27 - December 2025).

        Tracks work queue timeout metrics for monitoring and alerting.
        Timeouts may indicate:
        1. Node health issues
        2. Network connectivity problems
        3. Resource exhaustion on target node
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            work_id = payload.get("work_id", "")
            work_type = payload.get("work_type", "unknown")
            board_type = payload.get("board_type", "")
            num_players = payload.get("num_players", 0)
            node_id = payload.get("node_id", "")
            timeout_seconds = payload.get("timeout_seconds", 0)

            config_key = ""
            if board_type and num_players:
                config_key = make_config_key(board_type, num_players)

            logger.warning(
                f"[FeedbackLoopController] Work timed out: "
                f"id={work_id}, type={work_type}, config={config_key or 'N/A'}, "
                f"node={node_id}, timeout={timeout_seconds}s"
            )

            # Update metrics if we have a config
            if config_key:
                state = self._get_or_create_state(config_key)
                if not hasattr(state, 'work_timeout_count'):
                    state.work_timeout_count = 0
                state.work_timeout_count += 1

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Error handling work timeout: {e}")

    # =========================================================================
    # Internal Actions
    # =========================================================================

    # Sprint 17.9 (Jan 16, 2026): Quality feedback methods extracted to QualityFeedbackMixin
    # Extracted methods (~283 LOC):
    # - _assess_selfplay_quality(db_path, games_count) -> quality score
    # - _assess_selfplay_quality_async(db_path, games_count) -> quality score
    # - _compute_intensity_from_quality(quality_score) -> intensity level
    # - _update_training_intensity(config_key, quality_score)
    # - _update_curriculum_weight_from_selfplay(config_key, quality_score)
    # - _signal_training_ready(config_key, quality_score)
    # - _emit_quality_degraded(config_key, quality_score, threshold, previous_score)
    # See: app/coordination/feedback/quality_feedback.py

    def _trigger_evaluation(self, config_key: str, model_path: str) -> None:
        """Trigger multi-harness gauntlet evaluation automatically after training.

        December 2025: Wires TRAINING_COMPLETED → auto-gauntlet evaluation.
        This closes the training feedback loop by automatically evaluating
        newly trained models against baselines under ALL compatible harnesses.

        The gauntlet results determine whether the model should be promoted
        to production or if more training is needed. Multi-harness evaluation
        enables finding the best (model, harness) combination for deployment.
        """
        logger.info(f"[FeedbackLoopController] Triggering multi-harness evaluation for {config_key}")

        try:
            # Parse config_key using canonical utility
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.warning(f"[FeedbackLoopController] Invalid config_key format: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            # Launch multi-harness gauntlet evaluation asynchronously
            async def run_multi_harness_gauntlet():
                """Run multi-harness gauntlet evaluation for the trained model.

                December 2025: Evaluates model under ALL compatible harnesses
                (e.g., policy_only, mcts, gumbel_mcts, descent for NN models).
                Registers all harness results in Elo system and uses the best
                harness for promotion decisions.

                Falls back to single-harness evaluation if multi-harness unavailable.
                """
                try:
                    from app.training.multi_harness_gauntlet import (
                        MultiHarnessGauntlet,
                        register_multi_harness_results,
                    )

                    # January 3, 2026 (Session 8): Wait for model distribution before evaluation
                    # This prevents wasting 45-60s evaluating with stale baseline models
                    # on cluster nodes that haven't received the new model yet.
                    try:
                        from app.coordination.unified_distribution_daemon import (
                            wait_for_model_availability,
                        )
                        from app.config.coordination_defaults import DistributionDefaults

                        # Wait for model to be distributed to at least MIN_NODES_FOR_PROMOTION nodes
                        # with a reasonable timeout (default 180s for distribution + buffer)
                        distribution_timeout = getattr(
                            DistributionDefaults, "DISTRIBUTION_TIMEOUT_SECONDS", 180.0
                        )
                        success, node_count = await wait_for_model_availability(
                            model_path=model_path,
                            min_nodes=getattr(DistributionDefaults, "MIN_NODES_FOR_PROMOTION", 3),
                            timeout=distribution_timeout,
                        )

                        if success:
                            logger.info(
                                f"[FeedbackLoopController] Model {config_key} distributed to "
                                f"{node_count} nodes, proceeding with evaluation"
                            )
                        else:
                            logger.warning(
                                f"[FeedbackLoopController] Model {config_key} only on {node_count} nodes "
                                f"after {distribution_timeout}s, proceeding anyway"
                            )
                    except ImportError:
                        logger.debug(
                            "[FeedbackLoopController] Distribution verification not available, "
                            "proceeding with evaluation"
                        )

                    gauntlet = MultiHarnessGauntlet(default_games_per_baseline=30)
                    result = await gauntlet.evaluate_model(
                        model_path=model_path,
                        board_type=board_type,
                        num_players=num_players,
                    )

                    # Register all harness results in Elo system
                    participant_ids = register_multi_harness_results(result)
                    logger.info(
                        f"[FeedbackLoopController] Multi-harness evaluation complete for {config_key}: "
                        f"best={result.best_harness} Elo={result.best_elo:.0f}, "
                        f"harnesses={list(result.harness_results.keys())}"
                    )

                    # Use best harness result for promotion decision
                    if result.best_elo > 0 and result.harness_results:
                        best_rating = result.harness_results.get(result.best_harness)
                        if best_rating:
                            win_rate = getattr(best_rating, "win_rate", 0.0)
                            elo_delta = result.best_elo - 1000  # Delta from baseline
                            if win_rate >= 0.55:  # Minimum threshold for promotion consideration
                                self._consider_promotion(
                                    config_key,
                                    model_path,
                                    win_rate,
                                    elo_delta,
                                )

                except ImportError as e:
                    # Fall back to single-harness evaluation if multi-harness unavailable
                    logger.debug(
                        f"[FeedbackLoopController] MultiHarnessGauntlet not available, "
                        f"falling back to single-harness: {e}"
                    )
                    await self._run_single_harness_gauntlet(
                        config_key, model_path, board_type, num_players
                    )
                except Exception as e:
                    logger.error(
                        f"[FeedbackLoopController] Multi-harness evaluation failed for {config_key}: {e}"
                    )

            _safe_create_task(run_multi_harness_gauntlet(), f"run_multi_harness_gauntlet({config_key})")

        except ImportError as e:
            logger.debug(f"[FeedbackLoopController] trigger_evaluation not available: {e}")

    async def _run_single_harness_gauntlet(
        self, config_key: str, model_path: str, board_type: str, num_players: int
    ) -> None:
        """Run single-harness gauntlet as fallback when MultiHarnessGauntlet unavailable.

        This is the legacy evaluation path using pipeline_actions.trigger_evaluation.
        Used when the multi-harness gauntlet is not available (e.g., missing dependencies).

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            model_path: Path to model checkpoint
            board_type: Board type string
            num_players: Number of players
        """
        try:
            from app.coordination.pipeline_actions import trigger_evaluation

            result = await trigger_evaluation(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                num_games=50,  # Standard gauntlet size
            )
            if result.success:
                logger.info(
                    f"[FeedbackLoopController] Single-harness gauntlet passed for {config_key}: "
                    f"eligible={result.metadata.get('promotion_eligible')}"
                )
                if result.metadata.get("promotion_eligible"):
                    self._consider_promotion(
                        config_key,
                        model_path,
                        result.metadata.get("win_rates", {}).get("heuristic", 0) / 100,
                        result.metadata.get("elo_delta", 0),
                    )
            else:
                logger.warning(
                    f"[FeedbackLoopController] Single-harness gauntlet failed for {config_key}: "
                    f"{result.error or 'unknown error'}"
                )
        except ImportError:
            logger.debug("[FeedbackLoopController] trigger_evaluation not available for fallback")
        except Exception as e:
            logger.error(f"[FeedbackLoopController] Single-harness fallback failed: {e}")

    def _trigger_gauntlet_all_baselines(self, config_key: str) -> None:
        """Trigger gauntlet evaluation against all baselines after regression.

        January 3, 2026 (Sprint 12 P1): When a major regression is detected,
        we trigger a comprehensive gauntlet evaluation against ALL baseline
        opponents (random, heuristic, and any other canonical models) to
        reassess the model's true strength.

        This helps detect if the regression was a fluke or if the model
        genuinely needs more training. Fresh Elo data guides curriculum
        and training decisions.
        """
        logger.info(
            f"[FeedbackLoopController] Triggering all-baseline gauntlet for {config_key} "
            f"(post-regression reassessment)"
        )

        try:
            parsed = parse_config_key(config_key)
            if not parsed:
                logger.warning(f"[FeedbackLoopController] Invalid config_key format: {config_key}")
                return
            board_type = parsed.board_type
            num_players = parsed.num_players

            # Find the current canonical model for this config
            try:
                from app.models.discovery import get_canonical_model_path

                model_path = get_canonical_model_path(board_type, num_players)
                if not model_path:
                    logger.debug(
                        f"[FeedbackLoopController] No canonical model found for {config_key}"
                    )
                    return
            except ImportError:
                logger.debug("[FeedbackLoopController] Model discovery not available")
                return

            # Use _trigger_evaluation which handles multi-harness gauntlet
            self._trigger_evaluation(config_key, str(model_path))

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"[FeedbackLoopController] Failed to trigger gauntlet: {e}")

    def _record_training_in_curriculum(self, config_key: str) -> None:
        """Record training completion in curriculum."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            feedback.record_training(config_key)
        except ImportError:
            pass

    def _get_adaptive_promotion_threshold(
        self, elo: float, state: FeedbackState
    ) -> float:
        """Get Elo-adaptive promotion threshold.

        Dec 29, 2025: Higher Elo models need stronger win rates to be promoted.
        This prevents promoting models that are only marginally better at high
        Elo levels, ensuring real progress before promotion.

        Elo-based threshold tiers:
        - Beginner (< 1300): 0.55 (55%) - easier to promote beginners
        - Intermediate (1300-1600): 0.60 (60%) - standard threshold
        - Advanced (1600-1800): 0.65 (65%) - need to show real progress
        - Elite (> 1800): 0.70 (70%) - near target, must be clearly better

        Additional modifiers:
        - Consecutive successes (3+): -0.03 (reward momentum)
        - Consecutive failures (2+): +0.02 (require more evidence)
        - Fast velocity (improving): -0.02 (capitalize on momentum)
        - Plateau (low velocity): +0.02 (wait for real improvement)

        Returns:
            Adaptive promotion threshold (0.50 - 0.75 range)
        """
        # Base threshold based on Elo tier
        if elo < 1300:
            base_threshold = 0.55  # Beginner tier - easier promotion
        elif elo < 1600:
            base_threshold = 0.60  # Intermediate tier - standard
        elif elo < 1800:
            base_threshold = 0.65  # Advanced tier - harder
        else:
            base_threshold = 0.70  # Elite tier - hardest

        # Apply modifiers
        modifier = 0.0

        # Momentum modifier: reward consecutive successes
        if state.consecutive_successes >= 3:
            modifier -= 0.03
            logger.debug(
                f"[FeedbackLoopController] Threshold modifier -0.03 for "
                f"{state.consecutive_successes} consecutive successes"
            )

        # Caution modifier: require more evidence after failures
        if state.consecutive_failures >= 2:
            modifier += 0.02
            logger.debug(
                f"[FeedbackLoopController] Threshold modifier +0.02 for "
                f"{state.consecutive_failures} consecutive failures"
            )

        # Velocity modifier: adjust based on improvement speed
        if len(state.elo_history or []) >= 3:
            velocity = state.elo_velocity
            if velocity > ELO_FAST_IMPROVEMENT_PER_HOUR:
                modifier -= 0.02  # Fast improvement - capitalize
                logger.debug(
                    f"[FeedbackLoopController] Threshold modifier -0.02 for "
                    f"fast velocity ({velocity:.1f} Elo/hr)"
                )
            elif velocity < ELO_PLATEAU_PER_HOUR:
                modifier += 0.02  # Plateau - wait for real improvement
                logger.debug(
                    f"[FeedbackLoopController] Threshold modifier +0.02 for "
                    f"plateau velocity ({velocity:.1f} Elo/hr)"
                )

        # Clamp final threshold to reasonable range
        threshold = max(0.50, min(0.75, base_threshold + modifier))

        logger.debug(
            f"[FeedbackLoopController] Adaptive threshold: base={base_threshold:.2f}, "
            f"modifier={modifier:+.2f}, final={threshold:.2f} (elo={elo:.0f})"
        )

        return threshold

    def _update_training_final_elo(self, config_key: str, elo: float) -> None:
        """Update the final_elo in training_history after evaluation.

        Jan 6, 2026: P1 improvement - closes the feedback loop between
        training and evaluation by recording the Elo achieved by each
        training run.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            elo: The Elo rating from evaluation
        """
        if not config_key:
            return

        try:
            parsed = parse_config_key(config_key)
            if not parsed:
                return

            # Lazy import to avoid circular dependency
            from app.coordination.training_coordinator import get_training_coordinator

            coordinator = get_training_coordinator()
            coordinator.update_training_final_elo(
                board_type=parsed.board_type,
                num_players=parsed.num_players,
                final_elo=elo,
            )
        except ImportError:
            logger.debug("[FeedbackLoopController] TrainingCoordinator not available")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(
                f"[FeedbackLoopController] Failed to update training final_elo: {e}"
            )

    def _consider_promotion(
        self,
        config_key: str,
        model_path: str,
        win_rate: float,
        elo: float,
    ) -> None:
        """Consider model for promotion."""
        logger.info(
            f"[FeedbackLoopController] Considering promotion for {config_key}: "
            f"win_rate={win_rate:.2%}, elo={elo:.0f}"
        )

        try:
            from app.coordination.pipeline_actions import trigger_promotion

            _safe_create_task(
                trigger_promotion({
                    "config": config_key,
                    "model_path": model_path,
                    "win_rate": win_rate,
                    "elo": elo,
                }),
                f"trigger_promotion({config_key})",
            )
        except ImportError:
            pass

    def _report_engine_bandit_feedback(
        self,
        config_key: str,
        state: FeedbackState,
        current_elo: float,
    ) -> None:
        """Report feedback to the selfplay engine bandit.

        Dec 29 2025: This closes the feedback loop for the multi-armed bandit
        that selects selfplay engines. After evaluation completes, we report
        the Elo improvement (or loss) to help the bandit learn which engine
        produces the best training data for each config.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            state: FeedbackState with engine tracking data
            current_elo: Elo from the just-completed evaluation
        """
        # Skip if no selfplay data to attribute
        if state.last_selfplay_games <= 0:
            return

        # Calculate Elo gain since training started
        elo_gain = current_elo - state.elo_before_training

        try:
            from app.coordination.selfplay_engine_bandit import get_selfplay_engine_bandit

            bandit = get_selfplay_engine_bandit()
            bandit.record_feedback(
                config_key=config_key,
                engine=state.last_selfplay_engine,
                elo_gain=elo_gain,
                games=state.last_selfplay_games,
            )

            logger.info(
                f"[EngineBandit] Recorded feedback for {config_key}/{state.last_selfplay_engine}: "
                f"elo_gain={elo_gain:+.1f}, games={state.last_selfplay_games}"
            )

            # Reset games counter after reporting (start fresh for next training cycle)
            state.last_selfplay_games = 0

        except ImportError:
            logger.debug("[EngineBandit] Bandit module not available")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[EngineBandit] Failed to report feedback: {e}")

    def _signal_urgent_training(self, config_key: str, failure_count: int) -> None:
        """Signal that urgent training is needed after failures."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()

            if failure_count >= 3:
                urgency = "critical"
                priority = 1  # Highest priority
            elif failure_count >= 2:
                urgency = "high"
                priority = 3
            else:
                urgency = "normal"
                priority = 5

            accelerator.signal_training_needed(
                config_key=config_key,
                urgency=urgency,
                reason=f"consecutive_promotion_failures:{failure_count}",
            )

            # Emit SELFPLAY_TARGET_UPDATED to scale selfplay (Phase 21.2 - Dec 2025)
            # More selfplay games needed when promotion fails to generate better training data
            if HAS_SELFPLAY_EVENTS and emit_selfplay_target_updated:
                # Calculate target games: base * (1 + 0.5 * failure_count)
                base_games = 500
                target_games = int(base_games * (1 + 0.5 * failure_count))

                try:
                    # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                    loop = asyncio.get_running_loop()
                    task = _safe_create_task(
                        emit_selfplay_target_updated(
                            config_key=config_key,
                            target_games=target_games,
                            reason=f"promotion_failure_x{failure_count}",
                            priority=priority,
                            source="feedback_loop_controller.py",
                        ),
                        f"emit_selfplay_target_updated({config_key})",
                    )
                    if task:
                        logger.info(
                            f"[FeedbackLoopController] Emitted SELFPLAY_TARGET_UPDATED: "
                            f"{config_key} -> {target_games} games (priority={priority})"
                        )
                except RuntimeError:
                    pass  # No event loop available

        except ImportError:
            pass

    def _apply_intensity_feedback(self, config_key: str, state: FeedbackState) -> None:
        """Apply current feedback state to FeedbackAccelerator."""
        try:
            from app.training.feedback_accelerator import (
                TrainingIntensity,
                get_feedback_accelerator,
            )

            accelerator = get_feedback_accelerator()

            # Map intensity string to enum
            intensity_map = {
                "paused": TrainingIntensity.PAUSED,
                "reduced": TrainingIntensity.REDUCED,
                "normal": TrainingIntensity.NORMAL,
                "accelerated": TrainingIntensity.ACCELERATED,
                "hot_path": TrainingIntensity.HOT_PATH,
            }

            intensity = intensity_map.get(
                state.current_training_intensity,
                TrainingIntensity.NORMAL,
            )

            # Update accelerator state
            accelerator.set_intensity(config_key, intensity)

        except ImportError:
            pass
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to apply intensity feedback: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_state(self, config_key: str) -> FeedbackState | None:
        """Get current feedback state for a config."""
        return self._states.get(config_key)

    def get_all_states(self) -> dict[str, FeedbackState]:
        """Get all feedback states."""
        return dict(self._states)

    def get_summary(self) -> dict[str, Any]:
        """Get feedback loop summary."""
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "configs_tracked": len(self._states),
            "states": {
                k: {
                    "training_intensity": v.current_training_intensity,
                    "exploration_boost": v.current_exploration_boost,
                    "consecutive_successes": v.consecutive_successes,
                    "consecutive_failures": v.consecutive_failures,
                    "last_selfplay_quality": v.last_selfplay_quality,
                    "last_training_accuracy": v.last_training_accuracy,
                    "last_evaluation_win_rate": v.last_evaluation_win_rate,
                    "work_completed_count": v.work_completed_count,
                    "last_work_completion_time": v.last_work_completion_time,
                }
                for k, v in self._states.items()
            },
        }

    # NOTE: _on_daemon_status_changed and _on_p2p_cluster_unhealthy are now
    # provided by FeedbackClusterHealthMixin (Sprint 17.9)

    def _on_training_rollback_needed(self, event) -> None:
        """Handle TRAINING_ROLLBACK_NEEDED - training needs checkpoint rollback.

        Coordinates rollback to previous checkpoint and boosts exploration
        to escape the failure mode.

        Added: December 2025
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config = extract_config_key(payload)
        reason = payload.get("reason", "")
        epoch = payload.get("epoch", 0)

        logger.error(
            f"[FeedbackLoopController] Training rollback needed: {config} "
            f"(epoch {epoch}, reason: {reason})"
        )

        if config:
            state = self._get_or_create_state(config)
            state.consecutive_failures += 1

            # Emit exploration boost to escape failure mode
            try:
                from app.coordination.event_router import emit_exploration_boost
                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config,
                        boost_factor=2.0,
                        reason=reason,
                        anomaly_count=state.consecutive_failures,
                        source="feedback_loop_controller",
                    ),
                    "exploration_boost_emit"
                )
            except ImportError:
                pass

    def _on_training_rollback_completed(self, event) -> None:
        """Handle TRAINING_ROLLBACK_COMPLETED - checkpoint rollback completed.

        Updates feedback state after a rollback has been completed:
        1. Resets consecutive failure count to allow fresh start
        2. Reduces exploration boost gradually (rollback fixed the issue)
        3. Emits TRAINING_INTENSITY_CHANGED to resume training with adjusted params

        Added: December 28, 2025 - Fixes orphan event (no prior subscriber)
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        model_id = payload.get("model_id", "")
        rollback_from = payload.get("rollback_from", "")
        rollback_to = payload.get("rollback_to", "")

        logger.info(
            f"[FeedbackLoopController] Training rollback completed: {config_key} "
            f"(from {rollback_from} to {rollback_to})"
        )

        if config_key:
            state = self._get_or_create_state(config_key)

            # Reset failure count - rollback gives us a fresh start
            old_failures = state.consecutive_failures
            state.consecutive_failures = max(0, state.consecutive_failures - 1)

            # Reduce exploration boost (rollback should have fixed the issue)
            if state.exploration_boost > 1.0:
                state.exploration_boost = max(1.0, state.exploration_boost * 0.7)

            # Update training intensity to resume
            old_intensity = state.training_intensity
            if old_intensity == "paused":
                state.training_intensity = "reduced"  # Cautious restart
            elif old_intensity == "reduced":
                state.training_intensity = "normal"  # Gradual increase

            logger.info(
                f"[FeedbackLoopController] Post-rollback: {config_key} "
                f"failures={old_failures}→{state.consecutive_failures}, "
                f"intensity={old_intensity}→{state.training_intensity}"
            )

            # Emit training intensity change to inform training triggers (January 2026 - migrated to event_router)
            safe_emit_event(
                "TRAINING_INTENSITY_CHANGED",
                {
                    "config_key": config_key,
                    "old_intensity": old_intensity,
                    "new_intensity": state.training_intensity,
                    "reason": "post_rollback_recovery",
                },
                context="feedback_loop_controller",
            )

    def _on_training_timeout_reached(self, event: Any) -> None:
        """Handle TRAINING_TIMEOUT_REACHED - training job exceeded timeout threshold.

        Jan 3, 2026: Closes critical gap where TRAINING_TIMEOUT_REACHED was emitted
        at training_trigger_daemon.py:3314 but had no subscriber/handler.

        Triggers recovery actions:
        1. Boost exploration to help break potential training plateau
        2. Increase selfplay games multiplier to generate fresh data
        3. Track timeout for metrics/observability

        Expected improvement: Faster recovery from stuck training situations,
        contributing to overall Elo gains via reduced downtime.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            timeout_hours = payload.get("timeout_hours", 0)
            training_job_id = payload.get("job_id", "")

            if not config_key:
                logger.warning("[FeedbackLoopController] TRAINING_TIMEOUT_REACHED missing config_key")
                return

            logger.warning(
                f"[FeedbackLoopController] Training timeout for {config_key} after {timeout_hours}h, "
                f"triggering recovery actions (job_id={training_job_id})"
            )

            state = self._get_or_create_state(config_key)

            # 1. Boost exploration to help break potential training plateau
            old_exploration = getattr(state, 'current_exploration_boost', 1.0)
            state.current_exploration_boost = min(old_exploration * 1.5, 2.5)
            state.exploration_boost_expires_at = time.time() + 3600  # 1 hour boost

            # 2. Increase selfplay games multiplier to generate fresh data
            old_games_mult = getattr(state, 'games_multiplier', 1.0)
            state.games_multiplier = min(old_games_mult * 1.5, 2.5)

            # 3. Track consecutive timeouts for escalation
            if not hasattr(state, 'consecutive_timeouts'):
                state.consecutive_timeouts = 0
            state.consecutive_timeouts += 1
            state.last_timeout_time = time.time()

            logger.info(
                f"[FeedbackLoopController] Timeout recovery applied to {config_key}: "
                f"exploration={old_exploration:.2f}→{state.current_exploration_boost:.2f}, "
                f"games_mult={old_games_mult:.2f}→{state.games_multiplier:.2f}, "
                f"consecutive_timeouts={state.consecutive_timeouts}"
            )

            # 4. Emit exploration boost event to propagate to SelfplayScheduler
            try:
                from app.coordination.event_router import emit_exploration_boost

                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config_key,
                        boost_factor=state.current_exploration_boost,
                        reason="timeout_recovery",
                        source="feedback_loop_controller",
                    ),
                    "exploration_boost_timeout_recovery"
                )
            except ImportError:
                pass

        except (AttributeError, TypeError, KeyError, ValueError) as e:
            logger.error(f"[FeedbackLoopController] Error handling training timeout: {e}")

    def _on_quality_check_failed(self, event) -> None:
        """Handle QUALITY_CHECK_FAILED - data quality check failed.

        Triggers additional selfplay to improve data quality and adjusts
        training parameters to be more conservative.

        Added: December 2025
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.0)
        threshold = payload.get("threshold", 0.6)

        logger.warning(
            f"[FeedbackLoopController] Quality check failed: {config} "
            f"(score={quality_score:.2f}, threshold={threshold:.2f})"
        )

        if config:
            state = self._get_or_create_state(config)
            state.last_selfplay_quality = quality_score

            # Trigger more selfplay to improve quality
            if HAS_SELFPLAY_EVENTS:
                try:
                    _safe_create_task(
                        emit_selfplay_target_updated(
                            config_key=config,
                            target_games=1000,  # Request more games
                            reason="quality_check_failed",
                            priority=8,
                            source="feedback_loop_controller",
                        ),
                        "selfplay_target_emit"
                    )
                except (ImportError, AttributeError) as e:
                    # Dec 29, 2025: Narrowed from bare Exception
                    # Emitter not available or wrong signature
                    logger.debug(f"[FeedbackLoop] Emitter not available: {e}")
                except (TypeError, ValueError) as e:
                    # Invalid parameter types/values
                    logger.error(f"[FeedbackLoop] Invalid selfplay target parameters: {e}")
                except RuntimeError as e:
                    # Event bus errors
                    logger.warning(f"[FeedbackLoop] Failed to emit selfplay target: {e}")

    def _on_quality_feedback_adjusted(self, event) -> None:
        """Handle QUALITY_FEEDBACK_ADJUSTED - quality assessment triggered adjustments.

        Adjusts training intensity and exploration based on data quality feedback.
        When quality improves, accelerate training. When quality degrades, boost exploration.

        Added: December 2025 - Closes critical feedback loop gap
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.5)
        budget_multiplier = payload.get("budget_multiplier", 1.0)
        adjustment_type = payload.get("adjustment_type", "unknown")

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        # Update quality tracking
        state.last_selfplay_quality = quality_score

        # Adjust training intensity based on quality feedback
        if budget_multiplier > 1.0:
            # Quality is good - accelerate training
            state.current_training_intensity = "accelerated"
            logger.info(
                f"[FeedbackLoopController] Quality feedback positive for {config_key}: "
                f"score={quality_score:.2f}, multiplier={budget_multiplier:.2f} → accelerated training"
            )
        elif budget_multiplier < 0.8:
            # Quality is poor - boost exploration, slow training
            state.current_training_intensity = "conservative"
            logger.info(
                f"[FeedbackLoopController] Quality feedback negative for {config_key}: "
                f"score={quality_score:.2f}, multiplier={budget_multiplier:.2f} → conservative training"
            )

            # Trigger exploration boost for poor quality
            if HAS_EXPLORATION_EVENTS:
                try:
                    _safe_create_task(
                        emit_exploration_boost(
                            config_key=config_key,
                            boost_factor=1.5,
                            reason=f"quality_feedback_{adjustment_type}",
                            source="feedback_loop_controller",
                        ),
                        "quality_exploration_boost_emit"
                    )
                except (ImportError, RuntimeError, asyncio.CancelledError) as e:
                    # Dec 2025: Narrowed from bare Exception to specific event emission errors
                    logger.debug(f"Failed to emit exploration boost: {e}")
        else:
            # Quality is normal
            state.current_training_intensity = "normal"

    def _on_high_quality_data_available(self, event) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE - quality recovered above threshold.

        When data quality recovers to "good" levels, this handler:
        1. Reduces exploration boost (no longer needed)
        2. Accelerates training intensity
        3. Updates quality tracking metrics

        Added: December 2025 - Closes quality recovery feedback loop
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.0)
        sample_count = payload.get("sample_count", 0)

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        # Update quality tracking
        prev_quality = state.last_selfplay_quality
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()

        # Quality recovered - accelerate training
        state.current_training_intensity = "accelerated"

        logger.info(
            f"[FeedbackLoopController] High quality data available for {config_key}: "
            f"score={quality_score:.2f} (prev={prev_quality:.2f}), "
            f"samples={sample_count} → accelerated training"
        )

        # Track metrics
        if hasattr(self, "_metrics") and self._metrics:
            self._metrics.increment("high_quality_events", {"config_key": config_key})

    def _on_quality_score_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED - game quality recalculated.

        Dec 27, 2025: Closes quality monitoring → training feedback loop.
        When quality scores are updated, this handler:
        1. Updates per-config quality tracking
        2. Adjusts training intensity based on quality trends
        3. Triggers exploration boost if quality is declining

        Args:
            event: Event with payload containing config_key, quality_score, trend
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.0)
        trend = payload.get("trend", "stable")  # improving, declining, stable
        sample_count = payload.get("sample_count", 0)

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        # Update quality tracking
        prev_quality = state.last_selfplay_quality
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()

        # Adjust training intensity based on trend
        if trend == "declining" and quality_score < 0.6:
            state.current_training_intensity = "conservative"
            logger.warning(
                f"[FeedbackLoopController] Quality declining for {config_key}: "
                f"score={quality_score:.2f}, trend={trend} → conservative training"
            )
        elif trend == "improving" and quality_score > 0.8:
            state.current_training_intensity = "accelerated"
            logger.info(
                f"[FeedbackLoopController] Quality improving for {config_key}: "
                f"score={quality_score:.2f} → accelerated training"
            )
        else:
            logger.debug(
                f"[FeedbackLoopController] Quality update for {config_key}: "
                f"score={quality_score:.2f} (prev={prev_quality:.2f})"
            )

        # Dec 29, 2025: Emit EXPLORATION_ADJUSTED for quality-driven selfplay adjustment
        # This closes the feedback loop: quality → exploration → data quality → training
        self._emit_exploration_adjustment(config_key, quality_score, trend)

    def _emit_exploration_adjustment(
        self, config_key: str, quality_score: float, trend: str
    ) -> None:
        """Emit exploration adjustment signal based on quality score.

        Dec 29, 2025: Quality-driven selfplay exploration signals.
        Adjusts exploration parameters to match data quality needs:
        - Low quality → harder positions, more MCTS budget
        - High quality → standard exploration, normal budget
        - Declining trend → boost exploration temperature

        Args:
            config_key: Configuration key
            quality_score: Current quality score (0-1)
            trend: Quality trend (improving, declining, stable)
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            # Determine exploration adjustments based on quality
            if quality_score < 0.5:
                # Very low quality → aggressive exploration needed
                position_difficulty = "hard"
                mcts_budget_multiplier = 1.5  # 50% more MCTS budget
                exploration_temp_boost = 1.3  # Higher temperature
            elif quality_score < 0.7:
                # Medium quality → slightly harder positions
                position_difficulty = "medium-hard"
                mcts_budget_multiplier = 1.2  # 20% more MCTS budget
                exploration_temp_boost = 1.15
            elif quality_score > 0.9:
                # High quality → can reduce budget for efficiency
                position_difficulty = "normal"
                mcts_budget_multiplier = 0.8  # 20% less budget (already have good data)
                exploration_temp_boost = 1.0
            else:
                # Normal quality
                position_difficulty = "normal"
                mcts_budget_multiplier = 1.0
                exploration_temp_boost = 1.0

            # Boost exploration if trend is declining (need to find better positions)
            if trend == "declining":
                exploration_temp_boost *= 1.2
                mcts_budget_multiplier = max(mcts_budget_multiplier, 1.3)

            # Only emit if adjustments differ from baseline
            if (mcts_budget_multiplier != 1.0 or exploration_temp_boost != 1.0 or
                position_difficulty != "normal"):

                payload = {
                    "config_key": config_key,
                    "quality_score": quality_score,
                    "trend": trend,
                    "position_difficulty": position_difficulty,
                    "mcts_budget_multiplier": mcts_budget_multiplier,
                    "exploration_temp_boost": exploration_temp_boost,
                    "timestamp": time.time(),
                }

                bus = get_event_bus()
                from app.distributed.data_events import DataEvent
                event = DataEvent(
                    event_type=DataEventType.EXPLORATION_ADJUSTED,
                    payload=payload,
                    source="FeedbackLoopController",
                )
                bus.publish(event)

                logger.info(
                    f"[FeedbackLoopController] Exploration adjusted for {config_key}: "
                    f"difficulty={position_difficulty}, mcts_mult={mcts_budget_multiplier:.1f}, "
                    f"temp_boost={exploration_temp_boost:.2f} (quality={quality_score:.2f})"
                )

        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Failed to emit exploration adjustment: {e}")

    # NOTE: _on_cluster_capacity_changed, _on_health_check_passed, _on_health_check_failed
    # are now provided by FeedbackClusterHealthMixin (Sprint 17.9)

    def signal_selfplay_quality(self, config_key: str, quality_score: float) -> None:
        """Manually signal selfplay quality (for testing/manual intervention)."""
        state = self._get_or_create_state(config_key)
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()
        self._update_training_intensity(config_key, quality_score)

    def signal_training_complete(
        self,
        config_key: str,
        policy_accuracy: float,
        value_accuracy: float = 0.0,
    ) -> None:
        """Manually signal training completion (for testing/manual intervention)."""
        state = self._get_or_create_state(config_key)
        state.last_training_accuracy = policy_accuracy
        state.last_training_time = time.time()

        if policy_accuracy >= self.policy_accuracy_threshold:
            logger.info(f"Manual training signal: {config_key} ready for evaluation")

    def health_check(self) -> HealthCheckResult:
        """Check controller health.

        December 30, 2025: Now merges HandlerBase stats with feedback-specific metrics.

        Returns:
            Health check result with feedback loop status and metrics.
        """
        # Calculate active states
        active_configs = sum(
            1 for state in self._states.values()
            if time.time() - state.last_training_time < 3600  # Active in last hour
        )

        # Determine health status
        healthy = self.is_running and self._subscribed

        message = "Running" if healthy else (
            "Controller stopped" if not self.is_running else
            "Not subscribed to events"
        )

        # Build details dict with both HandlerBase stats and feedback-specific metrics
        details = {
            "running": self._running,
            "subscribed": self._subscribed,
            "configs_tracked": len(self._states),
            "active_configs": active_configs,
            "cluster_healthy": self._cluster_healthy,
            "policy_accuracy_threshold": self.policy_accuracy_threshold,
            "promotion_threshold": self.promotion_threshold,
            # December 30, 2025: Add HandlerBase stats
            "cycles_completed": self._stats.cycles_completed,
            "events_processed": self._stats.events_processed,
            "errors_count": self._stats.errors_count,
            "uptime_seconds": time.time() - self._stats.started_at if self._stats.started_at else 0,
        }

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details=details,
        )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive controller status for monitoring dashboards (Dec 2025).

        Returns:
            Dictionary with controller status, metrics, and config state.
            Used by DaemonManager for monitoring and dashboards.
        """
        # Calculate metrics
        now = time.time()
        active_configs = []
        pending_configs = []

        for config_key, state in self._states.items():
            hours_since_training = (now - state.last_training_time) / 3600 if state.last_training_time > 0 else float("inf")

            config_info = {
                "config": config_key,
                "current_elo": state.last_elo,
                "last_training": state.last_training_time,
                "hours_since_training": round(hours_since_training, 2),
                "games_since_training": state.games_since_training,
                "training_triggered_count": state.training_triggered_count,
            }

            if hours_since_training < 1.0:  # Active in last hour
                active_configs.append(config_info)
            else:
                pending_configs.append(config_info)

        return {
            "name": "FeedbackLoopController",
            "running": self._running,
            "subscribed": self._subscribed,
            "healthy": self._running and self._subscribed,
            "cluster_healthy": self._cluster_healthy,
            "thresholds": {
                "policy_accuracy": self.policy_accuracy_threshold,
                "promotion": self.promotion_threshold,
            },
            "configs": {
                "total": len(self._states),
                "active": len(active_configs),
                "pending": len(pending_configs),
            },
            "active_configs": active_configs[:10],  # Limit for dashboard
            "metrics": {
                "total_trainings_triggered": sum(s.training_triggered_count for s in self._states.values()),
                "total_games_tracked": sum(s.games_since_training for s in self._states.values()),
            },
        }


# =============================================================================
# Singleton (December 30, 2025: Delegated to HandlerBase)
# =============================================================================

# Note: Module-level globals kept for backward compatibility, but primary
# singleton management now goes through HandlerBase.get_instance()
_controller: FeedbackLoopController | None = None
_controller_lock = threading.Lock()


def get_feedback_loop_controller() -> FeedbackLoopController:
    """Get the singleton FeedbackLoopController instance.

    December 30, 2025: Now delegates to HandlerBase.get_instance() while
    maintaining backward compatibility with module-level _controller.
    """
    global _controller

    # Use HandlerBase singleton management
    controller = FeedbackLoopController.get_instance()

    # Keep module-level reference in sync
    _controller = controller

    return controller


def reset_feedback_loop_controller() -> None:
    """Reset the singleton (for testing).

    December 30, 2025: Now delegates to HandlerBase.reset_instance().
    """
    global _controller

    # Stop running instance if any
    if _controller is not None:
        _safe_create_task(_controller.stop(), "feedback_loop_controller_stop")

    # Reset both module-level and HandlerBase singletons
    _controller = None
    FeedbackLoopController.reset_instance()


__all__ = [
    "FeedbackLoopController",
    "FeedbackState",
    "AdaptiveTrainingSignal",  # Dec 29, 2025: Phase 6 adaptive training
    "get_feedback_loop_controller",
    "reset_feedback_loop_controller",
]
