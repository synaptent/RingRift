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
from app.coordination.feedback.loss_monitoring_mixin import LossMonitoringMixin
from app.coordination.feedback.evaluation_feedback_mixin import EvaluationFeedbackMixin
from app.coordination.feedback.regression_handling_mixin import RegressionHandlingMixin
from app.coordination.feedback.selfplay_feedback_mixin import SelfplayFeedbackMixin
from app.coordination.feedback.plateau_handling_mixin import PlateauHandlingMixin
from app.coordination.feedback.quality_events_mixin import QualityEventsMixin
from app.coordination.feedback.recovery_handling_mixin import RecoveryHandlingMixin
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
    curriculum_last_reset: float = 0.0  # Jan 26, 2026: Timestamp of last curriculum weight reset (stall recovery)

    # Evaluation state (Jan 26, 2026)
    evaluation_in_progress: bool = False  # True when evaluation is running for this config

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


class FeedbackLoopController(SelfplayFeedbackMixin, RegressionHandlingMixin, EvaluationFeedbackMixin, LossMonitoringMixin, TrainingCurriculumFeedbackMixin, EloVelocityAdaptationMixin, QualityFeedbackMixin, ExplorationBoostMixin, FeedbackClusterHealthMixin, PlateauHandlingMixin, QualityEventsMixin, RecoveryHandlingMixin, HandlerBase):
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

    January 2026 Sprint 17.9: Selfplay handlers extracted to SelfplayFeedbackMixin.
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

    # Sprint 17.9 (Jan 16, 2026): Selfplay event handlers extracted to SelfplayFeedbackMixin
    # Extracted methods (~230 LOC):
    # - _on_selfplay_complete(event) - async, handles SELFPLAY_COMPLETE
    # - _on_selfplay_rate_changed(event) - handles SELFPLAY_RATE_CHANGED
    # - _on_cpu_pipeline_job_completed(event) - async, handles CPU_PIPELINE_JOB_COMPLETED
    # - _on_database_created(event) - handles DATABASE_CREATED
    # See: app/coordination/feedback/selfplay_feedback_mixin.py

    # NOTE: _on_training_loss_anomaly and _on_training_loss_trend are now provided
    # by LossMonitoringMixin (Sprint 17.9 Phase 4 decomposition, ~120 LOC extracted)

    # Sprint 18 (Feb 2026): Plateau handling methods extracted to PlateauHandlingMixin
    # Extracted methods (~330 LOC):
    # - _on_plateau_detected(event) - handles PLATEAU_DETECTED
    # - _advance_curriculum_on_velocity_plateau(config_key, state)
    # - _trigger_hyperparameter_search(config_key, state)
    # See: app/coordination/feedback/plateau_handling_mixin.py

    # Sprint 18 (Feb 2026): Quality event handlers extracted to QualityEventsMixin
    # Extracted methods (~380 LOC):
    # - _on_quality_degraded_for_training(event) - handles QUALITY_DEGRADED
    # - _trigger_quality_check(config_key, reason)
    # - _on_quality_check_failed(event) - handles QUALITY_CHECK_FAILED
    # - _on_quality_feedback_adjusted(event) - handles QUALITY_FEEDBACK_ADJUSTED
    # - _on_high_quality_data_available(event) - handles HIGH_QUALITY_DATA_AVAILABLE
    # - _on_quality_score_updated(event) - handles QUALITY_SCORE_UPDATED
    # - _emit_exploration_adjustment(config_key, quality_score, trend)
    # See: app/coordination/feedback/quality_events_mixin.py

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

            # January 28, 2026: Enhanced observability logging for gauntlet dispatch debugging
            from app.config.env import env
            logger.info(
                f"[FeedbackLoopController] TRAINING_COMPLETED received: config={config_key}, "
                f"model_path={'present' if model_path else 'MISSING'}, "
                f"policy_acc={policy_accuracy:.2%}, value_acc={value_accuracy:.2%}, "
                f"gauntlet_enabled={env.gauntlet_enabled}"
            )

            # Always trigger evaluation for newly trained models (January 21, 2026 fix)
            # The policy accuracy threshold was too restrictive and blocked evaluation
            # when policy_accuracy wasn't reported in the training event (defaulted to 0.0).
            # Gauntlet evaluation provides the definitive quality assessment.
            if model_path:
                logger.info(
                    f"[FeedbackLoopController] Triggering gauntlet evaluation for {config_key} "
                    f"(will dispatch to cluster if gauntlet_enabled=false)"
                )
                self._trigger_evaluation(config_key, model_path)
            elif policy_accuracy >= self.policy_accuracy_threshold:
                # Fallback: trigger based on accuracy if no model path available
                logger.warning(
                    f"[FeedbackLoopController] No model_path in training event for {config_key}, "
                    f"using policy_accuracy threshold fallback"
                )
            else:
                # January 28, 2026: Log when gauntlet is skipped due to missing model_path
                logger.warning(
                    f"[FeedbackLoopController] Cannot trigger gauntlet for {config_key}: "
                    f"model_path is empty and policy_accuracy={policy_accuracy:.2%} below threshold"
                )

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

    # NOTE: _calculate_regression_amplitude and _on_regression_detected are now
    # provided by RegressionHandlingMixin (Sprint 17.9 Phase 4 decomposition)

    # NOTE: _retry_evaluation is now provided by EvaluationFeedbackMixin
    # (Sprint 17.9 Phase 4 decomposition)

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

    # NOTE: _trigger_evaluation, _run_single_harness_gauntlet, _trigger_gauntlet_all_baselines
    # are now provided by EvaluationFeedbackMixin (Sprint 17.9 Phase 4 decomposition, ~210 LOC extracted)

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

    # Sprint 18 (Feb 2026): Recovery handling methods extracted to RecoveryHandlingMixin
    # Extracted methods (~170 LOC):
    # - _on_training_rollback_needed(event) - handles TRAINING_ROLLBACK_NEEDED
    # - _on_training_rollback_completed(event) - handles TRAINING_ROLLBACK_COMPLETED
    # - _on_training_timeout_reached(event) - handles TRAINING_TIMEOUT_REACHED
    # See: app/coordination/feedback/recovery_handling_mixin.py

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
