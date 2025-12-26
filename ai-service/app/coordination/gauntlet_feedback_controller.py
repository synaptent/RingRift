"""Gauntlet Feedback Controller - Closes the evaluation→training feedback loop.

This controller subscribes to EVALUATION_COMPLETED events from the gauntlet
and adjusts training parameters based on model performance. This creates a
tightly-coupled adaptive loop where stronger models automatically produce
better training conditions.

Feedback Actions:
- Strong model (>80% vs heuristic): Reduce exploration, raise quality threshold
- Weak model (<70% vs random): Trigger extra selfplay, extend epochs
- ELO plateau detected: Advance curriculum stage
- Regression detected: Trigger rollback consideration

Usage:
    from app.coordination.gauntlet_feedback_controller import (
        GauntletFeedbackController,
        get_gauntlet_feedback_controller,
        start_gauntlet_feedback_loop,
    )

    # Option 1: Get singleton and start
    controller = get_gauntlet_feedback_controller()
    await controller.start()

    # Option 2: One-liner
    await start_gauntlet_feedback_loop()

Created: December 2025
Purpose: Strengthen AI training self-improvement loop (Phase 14)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    HealthCheckResult,
    register_coordinator,
    unregister_coordinator,
)
from app.distributed.data_events import DataEventType

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GauntletFeedbackConfig:
    """Configuration for gauntlet feedback controller."""

    # Performance thresholds
    strong_vs_heuristic_threshold: float = 0.80  # Above this = reduce exploration
    weak_vs_random_threshold: float = 0.70  # Below this = need more training
    plateau_variance_threshold: float = 0.01  # ELO variance below this = plateau
    plateau_window_size: int = 5  # Number of evaluations to check for plateau

    # Training adjustments
    exploration_reduction_factor: float = 0.8  # Multiply temperature by this
    quality_threshold_boost: float = 0.05  # Add to quality threshold
    extra_selfplay_games: int = 5000  # Games to trigger for weak models
    epoch_extension_factor: float = 1.5  # Multiply epochs by this

    # Cooldowns (seconds)
    adjustment_cooldown: float = 300.0  # Between adjustments to same config
    curriculum_cooldown: float = 600.0  # Between curriculum advancements

    # Rollback thresholds
    regression_elo_drop: float = 50.0  # ELO drop to consider regression
    severe_regression_elo_drop: float = 100.0  # December 2025: Immediate CRITICAL for severe drops
    consecutive_regressions_for_rollback: int = 3


class FeedbackAction(str, Enum):
    """Types of feedback actions taken."""

    REDUCE_EXPLORATION = "reduce_exploration"
    RAISE_QUALITY_THRESHOLD = "raise_quality_threshold"
    TRIGGER_EXTRA_SELFPLAY = "trigger_extra_selfplay"
    EXTEND_TRAINING_EPOCHS = "extend_training_epochs"
    ADVANCE_CURRICULUM = "advance_curriculum"
    CONSIDER_ROLLBACK = "consider_rollback"
    NO_ACTION = "no_action"


@dataclass
class EvaluationRecord:
    """Record of a single evaluation result."""

    timestamp: float
    config_key: str
    board_type: str
    num_players: int
    elo: float
    win_rate_vs_random: float
    win_rate_vs_heuristic: float
    games_played: int


@dataclass
class ConfigTracker:
    """Tracks evaluation history for a single config."""

    elo_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_adjustment_time: float = 0.0
    last_curriculum_advance_time: float = 0.0
    consecutive_regressions: int = 0
    last_elo: float = 1000.0


# =============================================================================
# Main Controller
# =============================================================================


class GauntletFeedbackController:
    """Adjusts training parameters based on gauntlet evaluation results.

    This controller creates a feedback loop from gauntlet → training:
    - Subscribes to EVALUATION_COMPLETED events
    - Analyzes win rates and ELO trends
    - Emits HYPERPARAMETER_UPDATED events to adjust training
    - Emits CURRICULUM_ADVANCED events to adjust selfplay
    """

    def __init__(self, config: GauntletFeedbackConfig | None = None):
        """Initialize the gauntlet feedback controller.

        Args:
            config: Configuration for feedback thresholds and actions
        """
        self.config = config or GauntletFeedbackConfig()
        self._config_trackers: dict[str, ConfigTracker] = {}
        self._subscribed = False
        self._started = False
        self._status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._actions_taken: list[dict[str, Any]] = []
        self._errors_count: int = 0
        self._last_error: str = ""
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Controller name for identification."""
        return "GauntletFeedbackController"

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Whether the controller is actively processing events."""
        return self._started and self._subscribed

    @property
    def uptime_seconds(self) -> float:
        """Time since controller started, in seconds."""
        if self._start_time <= 0:
            return 0.0
        return time.time() - self._start_time

    async def start(self) -> bool:
        """Start the feedback controller.

        Returns:
            True if successfully started
        """
        if self._started:
            return True

        success = await self._subscribe_to_events()
        if success:
            self._started = True
            self._status = CoordinatorStatus.RUNNING
            self._start_time = time.time()
            register_coordinator(self)
            logger.info(f"[{self.name}] Started - listening for EVALUATION_COMPLETED events")
        else:
            self._status = CoordinatorStatus.ERROR
            self._last_error = "Failed to subscribe to events"
        return success

    async def stop(self) -> None:
        """Stop the feedback controller."""
        self._status = CoordinatorStatus.STOPPING
        await self._unsubscribe_from_events()
        unregister_coordinator(self.name)
        self._started = False
        self._status = CoordinatorStatus.STOPPED
        logger.info(f"[{self.name}] Stopped")

    def get_metrics(self) -> dict[str, Any]:
        """Get controller metrics.

        Returns:
            Dictionary of metrics (CoordinatorProtocol compliant)
        """
        return {
            "name": self.name,
            "status": self._status.value,
            "uptime_seconds": self.uptime_seconds,
            "start_time": self._start_time,
            "events_processed": sum(t.elo_history.maxlen for t in self._config_trackers.values() if t.elo_history),
            "errors_count": self._errors_count,
            "last_error": self._last_error,
            # Custom metrics
            "is_running": self.is_running,
            "configs_tracked": len(self._config_trackers),
            "actions_taken": len(self._actions_taken),
            "recent_actions": self._actions_taken[-10:] if self._actions_taken else [],
        }

    def health_check(self) -> HealthCheckResult:
        """Check controller health.

        Returns:
            HealthCheckResult (CoordinatorProtocol compliant)
        """
        if self._status == CoordinatorStatus.ERROR:
            return HealthCheckResult.unhealthy(
                f"Controller in error state: {self._last_error}",
                subscribed=self._subscribed,
                configs_tracked=len(self._config_trackers),
            )

        if self._status == CoordinatorStatus.STOPPED:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Controller is stopped",
            )

        if not self._subscribed:
            return HealthCheckResult.degraded(
                "Controller running but not subscribed to events",
                configs_tracked=len(self._config_trackers),
            )

        return HealthCheckResult(
            healthy=True,
            status=self._status,
            details={
                "subscribed": self._subscribed,
                "configs_tracked": len(self._config_trackers),
                "uptime_seconds": self.uptime_seconds,
                "actions_taken": len(self._actions_taken),
            },
        )

    # =========================================================================
    # Event Subscription
    # =========================================================================

    async def _subscribe_to_events(self) -> bool:
        """Subscribe to EVALUATION_COMPLETED events.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            if bus is None:
                logger.warning(f"[{self.name}] Event bus not available")
                return False

            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
            self._subscribed = True
            logger.info(f"[{self.name}] Subscribed to EVALUATION_COMPLETED events")
            return True

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to subscribe: {e}")
            return False

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from evaluation events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            if bus:
                bus.unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
            self._subscribed = False

        except Exception:
            pass

    # =========================================================================
    # Event Handling
    # =========================================================================

    def _on_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event synchronously.

        Schedules async processing via fire_and_forget.
        """
        try:
            from app.core.async_context import fire_and_forget

            fire_and_forget(self._handle_evaluation_async(event))
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to schedule evaluation handling: {e}")

    async def _handle_evaluation_async(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event asynchronously."""
        async with self._lock:
            try:
                await self._process_evaluation(event)
            except Exception as e:
                self._errors_count += 1
                self._last_error = str(e)
                logger.error(f"[{self.name}] Error processing evaluation: {e}")

    async def _process_evaluation(self, event: Any) -> None:
        """Process an evaluation event and take appropriate action.

        Args:
            event: The EVALUATION_COMPLETED event
        """
        payload = event.payload if hasattr(event, "payload") else {}

        # Extract evaluation results
        config_key = payload.get("config", "")
        board_type = payload.get("board_type", "")
        num_players = payload.get("num_players", 2)
        elo = payload.get("elo", 1000.0)
        games_played = payload.get("games_played", 0)

        # Extract win rates (may be nested in win_rates dict)
        win_rates = payload.get("win_rates", {})
        if isinstance(win_rates, dict):
            win_rate_vs_random = win_rates.get("random", payload.get("win_rate_vs_random", 0.5))
            win_rate_vs_heuristic = win_rates.get(
                "heuristic", payload.get("win_rate_vs_heuristic", 0.5)
            )
        else:
            win_rate_vs_random = payload.get("win_rate_vs_random", 0.5)
            win_rate_vs_heuristic = payload.get("win_rate_vs_heuristic", 0.5)

        # Need config_key and meaningful sample
        if not config_key:
            config_key = f"{board_type}_{num_players}p"
        if games_played < 5:
            logger.debug(f"[{self.name}] Skipping evaluation with only {games_played} games")
            return

        # Get or create tracker
        tracker = self._config_trackers.setdefault(config_key, ConfigTracker())

        # Record evaluation
        record = EvaluationRecord(
            timestamp=time.time(),
            config_key=config_key,
            board_type=board_type,
            num_players=num_players,
            elo=elo,
            win_rate_vs_random=win_rate_vs_random,
            win_rate_vs_heuristic=win_rate_vs_heuristic,
            games_played=games_played,
        )

        logger.info(
            f"[{self.name}] Evaluation for {config_key}: "
            f"ELO={elo:.0f}, vs_random={win_rate_vs_random:.1%}, "
            f"vs_heuristic={win_rate_vs_heuristic:.1%}, games={games_played}"
        )

        # Analyze and take action
        actions = await self._analyze_and_act(record, tracker)

        # Log actions taken
        for action in actions:
            self._actions_taken.append(
                {
                    "timestamp": time.time(),
                    "config": config_key,
                    "action": action.value,
                    "elo": elo,
                    "win_rate_vs_random": win_rate_vs_random,
                    "win_rate_vs_heuristic": win_rate_vs_heuristic,
                }
            )

        # Keep recent actions only
        if len(self._actions_taken) > 100:
            self._actions_taken = self._actions_taken[-100:]

    async def _analyze_and_act(
        self,
        record: EvaluationRecord,
        tracker: ConfigTracker,
    ) -> list[FeedbackAction]:
        """Analyze evaluation results and take appropriate actions.

        Args:
            record: The evaluation record
            tracker: The config tracker

        Returns:
            List of actions taken
        """
        actions: list[FeedbackAction] = []
        now = time.time()
        cfg = self.config

        # Update ELO history
        tracker.elo_history.append(record.elo)

        # Check for regression
        elo_drop = tracker.last_elo - record.elo
        if elo_drop > cfg.regression_elo_drop:
            tracker.consecutive_regressions += 1
            logger.warning(
                f"[{self.name}] Regression detected for {record.config_key}: "
                f"ELO {tracker.last_elo:.0f} -> {record.elo:.0f} (drop: {elo_drop:.0f}) "
                f"(consecutive: {tracker.consecutive_regressions})"
            )

            # December 2025: Always emit REGRESSION_DETECTED for feedback loop
            await self._emit_regression_detected(record, elo_drop)

            # December 2025: Emit REGRESSION_CRITICAL immediately for severe regressions
            # This enables automatic rollback without waiting for consecutive failures
            if elo_drop > cfg.severe_regression_elo_drop:
                logger.warning(
                    f"[{self.name}] SEVERE regression for {record.config_key}: "
                    f"ELO drop {elo_drop:.0f} > threshold {cfg.severe_regression_elo_drop:.0f}"
                )
                actions.append(FeedbackAction.CONSIDER_ROLLBACK)
                await self._emit_rollback_consideration(record, severity="severe", elo_drop=elo_drop)
            # Also emit CRITICAL after consecutive regressions
            elif tracker.consecutive_regressions >= cfg.consecutive_regressions_for_rollback:
                actions.append(FeedbackAction.CONSIDER_ROLLBACK)
                await self._emit_rollback_consideration(record, severity="consecutive", elo_drop=elo_drop)
        else:
            tracker.consecutive_regressions = 0

        tracker.last_elo = record.elo

        # Check cooldown for adjustments
        time_since_last = now - tracker.last_adjustment_time
        if time_since_last < cfg.adjustment_cooldown:
            logger.debug(
                f"[{self.name}] Skipping adjustment for {record.config_key} "
                f"(cooldown: {cfg.adjustment_cooldown - time_since_last:.0f}s remaining)"
            )
            return actions if actions else [FeedbackAction.NO_ACTION]

        # === Strong Model: Reduce exploration, raise quality threshold ===
        if record.win_rate_vs_heuristic > cfg.strong_vs_heuristic_threshold:
            logger.info(
                f"[{self.name}] Strong performance for {record.config_key} "
                f"({record.win_rate_vs_heuristic:.1%} vs heuristic)"
            )
            await self._reduce_exploration(record)
            await self._raise_quality_threshold(record)
            actions.extend(
                [FeedbackAction.REDUCE_EXPLORATION, FeedbackAction.RAISE_QUALITY_THRESHOLD]
            )
            tracker.last_adjustment_time = now

        # === Weak Model: Trigger extra selfplay, extend epochs ===
        elif record.win_rate_vs_random < cfg.weak_vs_random_threshold:
            logger.info(
                f"[{self.name}] Weak performance for {record.config_key} "
                f"({record.win_rate_vs_random:.1%} vs random)"
            )
            await self._trigger_extra_selfplay(record)
            await self._extend_training_epochs(record)
            actions.extend(
                [FeedbackAction.TRIGGER_EXTRA_SELFPLAY, FeedbackAction.EXTEND_TRAINING_EPOCHS]
            )
            tracker.last_adjustment_time = now

        # === ELO Plateau: Advance curriculum ===
        if await self._detect_elo_plateau(tracker):
            time_since_curriculum = now - tracker.last_curriculum_advance_time
            if time_since_curriculum >= cfg.curriculum_cooldown:
                logger.info(f"[{self.name}] ELO plateau detected for {record.config_key}")
                await self._advance_curriculum_stage(record)
                actions.append(FeedbackAction.ADVANCE_CURRICULUM)
                tracker.last_curriculum_advance_time = now

                # December 2025: Emit PLATEAU_DETECTED event for feedback loop
                # This enables AdaptiveController and other components to respond
                await self._emit_plateau_detected(
                    config_key=record.config_key,
                    current_elo=record.elo,
                    window_size=cfg.plateau_window_size,
                )

        # === Phase 5: Emit consolidated ADAPTIVE_PARAMS_CHANGED event ===
        if actions and actions != [FeedbackAction.NO_ACTION]:
            await self._emit_adaptive_params_changed(
                config_key=record.config_key,
                board_type=record.board_type,
                num_players=record.num_players,
                actions_taken=actions,
                reason=f"gauntlet_eval_elo_{record.elo:.0f}",
            )

        return actions if actions else [FeedbackAction.NO_ACTION]

    # =========================================================================
    # Feedback Actions
    # =========================================================================

    async def _reduce_exploration(self, record: EvaluationRecord) -> None:
        """Reduce exploration temperature for strong models.

        Emits HYPERPARAMETER_UPDATED event to signal training/selfplay.
        """
        await self._emit_hyperparameter_update(
            config_key=record.config_key,
            board_type=record.board_type,
            num_players=record.num_players,
            parameter="temperature_scale",
            old_value=1.0,
            new_value=self.config.exploration_reduction_factor,
            reason="strong_performance",
        )

    async def _raise_quality_threshold(self, record: EvaluationRecord) -> None:
        """Raise quality threshold for strong models.

        Emits HYPERPARAMETER_UPDATED event.
        """
        await self._emit_hyperparameter_update(
            config_key=record.config_key,
            board_type=record.board_type,
            num_players=record.num_players,
            parameter="quality_threshold_boost",
            old_value=0.0,
            new_value=self.config.quality_threshold_boost,
            reason="strong_performance",
        )

    async def _trigger_extra_selfplay(self, record: EvaluationRecord) -> None:
        """Trigger extra selfplay games for weak models.

        Emits SELFPLAY_TARGET_UPDATED event.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.SELFPLAY_TARGET_UPDATED,
                    payload={
                        "config": record.config_key,
                        "board_type": record.board_type,
                        "num_players": record.num_players,
                        "extra_games": self.config.extra_selfplay_games,
                        "reason": "weak_performance",
                        "win_rate_vs_random": record.win_rate_vs_random,
                    },
                    source=self.name,
                )
                logger.info(
                    f"[{self.name}] Triggered {self.config.extra_selfplay_games} extra selfplay games "
                    f"for {record.config_key}"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit selfplay target update: {e}")

    async def _extend_training_epochs(self, record: EvaluationRecord) -> None:
        """Extend training epochs for weak models.

        Emits HYPERPARAMETER_UPDATED event.
        """
        await self._emit_hyperparameter_update(
            config_key=record.config_key,
            board_type=record.board_type,
            num_players=record.num_players,
            parameter="epoch_multiplier",
            old_value=1.0,
            new_value=self.config.epoch_extension_factor,
            reason="weak_performance",
        )

    async def _advance_curriculum_stage(self, record: EvaluationRecord) -> None:
        """Advance curriculum stage when ELO plateaus.

        Emits CURRICULUM_ADVANCED event.
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.CURRICULUM_ADVANCED,
                    payload={
                        "config": record.config_key,
                        "board_type": record.board_type,
                        "num_players": record.num_players,
                        "current_elo": record.elo,
                        "reason": "elo_plateau",
                    },
                    source=self.name,
                )
                logger.info(f"[{self.name}] Advanced curriculum for {record.config_key}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit curriculum advancement: {e}")

    async def _emit_regression_detected(
        self, record: EvaluationRecord, elo_drop: float
    ) -> None:
        """Emit REGRESSION_DETECTED event for any regression (December 2025).

        This enables the feedback loop to respond to regressions even before
        they become severe or consecutive.

        Args:
            record: The evaluation record
            elo_drop: The ELO drop amount
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.REGRESSION_DETECTED,
                    payload={
                        "config": record.config_key,
                        "config_key": record.config_key,  # Alias for compatibility
                        "board_type": record.board_type,
                        "num_players": record.num_players,
                        "current_elo": record.elo,
                        "elo_drop": elo_drop,
                        "consecutive_regressions": self._config_trackers[
                            record.config_key
                        ].consecutive_regressions,
                        "win_rate_vs_random": record.win_rate_vs_random,
                        "win_rate_vs_heuristic": record.win_rate_vs_heuristic,
                        "source": self.name,
                    },
                    source=self.name,
                )
                logger.info(
                    f"[{self.name}] Emitted REGRESSION_DETECTED for {record.config_key} "
                    f"(ELO drop: {elo_drop:.0f})"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit regression detected: {e}")

    async def _emit_rollback_consideration(
        self,
        record: EvaluationRecord,
        severity: str = "consecutive",
        elo_drop: float = 0.0,
    ) -> None:
        """Emit REGRESSION_CRITICAL event to trigger model rollback.

        December 2025: Now emits immediately for severe regressions, not just
        after consecutive failures. This closes the critical feedback loop gap.

        Args:
            record: The evaluation record
            severity: "severe" for immediate critical, "consecutive" for after N failures
            elo_drop: The ELO drop amount
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.REGRESSION_CRITICAL,
                    payload={
                        "config": record.config_key,
                        "config_key": record.config_key,  # Alias for compatibility
                        "board_type": record.board_type,
                        "num_players": record.num_players,
                        "current_elo": record.elo,
                        "elo_drop": elo_drop,
                        "severity": severity,
                        "consecutive_regressions": self._config_trackers[
                            record.config_key
                        ].consecutive_regressions,
                        "recommendation": "rollback",
                        "reason": (
                            f"severe_elo_drop_{elo_drop:.0f}"
                            if severity == "severe"
                            else f"consecutive_regressions_{self._config_trackers[record.config_key].consecutive_regressions}"
                        ),
                        "source": self.name,
                    },
                    source=self.name,
                )
                logger.warning(
                    f"[{self.name}] Emitted REGRESSION_CRITICAL for {record.config_key} "
                    f"(severity: {severity}, ELO drop: {elo_drop:.0f}) - rollback recommended"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit regression critical: {e}")

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _emit_hyperparameter_update(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        parameter: str,
        old_value: float,
        new_value: float,
        reason: str,
    ) -> None:
        """Emit HYPERPARAMETER_UPDATED event.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            board_type: Board type
            num_players: Number of players
            parameter: Parameter name being updated
            old_value: Previous value
            new_value: New value
            reason: Reason for update
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.HYPERPARAMETER_UPDATED,
                    payload={
                        "config": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "parameter": parameter,
                        "old_value": old_value,
                        "new_value": new_value,
                        "reason": reason,
                        "source": self.name,
                    },
                    source=self.name,
                )
                # Record for synchronous consumption by train.py
                record_hyperparameter_update(config_key, parameter, new_value, reason)
                logger.info(
                    f"[{self.name}] Emitted HYPERPARAMETER_UPDATED: "
                    f"{parameter} {old_value} -> {new_value} for {config_key}"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit hyperparameter update: {e}")

    async def _emit_plateau_detected(
        self,
        config_key: str,
        current_elo: float,
        window_size: int,
    ) -> None:
        """Emit PLATEAU_DETECTED event when ELO plateaus.

        December 2025: This event enables components like AdaptiveController,
        FeedbackAccelerator, and CurriculumFeedback to respond to plateaus.
        Previously, plateau detection only logged and advanced curriculum locally.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            current_elo: Current ELO rating
            window_size: Number of evaluations in plateau window
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                await router.publish(
                    event_type=DataEventType.PLATEAU_DETECTED,
                    payload={
                        "config_key": config_key,
                        "config": config_key,  # Alias for compatibility
                        "current_elo": current_elo,
                        "window_size": window_size,
                        "variance_threshold": self.config.plateau_variance_threshold,
                        "source": self.name,
                    },
                    source=self.name,
                )
                logger.info(
                    f"[{self.name}] Emitted PLATEAU_DETECTED: "
                    f"ELO={current_elo:.0f} for {config_key}"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to emit plateau detected: {e}")

    async def _emit_adaptive_params_changed(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
        actions_taken: list[FeedbackAction],
        reason: str = "gauntlet_feedback",
    ) -> None:
        """Emit ADAPTIVE_PARAMS_CHANGED event with bundled selfplay adjustments.

        Phase 5 (December 2025): Emit a consolidated event for selfplay to consume.
        This carries multiple related parameters in a single event for efficiency.

        Args:
            config_key: Config identifier
            board_type: Board type
            num_players: Number of players
            actions_taken: List of feedback actions that were taken
            reason: Reason for the adjustments
        """
        try:
            from app.distributed.data_events import DataEventType, DataEvent
            from app.coordination.event_router import get_router

            # Build payload based on actions taken
            payload = {
                "config_key": config_key,
                "config": config_key,  # Alias for compatibility
                "board_type": board_type,
                "num_players": num_players,
                "reason": reason,
                "actions": [a.value for a in actions_taken if hasattr(a, "value")],
            }

            # Add specific multipliers based on actions
            if FeedbackAction.REDUCE_EXPLORATION in actions_taken:
                # Strong model - reduce exploration
                payload["temperature_multiplier"] = 0.8
                payload["search_budget_multiplier"] = 0.9  # Slightly less search needed
            elif FeedbackAction.TRIGGER_EXTRA_SELFPLAY in actions_taken:
                # Weak model - increase exploration and search
                payload["temperature_multiplier"] = 1.2
                payload["search_budget_multiplier"] = 1.3
                payload["exploration_boost"] = 0.1

            if FeedbackAction.ADVANCE_CURRICULUM in actions_taken:
                # Curriculum advanced - increase opponent strength
                payload["opponent_strength_multiplier"] = 1.15

            if not payload.get("temperature_multiplier"):
                # No adjustments needed
                return

            router = get_router()
            if router:
                event = DataEvent(
                    event_type=DataEventType.ADAPTIVE_PARAMS_CHANGED,
                    payload=payload,
                    source=self.name,
                )
                await router.publish_async(DataEventType.ADAPTIVE_PARAMS_CHANGED.value, event)
                logger.info(
                    f"[{self.name}] Emitted ADAPTIVE_PARAMS_CHANGED for {config_key}: "
                    f"temp={payload.get('temperature_multiplier', 1.0):.2f}, "
                    f"budget={payload.get('search_budget_multiplier', 1.0):.2f}"
                )

        except Exception as e:
            logger.debug(f"[{self.name}] Failed to emit ADAPTIVE_PARAMS_CHANGED: {e}")

    async def _detect_elo_plateau(self, tracker: ConfigTracker) -> bool:
        """Detect if ELO has plateaued.

        Args:
            tracker: The config tracker with ELO history

        Returns:
            True if plateau detected
        """
        history = list(tracker.elo_history)
        window = self.config.plateau_window_size

        if len(history) < window:
            return False

        # Check variance of recent ELOs
        recent = history[-window:]
        mean_elo = sum(recent) / len(recent)
        variance = sum((e - mean_elo) ** 2 for e in recent) / len(recent)

        is_plateau = variance < self.config.plateau_variance_threshold

        if is_plateau:
            logger.debug(
                f"[{self.name}] Plateau detected: variance={variance:.4f} "
                f"< threshold={self.config.plateau_variance_threshold}"
            )

        return is_plateau


# =============================================================================
# Singleton & Helpers
# =============================================================================

_controller_instance: GauntletFeedbackController | None = None
_controller_lock = asyncio.Lock()


async def get_gauntlet_feedback_controller(
    config: GauntletFeedbackConfig | None = None,
) -> GauntletFeedbackController:
    """Get the singleton GauntletFeedbackController instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton controller instance
    """
    global _controller_instance

    async with _controller_lock:
        if _controller_instance is None:
            _controller_instance = GauntletFeedbackController(config=config)
        return _controller_instance


async def start_gauntlet_feedback_loop(
    config: GauntletFeedbackConfig | None = None,
) -> GauntletFeedbackController:
    """Start the gauntlet feedback loop.

    Convenience function to get and start the controller.

    Args:
        config: Optional configuration

    Returns:
        The started controller instance
    """
    controller = await get_gauntlet_feedback_controller(config)
    await controller.start()
    return controller


async def stop_gauntlet_feedback_loop() -> None:
    """Stop the gauntlet feedback loop."""
    global _controller_instance

    if _controller_instance:
        await _controller_instance.stop()


def get_gauntlet_feedback_controller_sync() -> GauntletFeedbackController | None:
    """Get the controller instance synchronously (if already created).

    Returns:
        The controller instance or None if not yet created
    """
    return _controller_instance


# =============================================================================
# Pending Hyperparameter Updates (December 2025)
# =============================================================================
# This mechanism allows train.py to query for pending hyperparameter updates
# that should be applied during training. Updates are stored per-config and
# consumed when retrieved.

_pending_hyperparameter_updates: dict[str, dict[str, Any]] = {}
_pending_updates_lock = asyncio.Lock()


def record_hyperparameter_update(
    config_key: str,
    parameter: str,
    value: Any,
    reason: str,
) -> None:
    """Record a hyperparameter update for a config to be consumed by training.

    This is called internally when emitting HYPERPARAMETER_UPDATED events.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        parameter: Parameter name ("learning_rate", "lr_multiplier", etc.)
        value: New value for the parameter
        reason: Reason for the update
    """
    if config_key not in _pending_hyperparameter_updates:
        _pending_hyperparameter_updates[config_key] = {}

    _pending_hyperparameter_updates[config_key][parameter] = {
        "value": value,
        "reason": reason,
        "timestamp": time.time(),
    }
    logger.debug(
        f"[GauntletFeedback] Recorded pending update for {config_key}: "
        f"{parameter}={value} (reason: {reason})"
    )


def get_pending_hyperparameter_updates(config_key: str) -> dict[str, dict[str, Any]]:
    """Get and consume pending hyperparameter updates for a config.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")

    Returns:
        Dict of parameter -> {value, reason, timestamp} for pending updates.
        Returns empty dict if no pending updates.
        Updates are consumed (removed) when retrieved.
    """
    if config_key not in _pending_hyperparameter_updates:
        return {}

    updates = _pending_hyperparameter_updates.pop(config_key, {})
    if updates:
        logger.debug(
            f"[GauntletFeedback] Consumed {len(updates)} pending updates for {config_key}"
        )
    return updates


def clear_pending_hyperparameter_updates(config_key: str | None = None) -> None:
    """Clear pending hyperparameter updates.

    Args:
        config_key: Configuration key to clear, or None to clear all
    """
    if config_key is None:
        _pending_hyperparameter_updates.clear()
    elif config_key in _pending_hyperparameter_updates:
        del _pending_hyperparameter_updates[config_key]
