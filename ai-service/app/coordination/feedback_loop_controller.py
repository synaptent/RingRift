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
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _handle_task_error(task: asyncio.Task, context: str = "") -> None:
    """Handle errors from fire-and-forget tasks.

    Call this via: task.add_done_callback(lambda t: _handle_task_error(t, "context"))
    """
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(f"[FeedbackLoopController] Task error{' in ' + context if context else ''}: {exc}")
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
try:
    from app.distributed.data_events import emit_selfplay_target_updated
    HAS_SELFPLAY_EVENTS = True
except ImportError:
    emit_selfplay_target_updated = None
    HAS_SELFPLAY_EVENTS = False


@dataclass
class FeedbackState:
    """State tracking for a single config's feedback loop."""

    config_key: str

    # Quality metrics
    last_selfplay_quality: float = 0.0
    last_training_accuracy: float = 0.0
    last_evaluation_win_rate: float = 0.0

    # Status
    last_promotion_success: bool | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Timing
    last_selfplay_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0

    # Feedback signals applied
    current_training_intensity: str = "normal"  # normal, accelerated, hot_path, reduced
    current_exploration_boost: float = 1.0  # 1.0 = normal, >1.0 = more exploration
    current_curriculum_weight: float = 1.0  # curriculum priority multiplier


class FeedbackLoopController:
    """Central controller orchestrating all feedback signals.

    Subscribes to:
    - SELFPLAY_COMPLETE: Assess data quality, adjust training intensity
    - TRAINING_COMPLETE: Trigger evaluation, adjust curriculum
    - EVALUATION_COMPLETED: Consider promotion, record results
    - PROMOTION_COMPLETE: Adjust curriculum and exploration based on outcome

    Emits decisions that are consumed by:
    - FeedbackAccelerator: Training intensity signals
    - CurriculumFeedback: Curriculum weight adjustments
    - TemperatureScheduler: Exploration rate adjustments
    """

    def __init__(self):
        self._states: dict[str, FeedbackState] = {}
        self._running = False
        self._subscribed = False
        self._lock = threading.Lock()

        # Phase 23.1: Track selfplay rate changes for monitoring
        self._rate_history: dict[str, list[dict[str, Any]]] = {}

        # Configuration
        self.policy_accuracy_threshold = 0.75  # Trigger evaluation above this
        self.promotion_threshold = 0.55  # Win rate for promotion consideration
        self.failure_exploration_boost = 1.3  # Boost exploration on failure
        self.success_intensity_reduction = 0.9  # Reduce intensity on success

    def _get_or_create_state(self, config_key: str) -> FeedbackState:
        """Get or create state for a config."""
        with self._lock:
            if config_key not in self._states:
                self._states[config_key] = FeedbackState(config_key=config_key)
            return self._states[config_key]

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

    def _subscribe_to_events(self) -> None:
        """Subscribe to all relevant training events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.subscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)

            # Phase 23.1: Subscribe to selfplay rate change events for monitoring
            if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                bus.subscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)
                logger.info("[FeedbackLoopController] Subscribed to 5 event types (including SELFPLAY_RATE_CHANGED)")
            else:
                logger.info("[FeedbackLoopController] Subscribed to 4 event types")

            self._subscribed = True
        except Exception as e:
            logger.warning(f"[FeedbackLoopController] Failed to subscribe: {e}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.unsubscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.unsubscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)

            # Phase 23.1: Unsubscribe from rate change events
            if hasattr(DataEventType, 'SELFPLAY_RATE_CHANGED'):
                bus.unsubscribe(DataEventType.SELFPLAY_RATE_CHANGED, self._on_selfplay_rate_changed)

            self._subscribed = False
        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Error unsubscribing: {e}")

    def _wire_curriculum_feedback(self) -> None:
        """Wire all curriculum feedback integrations."""
        try:
            from app.training.curriculum_feedback import wire_all_curriculum_feedback

            watchers = wire_all_curriculum_feedback(
                elo_significant_change=30.0,
                plateau_cooldown_seconds=600.0,
                tournament_cooldown_seconds=300.0,
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
        except Exception as e:
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
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus is None:
                return

            def on_scheduler_registered(event) -> None:
                """Wire exploration boost when new scheduler registers."""
                payload = event.payload if hasattr(event, "payload") else {}
                config_key = payload.get("config_key", "")

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
                except Exception as e:
                    logger.debug(f"[FeedbackLoopController] Lazy wiring failed: {e}")

            bus.subscribe("SCHEDULER_REGISTERED", on_scheduler_registered)
            logger.debug("[FeedbackLoopController] Subscribed to SCHEDULER_REGISTERED")

        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Failed to subscribe to lazy scheduler: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion.

        Actions:
        1. Assess data quality
        2. Update training intensity based on quality
        3. Signal training readiness if quality is sufficient
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
            games_count = payload.get("games_count", 0)
            db_path = payload.get("db_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            # Assess data quality
            quality_score = self._assess_selfplay_quality(db_path, games_count)
            state.last_selfplay_quality = quality_score

            logger.info(
                f"[FeedbackLoopController] Selfplay complete for {config_key}: "
                f"{games_count} games, quality={quality_score:.2f}"
            )

            # Update training intensity based on quality
            self._update_training_intensity(config_key, quality_score)

            # Signal training readiness if quality is good
            if quality_score >= 0.6:
                self._signal_training_ready(config_key, quality_score)

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling selfplay complete: {e}")

    def _on_selfplay_rate_changed(self, event: Any) -> None:
        """Handle selfplay rate change events (Phase 23.1).

        Tracks rate changes for monitoring and logs significant adjustments.
        This enables visibility into how Elo momentum affects selfplay rates.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
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

        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Error handling rate change: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion.

        Actions:
        1. Record training metrics
        2. Trigger evaluation if accuracy threshold met
        3. Adjust curriculum based on metrics
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
            policy_accuracy = payload.get("policy_accuracy", 0.0)
            value_accuracy = payload.get("value_accuracy", 0.0)
            model_path = payload.get("model_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_training_time = time.time()
            state.last_training_accuracy = policy_accuracy

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

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling training complete: {e}")

    def _emit_curriculum_training_feedback(
        self,
        config_key: str,
        policy_accuracy: float,
        value_accuracy: float,
    ) -> None:
        """Emit curriculum feedback event based on training metrics.

        December 2025: Closes the training → curriculum feedback loop.
        Low training accuracy indicates the model needs more/better data,
        triggering curriculum weight adjustments.
        """
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()

            # Determine curriculum adjustment based on accuracy
            # Low accuracy (<0.65) = boost training weight
            # High accuracy (>0.80) = reduce training weight
            adjustment = 0.0
            if policy_accuracy < 0.65:
                adjustment = 0.15  # Boost - needs more training
            elif policy_accuracy > 0.80:
                adjustment = -0.10  # Reduce - learning well

            if adjustment != 0.0:
                # Update internal weights
                current_weight = feedback._current_weights.get(config_key, 1.0)
                new_weight = max(
                    feedback.weight_min,
                    min(feedback.weight_max, current_weight + adjustment)
                )
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[FeedbackLoopController] Curriculum adjusted for {config_key}: "
                    f"policy_acc={policy_accuracy:.2%} → weight {current_weight:.2f} → {new_weight:.2f}"
                )

            # Emit CURRICULUM_REBALANCED event for downstream listeners
            bus = get_event_bus()
            event = DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={
                    "config": config_key,
                    "trigger": "training_complete",
                    "policy_accuracy": policy_accuracy,
                    "value_accuracy": value_accuracy,
                    "adjustment": adjustment,
                    "new_weights": dict(feedback._current_weights),
                    "timestamp": time.time(),
                },
                source="feedback_loop_controller",
            )

            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                asyncio.run(bus.publish(event))

            logger.debug(
                f"[FeedbackLoopController] Emitted CURRICULUM_REBALANCED (training_complete)"
            )

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Failed to emit curriculum event: {e}")

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle evaluation completion.

        Actions:
        1. Record evaluation results
        2. Consider promotion if win rate threshold met
        3. Adjust curriculum based on performance
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
            win_rate = payload.get("win_rate", 0.0)
            elo = payload.get("elo", 1500.0)
            model_path = payload.get("model_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_evaluation_time = time.time()
            state.last_evaluation_win_rate = win_rate

            logger.info(
                f"[FeedbackLoopController] Evaluation complete for {config_key}: "
                f"win_rate={win_rate:.2%}, elo={elo:.0f}"
            )

            # Consider promotion if threshold met
            if win_rate >= self.promotion_threshold:
                self._consider_promotion(config_key, model_path, win_rate, elo)

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling evaluation complete: {e}")

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
            new_elo = metadata.get("new_elo") or metadata.get("elo")

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

                state.current_exploration_boost = max(1.0, state.current_exploration_boost - 0.1)

            else:
                state.consecutive_failures += 1
                state.consecutive_successes = 0
                logger.info(
                    f"[FeedbackLoopController] Promotion FAILED for {config_key} "
                    f"(streak: {state.consecutive_failures})"
                )

                # Boost exploration and intensity on failure
                state.current_exploration_boost = min(2.0, state.current_exploration_boost + 0.2)

                if state.consecutive_failures >= 3:
                    state.current_training_intensity = "hot_path"
                else:
                    state.current_training_intensity = "accelerated"

                # Signal urgent training needed
                self._signal_urgent_training(config_key, state.consecutive_failures)

            # Apply feedback to FeedbackAccelerator
            self._apply_intensity_feedback(config_key, state)

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling promotion complete: {e}")

    # =========================================================================
    # Internal Actions
    # =========================================================================

    def _assess_selfplay_quality(self, db_path: str, games_count: int) -> float:
        """Assess quality of selfplay data using UnifiedQualityScorer.

        Uses the proper quality scoring system that evaluates game content,
        not just game count. Falls back to count-based heuristics if the
        unified scorer is unavailable.

        Returns:
            Quality score 0.0-1.0
        """
        try:
            from pathlib import Path
            import sqlite3
            from app.quality.unified_quality import compute_game_quality_from_params

            db = Path(db_path)
            if not db.exists():
                logger.debug(f"Database not found: {db_path}")
                return 0.3

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT game_id, game_status, winner, termination_reason, total_moves
                    FROM games
                    WHERE game_status IN ('complete', 'finished', 'COMPLETE', 'FINISHED')
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                games = cursor.fetchall()

            if not games:
                logger.debug(f"No completed games in {db_path}")
                return 0.3

            quality_scores = []
            for game in games:
                try:
                    quality = compute_game_quality_from_params(
                        game_id=game["game_id"],
                        game_status=game["game_status"],
                        winner=game["winner"],
                        termination_reason=game["termination_reason"],
                        total_moves=game["total_moves"] or 0,
                    )
                    quality_scores.append(quality.quality_score)
                except Exception as e:
                    logger.debug(f"[FeedbackLoopController] Failed to compute quality for game {game.get('game_id', 'unknown')}: {e}")
                    continue

            if not quality_scores:
                return 0.3

            avg_quality = sum(quality_scores) / len(quality_scores)
            count_factor = min(1.0, games_count / 500)
            final_quality = 0.3 + (avg_quality - 0.3) * count_factor

            logger.debug(
                f"[FeedbackLoopController] Quality: avg={avg_quality:.3f}, "
                f"count_factor={count_factor:.2f}, final={final_quality:.3f}"
            )
            return final_quality

        except ImportError:
            logger.debug("UnifiedQualityScorer not available, using count heuristic")
        except Exception as e:
            logger.debug(f"Quality assessment error: {e}")

        # Fallback to count-based heuristic
        if games_count < 100:
            return 0.3
        elif games_count < 500:
            return 0.6
        elif games_count < 1000:
            return 0.8
        else:
            return 0.95

    def _compute_intensity_from_quality(self, quality_score: float) -> str:
        """Map continuous quality score to intensity level.

        December 2025 - Phase 2C.1: Continuous quality-to-intensity scaling.
        Replaces binary 0.6/0.8 thresholds with a 5-tier gradient.

        Returns:
            Intensity level: paused, reduced, normal, accelerated, hot_path
        """
        if quality_score >= 0.90:
            return "hot_path"  # Excellent quality → maximum training speed
        elif quality_score >= 0.80:
            return "accelerated"  # Very good quality
        elif quality_score >= 0.65:
            return "normal"  # Adequate quality
        elif quality_score >= 0.50:
            return "reduced"  # Poor quality → slower training
        else:
            return "paused"  # Very poor quality → pause training

    def _update_training_intensity(self, config_key: str, quality_score: float) -> None:
        """Update training intensity based on data quality.

        December 2025 - Phase 2C.1: Now uses continuous quality gradient
        instead of binary thresholds.
        """
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()

            # Compute intensity using continuous gradient
            intensity = self._compute_intensity_from_quality(quality_score)

            # Map to urgency for accelerator signaling
            urgency_map = {
                "hot_path": "critical",
                "accelerated": "high",
                "normal": "normal",
                "reduced": "low",
                "paused": "none",
            }

            urgency = urgency_map.get(intensity, "normal")

            if urgency != "none":
                accelerator.signal_training_needed(
                    config_key=config_key,
                    urgency=urgency,
                    reason=f"quality_score_{quality_score:.2f}_intensity_{intensity}",
                )

            # Update state with computed intensity
            state = self._get_or_create_state(config_key)
            state.current_training_intensity = intensity

            logger.debug(
                f"[FeedbackLoopController] Quality {quality_score:.2f} → intensity {intensity}"
            )

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to update training intensity: {e}")

    def _signal_training_ready(self, config_key: str, quality_score: float) -> None:
        """Signal that training data is ready."""
        try:
            from app.coordination.event_emitters import emit_data_quality_assessed

            asyncio.create_task(
                emit_data_quality_assessed(
                    config=config_key,
                    quality_score=quality_score,
                    samples_available=0,  # Unknown here
                    ready_for_training=True,
                )
            )
        except (ImportError, RuntimeError):
            pass

    def _trigger_evaluation(self, config_key: str, model_path: str) -> None:
        """Trigger gauntlet evaluation automatically after training.

        December 2025: Wires TRAINING_COMPLETED → auto-gauntlet evaluation.
        This closes the training feedback loop by automatically evaluating
        newly trained models against baselines.

        The gauntlet results determine whether the model should be promoted
        to production or if more training is needed.
        """
        logger.info(f"[FeedbackLoopController] Triggering gauntlet evaluation for {config_key}")

        try:
            from app.coordination.pipeline_actions import trigger_evaluation

            # Parse config_key into board_type and num_players
            # Format: "hex8_2p", "square8_4p", etc.
            parts = config_key.rsplit("_", 1)
            if len(parts) != 2:
                logger.warning(f"[FeedbackLoopController] Invalid config_key format: {config_key}")
                return

            board_type = parts[0]
            try:
                num_players = int(parts[1].replace("p", ""))
            except ValueError:
                logger.warning(f"[FeedbackLoopController] Cannot parse num_players from: {config_key}")
                return

            # Launch gauntlet evaluation asynchronously
            async def run_gauntlet():
                result = await trigger_evaluation(
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=50,  # Standard gauntlet size
                )
                if result.success:
                    logger.info(
                        f"[FeedbackLoopController] Gauntlet passed for {config_key}: "
                        f"eligible={result.metadata.get('promotion_eligible')}"
                    )
                    # Emit evaluation completed event
                    if result.metadata.get("promotion_eligible"):
                        self._consider_promotion(
                            config_key,
                            model_path,
                            result.metadata.get("win_rates", {}).get("heuristic", 0) / 100,
                            result.metadata.get("elo_delta", 0),
                        )
                else:
                    logger.warning(
                        f"[FeedbackLoopController] Gauntlet failed for {config_key}: "
                        f"{result.error or 'unknown error'}"
                    )

            asyncio.create_task(run_gauntlet())

        except ImportError as e:
            logger.debug(f"[FeedbackLoopController] trigger_evaluation not available: {e}")
        except RuntimeError as e:
            logger.debug(f"[FeedbackLoopController] Could not create gauntlet task: {e}")

    def _record_training_in_curriculum(self, config_key: str) -> None:
        """Record training completion in curriculum."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            feedback.record_training(config_key)
        except ImportError:
            pass

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

            asyncio.create_task(
                trigger_promotion({
                    "config": config_key,
                    "model_path": model_path,
                    "win_rate": win_rate,
                    "elo": elo,
                })
            )
        except (ImportError, RuntimeError):
            pass

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
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(emit_selfplay_target_updated(
                            config_key=config_key,
                            target_games=target_games,
                            reason=f"promotion_failure_x{failure_count}",
                            priority=priority,
                            source="feedback_loop_controller.py",
                        ))
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
        except Exception as e:
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
                }
                for k, v in self._states.items()
            },
        }

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


# =============================================================================
# Singleton
# =============================================================================

_controller: FeedbackLoopController | None = None
_controller_lock = threading.Lock()


def get_feedback_loop_controller() -> FeedbackLoopController:
    """Get the singleton FeedbackLoopController instance."""
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = FeedbackLoopController()
    return _controller


def reset_feedback_loop_controller() -> None:
    """Reset the singleton (for testing)."""
    global _controller
    if _controller is not None:
        asyncio.create_task(_controller.stop())
    _controller = None


__all__ = [
    "FeedbackLoopController",
    "FeedbackState",
    "get_feedback_loop_controller",
    "reset_feedback_loop_controller",
]
