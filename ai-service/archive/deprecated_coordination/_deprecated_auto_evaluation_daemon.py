"""Auto Evaluation Daemon - Automatic model evaluation (December 2025).

.. deprecated:: December 2025
    This module is deprecated and will be removed in Q2 2026.
    Use the cleaner architecture instead:
    - EvaluationDaemon (app.coordination.evaluation_daemon) for gauntlet evaluation
    - AutoPromotionDaemon (app.coordination.auto_promotion_daemon) for auto-promotion

    Example migration:
        # Old (deprecated):
        from app.coordination.auto_evaluation_daemon import AutoEvaluationDaemon
        daemon = AutoEvaluationDaemon()
        await daemon.start()

        # New (recommended):
        from app.coordination.evaluation_daemon import get_evaluation_daemon
        from app.coordination.auto_promotion_daemon import get_auto_promotion_daemon

        eval_daemon = get_evaluation_daemon()
        promo_daemon = get_auto_promotion_daemon()
        await eval_daemon.start()
        await promo_daemon.start()

    The separated architecture provides:
    - Cleaner separation of concerns
    - Independent configuration of evaluation vs promotion
    - Better integration with DaemonManager

This daemon automatically evaluates newly trained models via gauntlet,
eliminating the manual evaluation step. It monitors for training completion
and triggers gauntlet runs to validate model quality.

Key features:
- Subscribes to TRAINING_COMPLETE events
- Automatically triggers gauntlet evaluation
- Tracks evaluation results per model
- Triggers promotion when quality thresholds met
- Emits EVALUATION_COMPLETE/PROMOTION_TRIGGERED events
- Integrates with GauntletResultsDB for persistence

Decision Logic:
1. Detect training completion event
2. Queue gauntlet evaluation for new model
3. Run gauntlet against baselines (RANDOM, HEURISTIC)
4. Compare win rates to thresholds
5. Trigger promotion if thresholds met

Promotion Thresholds:
- vs RANDOM: 85% win rate required
- vs HEURISTIC: 60% win rate required

Usage:
    from app.coordination.auto_evaluation_daemon import AutoEvaluationDaemon

    daemon = AutoEvaluationDaemon()
    await daemon.start()

December 2025: Created as part of Phase 2 automation improvements.
December 2025: DEPRECATED - Use EvaluationDaemon + AutoPromotionDaemon instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)

# Module-level deprecation warning
warnings.warn(
    "auto_evaluation_daemon is deprecated and will be removed in Q2 2026. "
    "Use EvaluationDaemon + AutoPromotionDaemon instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class AutoEvaluationConfig:
    """Configuration for automatic evaluation."""

    enabled: bool = True
    # Games per opponent in gauntlet
    games_per_opponent: int = 50
    # Promotion thresholds
    random_win_rate_threshold: float = 0.85  # 85% vs RANDOM
    heuristic_win_rate_threshold: float = 0.60  # 60% vs HEURISTIC
    # Maximum concurrent evaluations
    max_concurrent_evaluations: int = 2
    # Cooldown between evaluations for same model
    evaluation_cooldown_seconds: int = 3600  # 1 hour
    # Timeout for gauntlet evaluation
    evaluation_timeout_seconds: int = 7200  # 2 hours
    # Auto-promote passing models
    auto_promote: bool = True
    # Sync promoted models to cluster
    sync_to_cluster: bool = True
    # Early stopping for gauntlet
    early_stopping: bool = True
    early_stopping_confidence: float = 0.95
    # Scan interval for pending evaluations
    scan_interval_seconds: int = 120  # 2 minutes


@dataclass
class ModelEvaluationState:
    """Tracks evaluation state for a model."""

    model_path: str
    config_key: str
    board_type: str
    num_players: int
    # Status
    evaluation_pending: bool = False
    evaluation_in_progress: bool = False
    last_evaluation_time: float = 0.0
    # Results
    random_win_rate: float = 0.0
    heuristic_win_rate: float = 0.0
    passed_random: bool = False
    passed_heuristic: bool = False
    overall_passed: bool = False
    # Tracking
    consecutive_failures: int = 0
    last_error: str = ""


class AutoEvaluationDaemon:
    """Daemon that automatically evaluates models via gauntlet."""

    def __init__(self, config: AutoEvaluationConfig | None = None):
        self.config = config or AutoEvaluationConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self._evaluation_states: dict[str, ModelEvaluationState] = {}
        self._evaluation_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_evaluations
        )
        self._evaluation_queue: list[str] = []  # model paths
        self._event_subscriptions: list[Any] = []

    async def start(self) -> None:
        """Start the auto evaluation daemon."""
        if self._running:
            logger.warning("[AutoEvaluation] Already running")
            return

        self._running = True
        logger.info("[AutoEvaluation] Starting auto evaluation daemon")

        # Subscribe to events
        await self._subscribe_to_events()

        # Start background task
        self._task = safe_create_task(
            self._monitor_loop(),
            name="auto_evaluation_monitor",
        )
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        """Stop the auto evaluation daemon."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from events
        for unsub in self._event_subscriptions:
            try:
                if callable(unsub):
                    unsub()
            except Exception as e:
                logger.debug(f"[AutoEvaluation] Error unsubscribing: {e}")

        logger.info("[AutoEvaluation] Stopped")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Handle task completion or failure."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[AutoEvaluation] Task failed: {exc}")
        except asyncio.CancelledError:
            pass
        except asyncio.InvalidStateError:
            pass

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Subscribe to training completion
        try:
            from app.coordination.event_router import StageEvent, get_stage_event_bus

            bus = get_stage_event_bus()
            unsub = bus.subscribe(
                StageEvent.TRAINING_COMPLETE, self._on_training_complete
            )
            self._event_subscriptions.append(unsub)
            logger.info("[AutoEvaluation] Subscribed to TRAINING_COMPLETE events")
        except ImportError:
            logger.debug("[AutoEvaluation] Stage events not available")

        # Also subscribe to direct data events (Phase 1.3)
        # This ensures we receive TRAINING_COMPLETED from train.py
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            # Subscribe to MODEL_UPDATED
            unsub = bus.subscribe(
                DataEventType.MODEL_UPDATED, self._on_model_updated
            )
            self._event_subscriptions.append(unsub)
            logger.info("[AutoEvaluation] Subscribed to MODEL_UPDATED events")

            # Subscribe to TRAINING_COMPLETED (Phase 1.3: bridge to evaluation)
            unsub = bus.subscribe(
                DataEventType.TRAINING_COMPLETED, self._on_training_completed_data_event
            )
            self._event_subscriptions.append(unsub)
            logger.info("[AutoEvaluation] Subscribed to TRAINING_COMPLETED events")
        except ImportError:
            pass

    async def _on_training_complete(self, result: Any) -> None:
        """Handle training completion event."""
        try:
            metadata = getattr(result, "metadata", {})
            model_path = metadata.get("model_path") or getattr(
                result, "model_path", None
            )
            board_type = metadata.get("board_type") or getattr(
                result, "board_type", None
            )
            num_players = metadata.get("num_players") or getattr(
                result, "num_players", None
            )

            if not model_path:
                logger.debug("[AutoEvaluation] No model path in training result")
                return

            if not board_type or not num_players:
                logger.debug("[AutoEvaluation] Missing config in training result")
                return

            # Queue for evaluation
            await self._queue_evaluation(
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
            )

        except Exception as e:
            logger.error(f"[AutoEvaluation] Error handling training complete: {e}")

    async def _on_model_updated(self, event: Any) -> None:
        """Handle model update event."""
        try:
            payload = getattr(event, "payload", event)
            if isinstance(payload, dict):
                model_path = payload.get("model_path")
                board_type = payload.get("board_type")
                num_players = payload.get("num_players")

                if model_path and board_type and num_players:
                    await self._queue_evaluation(
                        model_path=model_path,
                        board_type=board_type,
                        num_players=num_players,
                    )

        except Exception as e:
            logger.debug(f"[AutoEvaluation] Error handling model updated: {e}")

    async def _on_training_completed_data_event(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED event from data bus (Phase 1.3).

        This bridges the train.py event emission to evaluation queue.
        The event payload format from train.py:
        {
            "epochs_completed": int,
            "best_val_loss": float,
            "config": "board_type_Np",  # e.g., "hex8_2p"
            "checkpoint_path": str,
            "trigger_evaluation": bool,  # Optional: if True, queue for evaluation
        }
        """
        try:
            payload = getattr(event, "payload", event)
            if isinstance(payload, dict):
                # Check if evaluation should be triggered
                trigger_evaluation = payload.get("trigger_evaluation", False)
                if not trigger_evaluation:
                    logger.debug("[AutoEvaluation] Skipping - trigger_evaluation not set")
                    return

                checkpoint_path = payload.get("checkpoint_path")
                config = payload.get("config", "")

                if not checkpoint_path:
                    logger.debug("[AutoEvaluation] No checkpoint_path in TRAINING_COMPLETED")
                    return

                # Parse config (e.g., "hex8_2p" -> board_type="hex8", num_players=2)
                if "_" in config and config.endswith("p"):
                    parts = config.rsplit("_", 1)
                    board_type = parts[0]
                    try:
                        num_players = int(parts[1].rstrip("p"))
                    except ValueError:
                        logger.debug(f"[AutoEvaluation] Invalid config format: {config}")
                        return
                else:
                    logger.debug(f"[AutoEvaluation] Cannot parse config: {config}")
                    return

                logger.info(
                    f"[AutoEvaluation] Received TRAINING_COMPLETED for {config} "
                    f"at {checkpoint_path} - queueing for gauntlet evaluation"
                )

                # Queue for evaluation
                await self._queue_evaluation(
                    model_path=str(checkpoint_path),
                    board_type=board_type,
                    num_players=num_players,
                )

        except Exception as e:
            logger.error(f"[AutoEvaluation] Error handling TRAINING_COMPLETED: {e}")

    async def _queue_evaluation(
        self, model_path: str, board_type: str, num_players: int
    ) -> None:
        """Queue a model for evaluation."""
        config_key = f"{board_type}_{num_players}p"

        # Check if already evaluating or recently evaluated
        if model_path in self._evaluation_states:
            state = self._evaluation_states[model_path]
            if state.evaluation_in_progress:
                logger.debug(
                    f"[AutoEvaluation] {config_key}: evaluation already in progress"
                )
                return

            time_since_last = time.time() - state.last_evaluation_time
            if time_since_last < self.config.evaluation_cooldown_seconds:
                logger.debug(
                    f"[AutoEvaluation] {config_key}: cooldown active, "
                    f"{self.config.evaluation_cooldown_seconds - time_since_last:.0f}s remaining"
                )
                return

        # Create or update state
        self._evaluation_states[model_path] = ModelEvaluationState(
            model_path=model_path,
            config_key=config_key,
            board_type=board_type,
            num_players=num_players,
            evaluation_pending=True,
        )

        # Add to queue if not already present
        if model_path not in self._evaluation_queue:
            self._evaluation_queue.append(model_path)
            logger.info(
                f"[AutoEvaluation] Queued evaluation for {config_key}: {model_path}"
            )

    async def _monitor_loop(self) -> None:
        """Background loop to process evaluation queue."""
        while self._running:
            try:
                await self._process_queue()
                await asyncio.sleep(self.config.scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AutoEvaluation] Monitor loop error: {e}")
                await asyncio.sleep(60)

    async def _process_queue(self) -> None:
        """Process pending evaluations."""
        if not self._evaluation_queue:
            return

        # Process next item in queue
        model_path = self._evaluation_queue[0]
        state = self._evaluation_states.get(model_path)

        if not state or not state.evaluation_pending:
            self._evaluation_queue.pop(0)
            return

        # Start evaluation
        safe_create_task(
            self._run_evaluation(model_path),
            name=f"evaluation_{Path(model_path).stem}",
        )
        self._evaluation_queue.pop(0)

    async def _run_evaluation(self, model_path: str) -> bool:
        """Run gauntlet evaluation for a model."""
        state = self._evaluation_states.get(model_path)
        if not state:
            return False

        async with self._evaluation_semaphore:
            state.evaluation_pending = False
            state.evaluation_in_progress = True

            try:
                logger.info(
                    f"[AutoEvaluation] Starting gauntlet for {state.config_key}: "
                    f"{model_path}"
                )

                # Run gauntlet
                result = await self._run_gauntlet(
                    model_path=model_path,
                    board_type=state.board_type,
                    num_players=state.num_players,
                )

                state.last_evaluation_time = time.time()

                if result is None:
                    state.consecutive_failures += 1
                    state.last_error = "Gauntlet failed"
                    logger.error(
                        f"[AutoEvaluation] Gauntlet failed for {state.config_key}"
                    )
                    return False

                # Extract results
                state.random_win_rate = result.get("random_win_rate", 0.0)
                state.heuristic_win_rate = result.get("heuristic_win_rate", 0.0)
                state.passed_random = (
                    state.random_win_rate >= self.config.random_win_rate_threshold
                )
                state.passed_heuristic = (
                    state.heuristic_win_rate
                    >= self.config.heuristic_win_rate_threshold
                )
                state.overall_passed = state.passed_random and state.passed_heuristic
                state.consecutive_failures = 0

                logger.info(
                    f"[AutoEvaluation] Gauntlet complete for {state.config_key}: "
                    f"RANDOM={state.random_win_rate:.1%} "
                    f"({'PASS' if state.passed_random else 'FAIL'}), "
                    f"HEURISTIC={state.heuristic_win_rate:.1%} "
                    f"({'PASS' if state.passed_heuristic else 'FAIL'})"
                )

                # Emit evaluation complete event
                await self._emit_evaluation_complete(state, result)

                # Auto-promote if passed
                if state.overall_passed and self.config.auto_promote:
                    await self._trigger_promotion(state)

                return state.overall_passed

            except Exception as e:
                state.consecutive_failures += 1
                state.last_error = str(e)
                logger.error(
                    f"[AutoEvaluation] Evaluation error for {state.config_key}: {e}"
                )
                return False

            finally:
                state.evaluation_in_progress = False

    async def _run_gauntlet(
        self, model_path: str, board_type: str, num_players: int
    ) -> dict[str, Any] | None:
        """Run the gauntlet evaluation."""
        try:
            from app.training.game_gauntlet import (
                BaselineOpponent,
                run_baseline_gauntlet,
            )

            # Convert board_type string to enum
            from app.rules.types import BoardType

            board_type_enum = BoardType[board_type.upper()]

            # Run gauntlet in thread to not block event loop
            # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: run_baseline_gauntlet(
                        model_path=model_path,
                        board_type=board_type_enum,
                        num_players=num_players,
                        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
                        games_per_opponent=self.config.games_per_opponent,
                        check_baseline_gating=True,
                        verbose=False,
                        early_stopping=self.config.early_stopping,
                        early_stopping_confidence=self.config.early_stopping_confidence,
                    ),
                ),
                timeout=self.config.evaluation_timeout_seconds,
            )

            # Extract win rates from result
            random_wins = 0
            random_games = 0
            heuristic_wins = 0
            heuristic_games = 0

            for opponent, stats in result.opponent_results.items():
                if "random" in opponent.lower():
                    random_wins = stats.get("wins", 0)
                    random_games = stats.get("games", 0)
                elif "heuristic" in opponent.lower():
                    heuristic_wins = stats.get("wins", 0)
                    heuristic_games = stats.get("games", 0)

            return {
                "random_win_rate": random_wins / max(random_games, 1),
                "heuristic_win_rate": heuristic_wins / max(heuristic_games, 1),
                "random_wins": random_wins,
                "random_games": random_games,
                "heuristic_wins": heuristic_wins,
                "heuristic_games": heuristic_games,
                "total_games": result.total_games,
                "passed_baseline_gating": result.passed_baseline_gating,
            }

        except asyncio.TimeoutError:
            logger.error(f"[AutoEvaluation] Gauntlet timed out for {model_path}")
            return None
        except Exception as e:
            logger.error(f"[AutoEvaluation] Gauntlet error: {e}")
            return None

    async def _emit_evaluation_complete(
        self, state: ModelEvaluationState, result: dict[str, Any]
    ) -> None:
        """Emit EVALUATION_COMPLETE event."""
        try:
            from app.coordination.event_router import (
                StageCompletionResult,
                StageEvent,
                get_stage_event_bus,
            )

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.EVALUATION_COMPLETE,
                    success=state.overall_passed,
                    timestamp=__import__("datetime").datetime.now().isoformat(),
                    metadata={
                        "model_path": state.model_path,
                        "config": state.config_key,
                        "board_type": state.board_type,
                        "num_players": state.num_players,
                        "random_win_rate": state.random_win_rate,
                        "heuristic_win_rate": state.heuristic_win_rate,
                        "passed_random": state.passed_random,
                        "passed_heuristic": state.passed_heuristic,
                        "overall_passed": state.overall_passed,
                        **result,
                    },
                )
            )
            logger.debug(
                f"[AutoEvaluation] Emitted EVALUATION_COMPLETE for {state.config_key}"
            )

        except Exception as e:
            logger.warning(f"[AutoEvaluation] Failed to emit evaluation event: {e}")

    async def _trigger_promotion(self, state: ModelEvaluationState) -> None:
        """Trigger model promotion."""
        try:
            logger.info(
                f"[AutoEvaluation] Triggering promotion for {state.config_key}: "
                f"{state.model_path}"
            )

            # Try to use promotion controller
            from app.coordination.promotion_controller import get_promotion_controller

            controller = get_promotion_controller()
            success = await controller.promote_model(
                model_path=state.model_path,
                board_type=state.board_type,
                num_players=state.num_players,
                sync_to_cluster=self.config.sync_to_cluster,
            )

            if success:
                logger.info(
                    f"[AutoEvaluation] Promoted {state.config_key} successfully"
                )

                # Emit promotion event
                await self._emit_promotion_triggered(state)
            else:
                logger.warning(
                    f"[AutoEvaluation] Promotion failed for {state.config_key}"
                )

        except ImportError:
            # Fallback to direct promotion
            logger.info(
                f"[AutoEvaluation] Using direct promotion for {state.config_key}"
            )
            await self._direct_promote(state)
        except Exception as e:
            logger.error(f"[AutoEvaluation] Promotion error: {e}")

    async def _direct_promote(self, state: ModelEvaluationState) -> None:
        """Direct promotion without controller."""
        try:
            import shutil

            model_path = Path(state.model_path)
            models_dir = Path(__file__).parent.parent.parent / "models"
            canonical_name = f"canonical_{state.board_type}_{state.num_players}p.pth"
            canonical_path = models_dir / canonical_name

            # Backup existing
            if canonical_path.exists():
                backup_path = models_dir / f"{canonical_name}.backup"
                shutil.copy2(canonical_path, backup_path)

            # Copy new model
            shutil.copy2(model_path, canonical_path)

            # Update symlink
            symlink_name = f"ringrift_best_{state.board_type}_{state.num_players}p.pth"
            symlink_path = models_dir / symlink_name
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(canonical_name)

            logger.info(f"[AutoEvaluation] Direct promotion complete: {canonical_path}")

            await self._emit_promotion_triggered(state)

        except Exception as e:
            logger.error(f"[AutoEvaluation] Direct promotion error: {e}")

    async def _emit_promotion_triggered(self, state: ModelEvaluationState) -> None:
        """Emit PROMOTION_TRIGGERED event."""
        try:
            from app.coordination.event_router import (
                StageCompletionResult,
                StageEvent,
                get_stage_event_bus,
            )

            bus = get_stage_event_bus()
            await bus.emit(
                StageCompletionResult(
                    event=StageEvent.MODEL_PROMOTED,
                    success=True,
                    timestamp=__import__("datetime").datetime.now().isoformat(),
                    metadata={
                        "model_path": state.model_path,
                        "config": state.config_key,
                        "board_type": state.board_type,
                        "num_players": state.num_players,
                        "random_win_rate": state.random_win_rate,
                        "heuristic_win_rate": state.heuristic_win_rate,
                    },
                )
            )
            logger.debug(
                f"[AutoEvaluation] Emitted MODEL_PROMOTED for {state.config_key}"
            )

        except Exception as e:
            logger.warning(f"[AutoEvaluation] Failed to emit promotion event: {e}")

    def queue_evaluation(
        self, model_path: str, board_type: str, num_players: int
    ) -> None:
        """Public method to queue a model for evaluation."""
        safe_create_task(
            self._queue_evaluation(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
            ),
            name=f"queue_eval_{Path(model_path).stem}",
        )

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self._running,
            "queue_length": len(self._evaluation_queue),
            "pending_evaluations": self._evaluation_queue[:10],
            "evaluation_states": {
                path: {
                    "config": state.config_key,
                    "pending": state.evaluation_pending,
                    "in_progress": state.evaluation_in_progress,
                    "last_evaluation": state.last_evaluation_time,
                    "random_win_rate": state.random_win_rate,
                    "heuristic_win_rate": state.heuristic_win_rate,
                    "passed": state.overall_passed,
                    "failures": state.consecutive_failures,
                }
                for path, state in self._evaluation_states.items()
            },
        }


# Singleton instance
_daemon: AutoEvaluationDaemon | None = None


def get_auto_evaluation_daemon() -> AutoEvaluationDaemon:
    """Get or create the singleton auto evaluation daemon."""
    global _daemon
    if _daemon is None:
        _daemon = AutoEvaluationDaemon()
    return _daemon


async def start_auto_evaluation_daemon() -> AutoEvaluationDaemon:
    """Start the auto evaluation daemon (convenience function)."""
    daemon = get_auto_evaluation_daemon()
    await daemon.start()
    return daemon


__all__ = [
    "AutoEvaluationConfig",
    "AutoEvaluationDaemon",
    "ModelEvaluationState",
    "get_auto_evaluation_daemon",
    "start_auto_evaluation_daemon",
]
