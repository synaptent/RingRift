"""Training cleanup handler for RingRift training.

December 2025: Extracted from train.py to improve modularity.

This module provides the TrainCleanupHandler class which handles all
cleanup operations at the end of training, including:
- Hardened event emission (always emits completion/failure)
- Async checkpointer shutdown
- Heartbeat monitor stop
- Graceful shutdown handler teardown
- Enhancements manager cleanup
- Distributed training cleanup
- Training result construction

Usage:
    from app.training.train_cleanup_handler import (
        TrainCleanupHandler,
        CleanupConfig,
        TrainingResult,
    )

    config = CleanupConfig(emit_events=True)
    handler = TrainCleanupHandler(config)

    # Normal completion
    result = handler.cleanup(context, completed_normally=True)

    # Or in a finally block
    try:
        # Training loop
        ...
    except Exception as e:
        result = handler.cleanup(context, completed_normally=False, exception=e)
        raise
    finally:
        result = handler.cleanup(context, ...)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.training.train_context import TrainContext

logger = logging.getLogger(__name__)

# Feature flags for optional event emission
try:
    from app.coordination.event_router import (
        DataEvent,
        DataEventType,
        get_router,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    get_router = None
    DataEvent = None
    DataEventType = None


# =============================================================================
# Configuration and Result Types
# =============================================================================


@dataclass
class CleanupConfig:
    """Configuration for training cleanup.

    Controls which cleanup operations are performed.
    """

    # Event emission
    emit_events: bool = True
    emit_hardened_events: bool = True  # Always emit in finally block

    # Cleanup options
    shutdown_async_checkpointer: bool = True
    stop_heartbeat: bool = True
    teardown_shutdown_handler: bool = True
    stop_enhancements: bool = True
    cleanup_distributed: bool = True


@dataclass
class TrainingResult:
    """Result from a training run.

    Contains all metrics and state from training completion.
    """

    # Core metrics
    best_val_loss: float = float("inf")
    final_train_loss: float = float("inf")
    final_val_loss: float = float("inf")
    epochs_completed: int = 0

    # Per-epoch tracking
    epoch_losses: list[dict[str, float]] = field(default_factory=list)

    # Completion status
    completed_normally: bool = False
    early_stopped: bool = False
    exception: Exception | None = None

    # Timing
    duration_seconds: float = 0.0

    # Checkpoint info
    final_checkpoint_path: str = ""
    best_model_path: str = ""

    # Optional metrics
    final_policy_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary with training results
        """
        return {
            "best_val_loss": float(self.best_val_loss),
            "final_train_loss": float(self.final_train_loss),
            "final_val_loss": float(self.final_val_loss),
            "epochs_completed": self.epochs_completed,
            "epoch_losses": self.epoch_losses,
            "completed_normally": self.completed_normally,
            "early_stopped": self.early_stopped,
            "duration_seconds": self.duration_seconds,
            "final_checkpoint_path": self.final_checkpoint_path,
        }


@dataclass
class EnhancementSummary:
    """Summary of enhancement statistics from training.

    Captures reanalysis and distillation stats for logging.
    """

    reanalysis_enabled: bool = False
    reanalysis_positions: int = 0
    reanalysis_games: int = 0
    reanalysis_blend_ratio: float = 0.0

    distillation_enabled: bool = False
    distillation_last_epoch: int = 0
    distillation_teachers: int = 0
    distillation_temperature: float = 1.0


# =============================================================================
# Cleanup Handler
# =============================================================================


class TrainCleanupHandler:
    """Handles all training cleanup operations.

    Ensures proper cleanup regardless of how training ends:
    - Normal completion
    - Early stopping
    - Exception/crash

    The key feature is "hardened event emission" which always emits
    a TRAINING_COMPLETED or TRAINING_FAILED event, ensuring the
    feedback loop never breaks.

    Example:
        handler = TrainCleanupHandler(CleanupConfig())

        try:
            # Training loop
            result = train_loop(...)
            return handler.cleanup(context, completed_normally=True, result=result)
        except Exception as e:
            handler.cleanup(context, completed_normally=False, exception=e)
            raise
    """

    def __init__(self, config: CleanupConfig | None = None):
        """Initialize the handler.

        Args:
            config: Cleanup configuration
        """
        self.config = config or CleanupConfig()

    def cleanup(
        self,
        context: "TrainContext",
        completed_normally: bool = False,
        exception: Exception | None = None,
        result: TrainingResult | None = None,
        start_time: float | None = None,
    ) -> TrainingResult:
        """Perform all cleanup operations.

        This should be called in a finally block to ensure cleanup always runs.

        Args:
            context: Training context with all components
            completed_normally: Whether training completed successfully
            exception: Exception if training failed
            result: Partial training result (if available)
            start_time: Training start time for duration calculation

        Returns:
            TrainingResult with final metrics
        """
        # Initialize result if not provided
        if result is None:
            result = TrainingResult()

        result.completed_normally = completed_normally
        result.exception = exception

        # Calculate duration
        if start_time is not None:
            result.duration_seconds = time.time() - start_time

        # Log enhancement summaries
        if context.is_main_process:
            self._log_enhancement_summaries(context)

        # Emit hardened events
        if self.config.emit_hardened_events and context.is_main_process:
            self._emit_hardened_events(context, result)

        # Shutdown async checkpointer
        if self.config.shutdown_async_checkpointer:
            self._shutdown_async_checkpointer(context)

        # Stop heartbeat monitor
        if self.config.stop_heartbeat:
            self._stop_heartbeat(context)

        # Teardown shutdown handler
        if self.config.teardown_shutdown_handler:
            self._teardown_shutdown_handler(context)

        # Stop enhancements
        if self.config.stop_enhancements:
            self._stop_enhancements(context)

        # Cleanup distributed
        if self.config.cleanup_distributed:
            self._cleanup_distributed(context)

        return result

    def build_result(
        self,
        context: "TrainContext",
        completed_normally: bool = True,
        early_stopped: bool = False,
    ) -> TrainingResult:
        """Build a TrainingResult from the current context.

        Args:
            context: Training context
            completed_normally: Whether training completed normally
            early_stopped: Whether early stopping triggered

        Returns:
            TrainingResult with current state
        """
        progress = context.progress

        return TrainingResult(
            best_val_loss=progress.best_val_loss,
            final_train_loss=progress.current_train_loss,
            final_val_loss=progress.current_val_loss,
            epochs_completed=progress.epochs_completed,
            epoch_losses=progress.epoch_losses,
            completed_normally=completed_normally,
            early_stopped=early_stopped,
            duration_seconds=time.time() - progress.start_time,
            final_checkpoint_path=progress.last_good_checkpoint_path or "",
            best_model_path=context.save_path,
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _log_enhancement_summaries(self, context: "TrainContext") -> None:
        """Log enhancement statistics summaries.

        Args:
            context: Training context
        """
        if context.enhancements_manager is None:
            return

        try:
            # Reanalysis stats
            reanalysis_stats = context.enhancements_manager.get_reanalysis_stats()
            if (
                reanalysis_stats.get("enabled")
                and reanalysis_stats.get("positions_reanalyzed", 0) > 0
            ):
                logger.info(
                    f"[Reanalysis Summary] "
                    f"Positions: {reanalysis_stats['positions_reanalyzed']}, "
                    f"Games: {reanalysis_stats['games_reanalyzed']}, "
                    f"Blend ratio: {reanalysis_stats['blend_ratio']:.2f}"
                )

            # Distillation stats
            distillation_stats = context.enhancements_manager.get_distillation_stats()
            if (
                distillation_stats.get("enabled")
                and distillation_stats.get("last_distillation_epoch", 0) > 0
            ):
                logger.info(
                    f"[Distillation Summary] "
                    f"Last epoch: {distillation_stats['last_distillation_epoch']}, "
                    f"Teachers: {distillation_stats['available_teachers']}, "
                    f"Temperature: {distillation_stats['temperature']:.1f}"
                )
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Failed to get enhancement stats: {e}")

    def _emit_hardened_events(
        self,
        context: "TrainContext",
        result: TrainingResult,
    ) -> None:
        """Emit hardened training completion/failure events.

        These events ALWAYS emit in the finally block to ensure the
        feedback loop never breaks.

        Args:
            context: Training context
            result: Training result
        """
        if not HAS_EVENT_BUS or get_router is None:
            return

        try:
            config_key = f"{context.config.board_type.value}_{context.resolved.num_players}p"
            router = get_router()

            if result.completed_normally:
                # Training succeeded
                payload = {
                    "epochs_completed": result.epochs_completed,
                    "best_val_loss": float(result.best_val_loss),
                    "final_train_loss": float(result.final_train_loss),
                    "final_val_loss": float(result.final_val_loss),
                    "config": config_key,
                    "board_type": context.config.board_type.value,
                    "num_players": context.resolved.num_players,
                    "duration_seconds": result.duration_seconds,
                    "hardened_emit": True,  # Flag indicating finally block origin
                    "trigger_evaluation": True,
                    "model_path": context.save_path,
                    "policy_accuracy": result.final_policy_accuracy,
                }

                if result.final_checkpoint_path:
                    payload["checkpoint_path"] = result.final_checkpoint_path

                # Add enhancement stats
                self._add_enhancement_stats_to_payload(context, payload)

                router.publish_sync(DataEvent(
                    event_type=DataEventType.TRAINING_COMPLETED,
                    payload=payload,
                    source="train_cleanup",
                ))
                logger.info(f"[train] Hardened TRAINING_COMPLETED emitted for {config_key}")

                # Emit curriculum update if policy accuracy is high
                self._emit_curriculum_update(context, result, config_key)

            else:
                # Training failed
                error_msg = str(result.exception) if result.exception else "Unknown error"
                router.publish_sync(DataEvent(
                    event_type=DataEventType.TRAINING_FAILED,
                    payload={
                        "config": config_key,
                        "error": error_msg,
                        "epochs_completed": result.epochs_completed,
                        "duration_seconds": result.duration_seconds,
                        "best_val_loss": (
                            float(result.best_val_loss)
                            if result.best_val_loss != float("inf")
                            else None
                        ),
                    },
                    source="train_cleanup",
                ))
                logger.warning(
                    f"[train] Hardened TRAINING_FAILED emitted for {config_key}: {error_msg}"
                )

        except (RuntimeError, ConnectionError, TimeoutError, AttributeError, NameError) as e:
            logger.debug(f"Failed to emit hardened training event: {e}")

    def _add_enhancement_stats_to_payload(
        self,
        context: "TrainContext",
        payload: dict[str, Any],
    ) -> None:
        """Add enhancement statistics to event payload.

        Args:
            context: Training context
            payload: Event payload to update
        """
        if context.enhancements_manager is None:
            return

        try:
            reanalysis_stats = context.enhancements_manager.get_reanalysis_stats()
            if reanalysis_stats.get("enabled"):
                payload["reanalysis"] = reanalysis_stats

            distillation_stats = context.enhancements_manager.get_distillation_stats()
            if distillation_stats.get("enabled"):
                payload["distillation"] = distillation_stats
        except (AttributeError, TypeError) as e:
            logger.debug(f"Failed to get enhancement stats for payload: {e}")

    def _emit_curriculum_update(
        self,
        context: "TrainContext",
        result: TrainingResult,
        config_key: str,
    ) -> None:
        """Emit curriculum update event if policy accuracy is high.

        Args:
            context: Training context
            result: Training result
            config_key: Configuration key
        """
        try:
            from app.coordination.event_emission_helpers import safe_emit_event

            policy_accuracy_threshold = 0.75

            if result.final_policy_accuracy < policy_accuracy_threshold:
                return

            # Increase curriculum weight for well-performing configs
            new_weight = 1.0 + (result.final_policy_accuracy - 0.5) * 0.5

            safe_emit_event(
                "CURRICULUM_UPDATED",
                {
                    "config_key": config_key,
                    "new_weight": new_weight,
                    "trigger": "training_complete",
                    "policy_accuracy": result.final_policy_accuracy,
                    "value_loss": result.final_val_loss,
                },
                log_after=f"Triggered reweight for {config_key}: policy_acc={result.final_policy_accuracy:.1%} → weight={new_weight:.3f}",
                context="curriculum",
            )
        except ImportError:
            pass

    def _shutdown_async_checkpointer(self, context: "TrainContext") -> None:
        """Shutdown async checkpointer and wait for pending saves.

        Args:
            context: Training context
        """
        if context.async_checkpointer is None:
            return

        try:
            context.async_checkpointer.shutdown()
            logger.info("Async checkpointer shutdown complete")
        except (RuntimeError, OSError) as e:
            logger.warning(f"Async checkpointer shutdown error: {e}")

    def _stop_heartbeat(self, context: "TrainContext") -> None:
        """Stop heartbeat monitor.

        Args:
            context: Training context
        """
        if context.heartbeat_monitor is None:
            return

        try:
            context.heartbeat_monitor.stop()
            logger.info("Heartbeat monitor stopped")
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Heartbeat stop error: {e}")

    def _teardown_shutdown_handler(self, context: "TrainContext") -> None:
        """Teardown graceful shutdown handler.

        Args:
            context: Training context
        """
        if context.shutdown_handler is None:
            return

        try:
            context.shutdown_handler.teardown()
            logger.debug("Graceful shutdown handler teardown complete")
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Shutdown handler teardown error: {e}")

    def _stop_enhancements(self, context: "TrainContext") -> None:
        """Stop integrated enhancements background services.

        Args:
            context: Training context
        """
        if context.enhancements_manager is None:
            return

        try:
            context.enhancements_manager.stop_background_services()
            logger.info("Integrated enhancements background services stopped")
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Enhancements stop error: {e}")

    def _cleanup_distributed(self, context: "TrainContext") -> None:
        """Clean up distributed process group.

        Args:
            context: Training context
        """
        if not context.distributed:
            return

        try:
            from app.training.distributed_utils import cleanup_distributed
            cleanup_distributed()
            logger.info("Distributed process group cleaned up")
        except ImportError:
            pass
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Distributed cleanup error: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_cleanup_handler(
    emit_events: bool = True,
    emit_hardened_events: bool = True,
    **kwargs: Any,
) -> TrainCleanupHandler:
    """Create a TrainCleanupHandler with the specified settings.

    Args:
        emit_events: Enable event emission
        emit_hardened_events: Always emit in finally block
        **kwargs: Additional config parameters

    Returns:
        Configured TrainCleanupHandler instance
    """
    config = CleanupConfig(
        emit_events=emit_events,
        emit_hardened_events=emit_hardened_events,
        **{k: v for k, v in kwargs.items() if hasattr(CleanupConfig, k)},
    )
    return TrainCleanupHandler(config)


def build_training_result(
    best_val_loss: float,
    final_train_loss: float,
    final_val_loss: float,
    epochs_completed: int,
    epoch_losses: list[dict[str, float]] | None = None,
    **kwargs: Any,
) -> TrainingResult:
    """Build a TrainingResult from training metrics.

    Convenience function for backward compatibility with train_model().

    Args:
        best_val_loss: Best validation loss
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        epochs_completed: Number of epochs completed
        epoch_losses: Per-epoch loss records
        **kwargs: Additional result fields

    Returns:
        TrainingResult instance
    """
    return TrainingResult(
        best_val_loss=best_val_loss,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        epochs_completed=epochs_completed,
        epoch_losses=epoch_losses or [],
        completed_normally=kwargs.get("completed_normally", True),
        early_stopped=kwargs.get("early_stopped", False),
        duration_seconds=kwargs.get("duration_seconds", 0.0),
        final_checkpoint_path=kwargs.get("final_checkpoint_path", ""),
        best_model_path=kwargs.get("best_model_path", ""),
        final_policy_accuracy=kwargs.get("final_policy_accuracy", 0.0),
    )
