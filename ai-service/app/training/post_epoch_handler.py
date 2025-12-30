"""Post-epoch handling for RingRift training.

December 2025: Extracted from train.py to improve modularity.

This module provides the PostEpochHandler class which handles all processing
that occurs after each training epoch completes, including:
- Regression detection
- Epoch record creation
- Event emissions (epoch completed, loss trends, plateau detection)
- Metrics recording (Prometheus, dashboard)
- Early stopping checks
- Checkpointing
- Best model saving
- Knowledge distillation
- Checkpoint averaging

Usage:
    from app.training.post_epoch_handler import PostEpochHandler, PostEpochConfig

    config = PostEpochConfig(
        checkpoint_interval=5,
        emit_events=True,
    )
    handler = PostEpochHandler(config)
    result = handler.handle_epoch_end(context, epoch, metrics)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from app.config.thresholds import MIN_TRAINING_EPOCHS
from app.training.checkpoint_unified import save_checkpoint, save_model_checkpoint

if TYPE_CHECKING:
    from app.training.train_context import TrainContext

logger = logging.getLogger(__name__)

# Feature flags for optional event emission
try:
    from app.coordination.event_emitters import (
        emit_training_loss_anomaly,
        emit_training_loss_trend,
        emit_plateau_detected,
    )
    from app.coordination.event_router import emit_training_early_stopped
    HAS_TRAINING_EVENTS = True
except ImportError:
    HAS_TRAINING_EVENTS = False

try:
    from app.coordination.event_emitters import publish_epoch_completed
    HAS_EPOCH_EVENTS = True
except ImportError:
    HAS_EPOCH_EVENTS = False

try:
    from app.monitoring.prometheus_metrics import (
        TRAINING_EPOCHS,
        TRAINING_LOSS,
        TRAINING_DURATION,
        CALIBRATION_ECE,
        CALIBRATION_MCE,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    from app.training.regression_detector import get_regression_detector
    HAS_REGRESSION_DETECTOR = True
except ImportError:
    HAS_REGRESSION_DETECTOR = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PostEpochConfig:
    """Configuration for post-epoch handling.

    Controls which post-epoch operations are performed and their parameters.
    """

    # Checkpointing
    checkpoint_interval: int = 5
    checkpoint_dir: str = "checkpoints"
    save_all_epochs: bool = True

    # Event emission
    emit_events: bool = True
    emit_prometheus: bool = True
    emit_dashboard: bool = True

    # Regression detection
    enable_regression_detection: bool = True

    # Calibration
    calibration_interval: int = 5

    # Trend analysis
    trend_interval: int = 5
    plateau_interval: int = 10
    anomaly_threshold: float = 2.0  # Current loss > N * average

    # Knowledge distillation
    enable_distillation: bool = True


@dataclass
class EpochMetrics:
    """Metrics from a completed epoch.

    Aggregates all metrics that need to be processed post-epoch.
    """

    epoch: int
    avg_train_loss: float
    avg_val_loss: float
    avg_policy_accuracy: float
    learning_rate: float

    # Optional metrics
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    train_batches: int = 0
    samples_per_second: float = 0.0
    epoch_duration: float = 0.0

    # Previous epochs for trend analysis
    epoch_losses: list[dict[str, float]] = field(default_factory=list)


@dataclass
class PostEpochResult:
    """Result from post-epoch handling.

    Contains flags and information about what actions were taken.
    """

    should_stop: bool = False
    stop_reason: str = ""

    # Checkpoint info
    checkpoint_saved: bool = False
    checkpoint_path: str = ""
    best_model_saved: bool = False

    # Metrics record
    epoch_record: dict[str, Any] = field(default_factory=dict)

    # Regression/anomaly detection
    regression_detected: bool = False
    anomaly_detected: bool = False
    plateau_detected: bool = False

    # Distillation
    distillation_triggered: bool = False


# =============================================================================
# Post-Epoch Handler
# =============================================================================


class PostEpochHandler:
    """Handles all post-epoch processing.

    Orchestrates:
    1. Regression detection
    2. Epoch record creation
    3. Event emissions
    4. Metrics recording
    5. Early stopping checks
    6. Checkpointing
    7. Best model saving
    8. Knowledge distillation

    Example:
        handler = PostEpochHandler(PostEpochConfig())
        result = handler.handle_epoch_end(
            context=train_context,
            metrics=EpochMetrics(epoch=5, avg_train_loss=0.5, ...),
        )
        if result.should_stop:
            break
    """

    def __init__(self, config: PostEpochConfig | None = None):
        """Initialize the handler.

        Args:
            config: Configuration for post-epoch handling
        """
        self.config = config or PostEpochConfig()
        self._best_val_loss = float("inf")
        self._last_good_checkpoint_path: str | None = None
        self._last_good_epoch: int = 0

    def handle_epoch_end(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> PostEpochResult:
        """Handle all post-epoch processing.

        Args:
            context: Training context with model, optimizer, etc.
            metrics: Metrics from the completed epoch

        Returns:
            PostEpochResult with flags and information
        """
        result = PostEpochResult()

        # Create epoch record
        result.epoch_record = self._create_epoch_record(context, metrics)

        # Regression detection
        if self.config.enable_regression_detection:
            result.regression_detected = self._check_regression(context, metrics)

        # Calibration metrics
        self._compute_calibration(context, metrics, result.epoch_record)

        # Training facade/mining stats
        self._log_training_stats(context, metrics, result.epoch_record)

        # Event emissions
        if self.config.emit_events:
            self._emit_epoch_events(context, metrics)
            result.anomaly_detected, result.plateau_detected = self._emit_loss_events(
                context, metrics
            )

        # Prometheus metrics
        if self.config.emit_prometheus and HAS_PROMETHEUS:
            self._record_prometheus_metrics(context, metrics, result.epoch_record)

        # Dashboard metrics
        if self.config.emit_dashboard and context.metrics_collector is not None:
            self._record_dashboard_metrics(context, metrics, result.epoch_record)

        # Early stopping checks
        should_stop, stop_reason = self._check_early_stopping(context, metrics)
        if should_stop:
            result.should_stop = True
            result.stop_reason = stop_reason
            # Handle early stopping (save checkpoint, restore weights, etc.)
            self._handle_early_stop(context, metrics, result)
            return result

        # Checkpointing at intervals
        if self._should_checkpoint(context, metrics):
            result.checkpoint_saved, result.checkpoint_path = self._save_checkpoint(
                context, metrics
            )

        # Best model saving
        if metrics.avg_val_loss < self._best_val_loss:
            self._best_val_loss = metrics.avg_val_loss
            result.best_model_saved = self._save_best_model(context, metrics)

        # Knowledge distillation
        if self.config.enable_distillation:
            result.distillation_triggered = self._maybe_distill(context, metrics)

        # Heartbeat
        if context.heartbeat_monitor is not None:
            context.heartbeat_monitor.beat()

        # Circuit breaker success
        if context.training_breaker is not None:
            context.training_breaker.record_success("training_epoch")

        return result

    def handle_training_complete(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> PostEpochResult:
        """Handle final checkpoint and averaging when training completes normally.

        Called when the training loop completes without early stopping.

        Args:
            context: Training context
            metrics: Final epoch metrics

        Returns:
            PostEpochResult with final checkpoint info
        """
        result = PostEpochResult()

        if not context.is_main_process:
            return result

        model_to_save = context.model_to_save
        checkpoint_dir = Path(context.resolved.checkpoint_dir)

        # Save final checkpoint
        final_checkpoint_path = checkpoint_dir / f"checkpoint_final_epoch_{context.config.epochs}.pth"

        if context.async_checkpointer is not None:
            context.async_checkpointer.save_async(
                model_to_save,
                context.optimizer,
                context.config.epochs - 1,
                metrics.avg_val_loss,
                str(final_checkpoint_path),
                scheduler=context.epoch_scheduler,
                early_stopping=context.early_stopper,
            )
        else:
            save_checkpoint(
                model_to_save,
                context.optimizer,
                context.config.epochs - 1,
                metrics.avg_val_loss,
                str(final_checkpoint_path),
                scheduler=context.epoch_scheduler,
                early_stopping=context.early_stopper,
            )

        result.checkpoint_saved = True
        result.checkpoint_path = str(final_checkpoint_path)
        logger.info("Training completed. Final checkpoint saved.")

        # Apply checkpoint averaging
        self._apply_checkpoint_averaging(context, metrics, model_to_save)

        return result

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _create_epoch_record(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> dict[str, Any]:
        """Create epoch record for downstream analysis.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            Dictionary with epoch metrics
        """
        return {
            "epoch": metrics.epoch + 1,
            "train_loss": float(metrics.avg_train_loss),
            "val_loss": float(metrics.avg_val_loss),
            "policy_accuracy": float(metrics.avg_policy_accuracy),
            "lr": metrics.learning_rate,
            "train_batches": metrics.train_batches,
            "samples_per_second": metrics.samples_per_second,
            "epoch_duration": metrics.epoch_duration,
        }

    def _check_regression(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> bool:
        """Check for performance regression.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            True if regression detected
        """
        if not HAS_REGRESSION_DETECTOR:
            return False

        if metrics.epoch < 2 or not context.is_main_process:
            return False

        try:
            regression_detector = get_regression_detector(connect_event_bus=True)
            model_id = f"{context.config.board_type.value}_{context.resolved.num_players}p"

            # Set baseline on first check
            if metrics.epoch == 2:
                regression_detector.set_baseline(
                    model_id=model_id,
                    elo=self._best_val_loss * -1000,  # Convert loss to pseudo-Elo
                )

            # Check for regression
            regression_event = regression_detector.check_regression(
                model_id=model_id,
                current_elo=metrics.avg_val_loss * -1000,
                games_played=metrics.epoch + 1,
            )

            if regression_event is not None:
                logger.warning(
                    f"[RegressionDetector] {regression_event.severity.value.upper()} regression: "
                    f"val_loss {metrics.avg_val_loss:.4f} vs best {self._best_val_loss:.4f} "
                    f"({regression_event.reason})"
                )
                return True

        except (AttributeError, ValueError, TypeError) as e:
            logger.debug(f"Regression detection error: {e}")

        return False

    def _compute_calibration(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        epoch_record: dict[str, Any],
    ) -> None:
        """Compute and log calibration metrics.

        Args:
            context: Training context
            metrics: Epoch metrics
            epoch_record: Record to update with calibration metrics
        """
        if context.calibration_tracker is None:
            return

        if (metrics.epoch + 1) % self.config.calibration_interval != 0:
            return

        calibration_report = context.calibration_tracker.compute_current_calibration()
        if calibration_report is None:
            return

        epoch_record["calibration_ece"] = calibration_report.ece
        epoch_record["calibration_mce"] = calibration_report.mce
        epoch_record["calibration_overconfidence"] = calibration_report.overconfidence

        if context.is_main_process:
            logger.info(
                f"  Calibration: ECE={calibration_report.ece:.4f}, "
                f"MCE={calibration_report.mce:.4f}, "
                f"Overconfidence={calibration_report.overconfidence:.4f}"
            )
            if calibration_report.optimal_temperature is not None:
                logger.info(
                    f"  Optimal temperature: {calibration_report.optimal_temperature:.3f}"
                )

    def _log_training_stats(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        epoch_record: dict[str, Any],
    ) -> None:
        """Log training facade and hard example mining stats.

        Args:
            context: Training context
            metrics: Epoch metrics
            epoch_record: Record to update with stats
        """
        if not context.is_main_process:
            return

        # Training facade stats
        if context.training_facade is not None:
            try:
                facade_stats = context.training_facade.on_epoch_end()
                if facade_stats.get("mining_active", False):
                    logger.info(
                        f"  [Training Facade] "
                        f"tracked={facade_stats.get('tracked_samples', 0)}, "
                        f"hard_frac={facade_stats.get('hard_examples_fraction', 0):.1%}, "
                        f"mean_loss={facade_stats.get('mean_per_sample_loss', 0):.4f}, "
                        f"lr_scale={facade_stats.get('curriculum_lr_scale', 1.0):.3f}"
                    )
                epoch_record["facade_mean_loss"] = facade_stats.get("mean_loss", 0)
                epoch_record["facade_hard_fraction"] = facade_stats.get("hard_examples_fraction", 0)
                epoch_record["facade_curriculum_lr_scale"] = facade_stats.get("curriculum_lr_scale", 1.0)
                epoch_record["facade_mining_active"] = facade_stats.get("mining_active", False)
            except (AttributeError, ValueError) as e:
                logger.debug(f"[Training Facade] on_epoch_end error: {e}")

        # Hard example mining stats (fallback)
        elif context.hard_example_miner is not None:
            mining_stats = context.hard_example_miner.get_statistics()
            if mining_stats.get("mining_active", False):
                logger.info(
                    f"  [Hard Example Mining] "
                    f"tracked={mining_stats.get('tracked_examples', 0)}, "
                    f"mean_loss={mining_stats.get('mean_loss', 0):.4f}, "
                    f"loss_p90={mining_stats.get('loss_p90', 0):.4f}"
                )
                epoch_record["hard_mining_mean_loss"] = mining_stats.get("mean_loss", 0)
                epoch_record["hard_mining_p90_loss"] = mining_stats.get("loss_p90", 0)
                epoch_record["hard_mining_tracked"] = mining_stats.get("tracked_examples", 0)

    def _emit_epoch_events(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> None:
        """Emit epoch completed event.

        Args:
            context: Training context
            metrics: Epoch metrics
        """
        if not HAS_EPOCH_EVENTS or not context.is_main_process:
            return

        try:
            config_key = f"{context.config.board_type.value}_{context.resolved.num_players}p"
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(publish_epoch_completed(
                    config_key=config_key,
                    epoch=metrics.epoch + 1,
                    total_epochs=context.config.epochs,
                    train_loss=metrics.avg_train_loss,
                    val_loss=metrics.avg_val_loss,
                    learning_rate=metrics.learning_rate,
                ))
            except RuntimeError:
                # No event loop running
                pass
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            logger.debug(f"Failed to emit epoch completed event: {e}")

    def _emit_loss_events(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> tuple[bool, bool]:
        """Emit training loss events (anomaly, trend, plateau).

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            Tuple of (anomaly_detected, plateau_detected)
        """
        if not HAS_TRAINING_EVENTS or not context.is_main_process:
            return False, False

        anomaly_detected = False
        plateau_detected = False

        try:
            config_key = f"{context.config.board_type.value}_{context.resolved.num_players}p"

            # Calculate average loss over recent epochs
            recent_losses = [
                e.get("val_loss", e.get("train_loss", 0.0))
                for e in metrics.epoch_losses[-5:]
                if e
            ]

            if not recent_losses:
                return False, False

            avg_recent_loss = sum(recent_losses) / len(recent_losses)

            # Anomaly detection
            if (
                metrics.avg_val_loss > avg_recent_loss * self.config.anomaly_threshold
                and len(metrics.epoch_losses) > 2
            ):
                anomaly_detected = True
                anomaly_ratio = (
                    metrics.avg_val_loss / avg_recent_loss
                    if avg_recent_loss > 0
                    else 0.0
                )
                logger.warning(
                    f"[TRAINING ANOMALY] Loss spike detected: "
                    f"{metrics.avg_val_loss:.4f} vs avg {avg_recent_loss:.4f} "
                    f"(ratio: {anomaly_ratio:.2f}x)"
                )
                self._emit_async(emit_training_loss_anomaly(
                    config_key=config_key,
                    current_loss=metrics.avg_val_loss,
                    avg_loss=avg_recent_loss,
                    epoch=metrics.epoch + 1,
                    anomaly_ratio=anomaly_ratio,
                    source="post_epoch_handler.py",
                ))

            # Trend analysis every N epochs
            if (
                (metrics.epoch + 1) % self.config.trend_interval == 0
                and len(metrics.epoch_losses) >= 5
            ):
                self._emit_trend_event(context, metrics, config_key, recent_losses)

            # Plateau detection every M epochs
            if (
                (metrics.epoch + 1) % self.config.plateau_interval == 0
                and len(metrics.epoch_losses) >= 10
            ):
                plateau_detected = self._check_and_emit_plateau(
                    context, metrics, config_key
                )

        except (RuntimeError, ConnectionError, TimeoutError, AttributeError) as e:
            logger.debug(f"Failed to emit training events: {e}")

        return anomaly_detected, plateau_detected

    def _emit_trend_event(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        config_key: str,
        recent_losses: list[float],
    ) -> None:
        """Emit training trend event.

        Args:
            context: Training context
            metrics: Epoch metrics
            config_key: Config identifier
            recent_losses: Recent loss values
        """
        current_avg = sum(recent_losses) / len(recent_losses)
        older_losses = [
            e.get("val_loss", e.get("train_loss", 0.0))
            for e in metrics.epoch_losses[-10:-5]
            if e
        ]

        if not older_losses:
            return

        previous_avg = sum(older_losses) / len(older_losses)
        improvement_rate = (
            (previous_avg - current_avg) / previous_avg
            if previous_avg > 0
            else 0.0
        )

        # Classify trend
        if improvement_rate > 0.05:
            trend = "improving"
        elif improvement_rate < -0.05:
            trend = "degrading"
        else:
            trend = "stalled"

        logger.info(
            f"[TRAINING TREND] {trend} (epoch {metrics.epoch + 1}): "
            f"current_avg={current_avg:.4f}, previous_avg={previous_avg:.4f}, "
            f"improvement_rate={improvement_rate:.2%}"
        )

        self._emit_async(emit_training_loss_trend(
            config_key=config_key,
            trend=trend,
            epoch=metrics.epoch + 1,
            current_loss=current_avg,
            previous_loss=previous_avg,
            improvement_rate=improvement_rate,
            source="post_epoch_handler.py",
        ))

    def _check_and_emit_plateau(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        config_key: str,
    ) -> bool:
        """Check for plateau and emit event if detected.

        Args:
            context: Training context
            metrics: Epoch metrics
            config_key: Config identifier

        Returns:
            True if plateau detected
        """
        last_10_losses = [
            e.get("val_loss", e.get("train_loss", 0.0))
            for e in metrics.epoch_losses[-10:]
            if e
        ]
        prev_10_losses = [
            e.get("val_loss", e.get("train_loss", 0.0))
            for e in metrics.epoch_losses[-20:-10]
            if e
        ]

        if len(last_10_losses) < 10 or len(prev_10_losses) < 5:
            return False

        last_10_avg = sum(last_10_losses) / len(last_10_losses)
        prev_10_avg = sum(prev_10_losses) / len(prev_10_losses)
        long_term_improvement = (
            (prev_10_avg - last_10_avg) / prev_10_avg
            if prev_10_avg > 0
            else 0.0
        )

        # Plateau: < 0.1% improvement over 10 epochs
        if abs(long_term_improvement) >= 0.001:
            return False

        # Analyze plateau type
        last_10_train = [
            e.get("train_loss", 0.0)
            for e in metrics.epoch_losses[-10:]
            if e
        ]
        last_10_train_avg = (
            sum(last_10_train) / len(last_10_train)
            if last_10_train
            else 0.0
        )
        train_val_gap = last_10_avg - last_10_train_avg

        if train_val_gap > 0.05:
            plateau_type = "overfitting"
            recommendation = "reduce_epochs"
            exploration_boost = 1.5
        else:
            plateau_type = "data_limitation"
            recommendation = "more_games"
            exploration_boost = 1.3

        logger.warning(
            f"[TRAINING PLATEAU] Detected at epoch {metrics.epoch + 1}: "
            f"<0.1% improvement over 10 epochs "
            f"(last_10={last_10_avg:.5f}, prev_10={prev_10_avg:.5f}, "
            f"type={plateau_type}, gap={train_val_gap:.4f})"
        )

        # Emit trend event
        self._emit_async(emit_training_loss_trend(
            config_key=config_key,
            trend="plateau",
            epoch=metrics.epoch + 1,
            current_loss=last_10_avg,
            previous_loss=prev_10_avg,
            improvement_rate=long_term_improvement,
            source="post_epoch_handler.py",
            window_size=10,
        ))

        # Emit plateau detected event
        self._emit_async(emit_plateau_detected(
            metric_name="validation_loss",
            current_value=last_10_avg,
            best_value=prev_10_avg,
            epochs_since_improvement=10,
            plateau_type=plateau_type,
            config_key=config_key,
            epoch=metrics.epoch + 1,
            recommendation=recommendation,
            exploration_boost=exploration_boost,
            train_val_gap=train_val_gap,
            source="post_epoch_handler.py",
        ))

        return True

    def _emit_async(self, coro: Any) -> None:
        """Fire-and-forget async emission.

        Args:
            coro: Coroutine to execute
        """
        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(coro)
        except RuntimeError:
            # No event loop running
            pass

    def _record_prometheus_metrics(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        epoch_record: dict[str, Any],
    ) -> None:
        """Record Prometheus metrics.

        Args:
            context: Training context
            metrics: Epoch metrics
            epoch_record: Epoch record with additional metrics
        """
        if not context.is_main_process:
            return

        config_label = f"{context.config.board_type.value}_{context.resolved.num_players}p"
        TRAINING_EPOCHS.labels(config=config_label).inc()
        TRAINING_LOSS.labels(config=config_label, loss_type="train").set(
            metrics.avg_train_loss
        )
        TRAINING_LOSS.labels(config=config_label, loss_type="val").set(
            metrics.avg_val_loss
        )
        TRAINING_DURATION.labels(config=config_label).observe(
            epoch_record.get("epoch_duration", 0.0)
        )

        if "calibration_ece" in epoch_record:
            CALIBRATION_ECE.labels(config=config_label).set(
                epoch_record["calibration_ece"]
            )
            CALIBRATION_MCE.labels(config=config_label).set(
                epoch_record["calibration_mce"]
            )

    def _record_dashboard_metrics(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        epoch_record: dict[str, Any],
    ) -> None:
        """Record metrics to dashboard collector.

        Args:
            context: Training context
            metrics: Epoch metrics
            epoch_record: Epoch record with additional metrics
        """
        if not context.is_main_process:
            return

        try:
            # Get GPU memory usage
            gpu_memory_mb = 0.0
            if context.device.type == "cuda":
                gpu_memory_mb = torch.cuda.memory_allocated(context.device) / (1024 * 1024)

            context.metrics_collector.record_training_step(
                epoch=metrics.epoch + 1,
                step=epoch_record.get("train_batches", 0),
                loss=metrics.avg_val_loss,
                policy_loss=metrics.avg_policy_loss,
                value_loss=metrics.avg_value_loss,
                accuracy=metrics.avg_policy_accuracy,
                learning_rate=metrics.learning_rate,
                batch_size=context.config.batch_size,
                samples_per_second=metrics.samples_per_second,
                gpu_memory_mb=gpu_memory_mb,
                model_id=getattr(context.config, "model_id", "unknown"),
            )
        except (OSError, RuntimeError, AttributeError) as e:
            logger.debug(f"Failed to record metrics to dashboard: {e}")

    def _check_early_stopping(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> tuple[bool, str]:
        """Check if training should stop early.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            Tuple of (should_stop, reason)
        """
        model_to_save = context.model_to_save

        # Check enhancements manager early stopping
        if context.enhancements_manager is not None:
            if context.enhancements_manager.should_early_stop():
                if context.is_main_process:
                    logger.info(
                        f"Enhancements manager triggered early stop at epoch {metrics.epoch + 1} "
                        "(Elo regression detected)"
                    )
                return True, "elo_regression"

            # Baseline gating check
            passes_gating, failed_baselines, consecutive_failures = (
                context.enhancements_manager.get_baseline_gating_status()
            )
            if not passes_gating and context.is_main_process:
                logger.warning(
                    f"[BASELINE GATING] Epoch {metrics.epoch + 1}: Model failed baseline thresholds "
                    f"({', '.join(failed_baselines)}). Consecutive failures: {consecutive_failures}"
                )
                if consecutive_failures >= 5:
                    logger.error(
                        f"[BASELINE GATING] {consecutive_failures} consecutive failures! "
                        "Model may be overfitting to neural-vs-neural play."
                    )

        # Check standard early stopper
        if context.early_stopper is not None:
            current_elo = None
            if context.enhancements_manager is not None:
                current_elo = context.enhancements_manager.get_current_elo()

            should_stop = context.early_stopper.should_stop(
                val_loss=metrics.avg_val_loss,
                current_elo=current_elo,
                model=model_to_save,
                epoch=metrics.epoch,
            )

            # Enforce minimum epochs
            if should_stop and metrics.epoch + 1 < MIN_TRAINING_EPOCHS:
                if context.is_main_process:
                    logger.info(
                        f"Early stopping suppressed at epoch {metrics.epoch + 1} "
                        f"(minimum: {MIN_TRAINING_EPOCHS})"
                    )
                should_stop = False

            if should_stop:
                return True, "loss_stagnation"

        return False, ""

    def _handle_early_stop(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        result: PostEpochResult,
    ) -> None:
        """Handle early stopping (save checkpoint, restore weights, emit events).

        Args:
            context: Training context
            metrics: Epoch metrics
            result: Result to update
        """
        if not context.is_main_process:
            return

        early_stopper = context.early_stopper
        if early_stopper is None:
            return

        model_to_save = context.model_to_save
        checkpoint_dir = Path(context.resolved.checkpoint_dir)

        elo_info = (
            f", best Elo: {early_stopper.best_elo:.1f}"
            if early_stopper.best_elo > float("-inf")
            else ""
        )
        logger.info(
            f"Early stopping triggered at epoch {metrics.epoch + 1} "
            f"(best loss: {early_stopper.best_loss:.4f}{elo_info})"
        )

        # Emit early stopped event
        self._emit_early_stopped_event(context, metrics, early_stopper)

        # Restore best weights
        early_stopper.restore_best_weights(model_to_save)

        # Save final checkpoint
        final_checkpoint_path = checkpoint_dir / f"checkpoint_early_stop_epoch_{metrics.epoch + 1}.pth"

        if context.async_checkpointer is not None:
            context.async_checkpointer.save_async(
                model_to_save,
                context.optimizer,
                metrics.epoch,
                early_stopper.best_loss,
                str(final_checkpoint_path),
                scheduler=context.epoch_scheduler,
                early_stopping=early_stopper,
            )
        else:
            save_checkpoint(
                model_to_save,
                context.optimizer,
                metrics.epoch,
                early_stopper.best_loss,
                str(final_checkpoint_path),
                scheduler=context.epoch_scheduler,
                early_stopping=early_stopper,
            )

        result.checkpoint_saved = True
        result.checkpoint_path = str(final_checkpoint_path)

        # Save best model
        save_model_checkpoint(
            model_to_save,
            context.save_path,
            training_info={
                "epoch": metrics.epoch,
                "loss": float(early_stopper.best_loss),
                "early_stopped": True,
            },
            board_type=context.config.board_type,
            num_players=context.resolved.num_players,
        )
        logger.info(f"Best model saved to {context.save_path}")
        result.best_model_saved = True

    def _emit_early_stopped_event(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        early_stopper: Any,
    ) -> None:
        """Emit TRAINING_EARLY_STOPPED event.

        Args:
            context: Training context
            metrics: Epoch metrics
            early_stopper: Early stopping handler
        """
        if not HAS_TRAINING_EVENTS:
            return

        try:
            config_key = f"{context.config.board_type}_{context.resolved.num_players}p"
            best_elo = (
                early_stopper.best_elo
                if early_stopper.best_elo > float("-inf")
                else None
            )
            epochs_without_improvement = (
                early_stopper.counter
                if hasattr(early_stopper, "counter")
                else 0
            )

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(emit_training_early_stopped(
                    config_key=config_key,
                    epoch=metrics.epoch + 1,
                    best_loss=float(early_stopper.best_loss),
                    final_loss=float(metrics.avg_val_loss),
                    best_elo=best_elo,
                    reason="loss_stagnation",
                    epochs_without_improvement=epochs_without_improvement,
                ))
            except RuntimeError:
                asyncio.run(emit_training_early_stopped(
                    config_key=config_key,
                    epoch=metrics.epoch + 1,
                    best_loss=float(early_stopper.best_loss),
                    final_loss=float(metrics.avg_val_loss),
                    best_elo=best_elo,
                    reason="loss_stagnation",
                    epochs_without_improvement=epochs_without_improvement,
                ))

            logger.info(f"[train] Emitted TRAINING_EARLY_STOPPED for {config_key}")
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to emit TRAINING_EARLY_STOPPED: {e}")

    def _should_checkpoint(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> bool:
        """Check if checkpoint should be saved.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            True if checkpoint should be saved
        """
        if not context.is_main_process:
            return False

        if self.config.checkpoint_interval <= 0:
            return False

        return (metrics.epoch + 1) % self.config.checkpoint_interval == 0

    def _save_checkpoint(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> tuple[bool, str]:
        """Save checkpoint at interval.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            Tuple of (success, checkpoint_path)
        """
        model_to_save = context.model_to_save
        checkpoint_dir = Path(context.resolved.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{metrics.epoch + 1}.pth"

        if context.async_checkpointer is not None:
            context.async_checkpointer.save_async(
                model_to_save,
                context.optimizer,
                metrics.epoch,
                metrics.avg_val_loss,
                str(checkpoint_path),
                scheduler=context.epoch_scheduler,
                early_stopping=context.early_stopper,
            )
        else:
            save_checkpoint(
                model_to_save,
                context.optimizer,
                metrics.epoch,
                metrics.avg_val_loss,
                str(checkpoint_path),
                scheduler=context.epoch_scheduler,
                early_stopping=context.early_stopper,
            )

        # Track for circuit breaker rollback
        self._last_good_checkpoint_path = str(checkpoint_path)
        self._last_good_epoch = metrics.epoch

        return True, str(checkpoint_path)

    def _save_best_model(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> bool:
        """Save best model when validation loss improves.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            True if saved successfully
        """
        if not context.is_main_process:
            return False

        model_to_save = context.model_to_save
        train_size = len(context.train_loader.dataset) if hasattr(context.train_loader, "dataset") else 0

        # Save with versioning metadata
        save_model_checkpoint(
            model_to_save,
            context.save_path,
            training_info={
                "epoch": metrics.epoch + 1,
                "samples_seen": train_size * (metrics.epoch + 1),
                "val_loss": float(metrics.avg_val_loss),
                "train_loss": float(metrics.avg_train_loss),
            },
            board_type=context.config.board_type,
            num_players=context.resolved.num_players,
        )
        logger.info(f"  New best model saved (Val Loss: {metrics.avg_val_loss:.4f})")

        # Collect checkpoint for averaging
        if context.checkpoint_averager is not None:
            context.checkpoint_averager.add_checkpoint(
                model_to_save.state_dict(),
                epoch=metrics.epoch,
            )

        # Save timestamped checkpoint
        from datetime import datetime as dt
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        version_path = context.save_path.replace(".pth", f"_{timestamp}.pth")

        save_model_checkpoint(
            model_to_save,
            version_path,
            training_info={
                "epoch": metrics.epoch + 1,
                "samples_seen": train_size * (metrics.epoch + 1),
                "val_loss": float(metrics.avg_val_loss),
                "train_loss": float(metrics.avg_train_loss),
                "timestamp": timestamp,
            },
            board_type=context.config.board_type,
            num_players=context.resolved.num_players,
        )
        logger.info(f"  Versioned checkpoint saved: {version_path}")

        return True

    def _maybe_distill(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
    ) -> bool:
        """Check and perform knowledge distillation if needed.

        Args:
            context: Training context
            metrics: Epoch metrics

        Returns:
            True if distillation was triggered
        """
        if context.enhancements_manager is None:
            return False

        # Set checkpoint directory
        context.enhancements_manager.set_checkpoint_dir(
            context.resolved.checkpoint_dir
        )

        if not context.enhancements_manager.should_distill(metrics.epoch + 1):
            return False

        if context.is_main_process:
            logger.info(
                f"[Distillation] Triggering ensemble distillation at epoch {metrics.epoch + 1}"
            )
            distillation_success = context.enhancements_manager.run_distillation(
                current_epoch=metrics.epoch + 1,
                dataloader=context.train_loader,
            )
            if distillation_success:
                logger.info(
                    f"[Distillation] Epoch {metrics.epoch + 1}: Successfully distilled "
                    "ensemble knowledge into model"
                )
                return True

        return False

    def _apply_checkpoint_averaging(
        self,
        context: "TrainContext",
        metrics: EpochMetrics,
        model_to_save: nn.Module,
    ) -> None:
        """Apply checkpoint averaging at end of training.

        Args:
            context: Training context
            metrics: Epoch metrics
            model_to_save: Model to update with averaged weights
        """
        if context.checkpoint_averager is None:
            return

        if context.checkpoint_averager.num_stored < 2:
            logger.info(
                f"[Checkpoint Averaging] Skipped: only {context.checkpoint_averager.num_stored} "
                "checkpoint(s) available (need >= 2)"
            )
            context.checkpoint_averager.cleanup()
            return

        logger.info(
            f"[Checkpoint Averaging] Averaging {context.checkpoint_averager.num_stored} checkpoints..."
        )

        try:
            averaged_state_dict = context.checkpoint_averager.get_averaged_state_dict()

            # Save averaged model separately
            averaged_path = context.save_path.replace(".pth", "_averaged.pth")
            model_to_save.load_state_dict(averaged_state_dict)
            save_model_checkpoint(
                model_to_save,
                averaged_path,
                training_info={
                    "epoch": context.config.epochs,
                    "averaged_checkpoints": context.checkpoint_averager.num_stored,
                    "checkpoint_averaging": True,
                },
                board_type=context.config.board_type,
                num_players=context.resolved.num_players,
            )

            # Overwrite main save_path with averaged weights
            save_model_checkpoint(
                model_to_save,
                context.save_path,
                training_info={
                    "epoch": context.config.epochs,
                    "averaged_checkpoints": context.checkpoint_averager.num_stored,
                    "checkpoint_averaging": True,
                },
                board_type=context.config.board_type,
                num_players=context.resolved.num_players,
            )

            logger.info(
                f"[Checkpoint Averaging] Saved averaged model "
                f"({context.checkpoint_averager.num_stored} checkpoints) to {context.save_path}"
            )
        except (OSError, RuntimeError, ValueError, TypeError, MemoryError) as e:
            logger.warning(f"[Checkpoint Averaging] Failed to average checkpoints: {e}")
        finally:
            context.checkpoint_averager.cleanup()


# =============================================================================
# Factory Functions
# =============================================================================


def create_post_epoch_handler(
    checkpoint_interval: int = 5,
    checkpoint_dir: str = "checkpoints",
    emit_events: bool = True,
    **kwargs: Any,
) -> PostEpochHandler:
    """Create a PostEpochHandler with the specified settings.

    Args:
        checkpoint_interval: Epochs between checkpoints
        checkpoint_dir: Directory for checkpoints
        emit_events: Enable event emission
        **kwargs: Additional config parameters

    Returns:
        Configured PostEpochHandler instance
    """
    config = PostEpochConfig(
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        emit_events=emit_events,
        **{k: v for k, v in kwargs.items() if hasattr(PostEpochConfig, k)},
    )
    return PostEpochHandler(config)
