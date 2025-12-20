"""Metrics Integration for Training Components.

Provides pre-defined training metrics using the unified MetricsPublisher.
This module standardizes metric names and labels across training components.

Usage:
    from app.training.metrics_integration import (
        TrainingMetrics,
        publish_training_step,
        publish_eval_result,
        time_training_epoch,
    )

    # Publish a training step
    TrainingMetrics.step(
        config_key="square8_2p",
        step=1000,
        loss=0.01,
        learning_rate=0.001,
    )

    # Time an epoch
    with TrainingMetrics.epoch_timer(config_key="square8_2p", epoch=5):
        train_epoch(...)

    # Publish evaluation result
    TrainingMetrics.evaluation(
        config_key="square8_2p",
        elo=1650,
        win_rate=0.65,
    )
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from app.metrics.unified_publisher import (
    MetricTimer,
    publish_counter,
    publish_gauge,
    publish_histogram,
    time_operation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TrainingMetricNames",
    "TrainingMetrics",
    "get_training_metrics",
    "publish_checkpoint_saved",
    "publish_epoch_completed",
    "publish_eval_result",
    "publish_selfplay_completed",
    "publish_training_step",
    "time_training_epoch",
]


# =============================================================================
# Metric Name Constants
# =============================================================================

class TrainingMetricNames:
    """Standardized metric names for training."""

    # Counters
    STEPS_TOTAL = "training_steps_total"
    EPOCHS_TOTAL = "training_epochs_total"
    SAMPLES_TOTAL = "training_samples_total"
    GAMES_GENERATED = "selfplay_games_total"
    CHECKPOINTS_SAVED = "checkpoints_saved_total"
    EVALUATIONS_RUN = "evaluations_total"
    MODEL_PROMOTIONS = "model_promotions_total"

    # Gauges
    CURRENT_LOSS = "training_loss"
    CURRENT_LEARNING_RATE = "training_learning_rate"
    CURRENT_ELO = "model_elo"
    BEST_ELO = "model_best_elo"
    WIN_RATE = "eval_win_rate"
    EPOCH_PROGRESS = "training_epoch_progress"
    BUFFER_SIZE = "data_buffer_size"
    ACTIVE_TRAINING_JOBS = "active_training_jobs"
    GAMES_IN_BUFFER = "games_in_buffer"

    # Histograms
    EPOCH_DURATION = "training_epoch_duration_seconds"
    STEP_DURATION = "training_step_duration_seconds"
    EVAL_DURATION = "evaluation_duration_seconds"
    CHECKPOINT_DURATION = "checkpoint_duration_seconds"
    SELFPLAY_DURATION = "selfplay_duration_seconds"
    INFERENCE_LATENCY = "inference_latency_seconds"
    DATA_LOAD_LATENCY = "data_load_latency_seconds"


# =============================================================================
# Training Metrics Class
# =============================================================================

class TrainingMetrics:
    """Static class for publishing training metrics.

    Provides a clean interface for publishing standardized training metrics.
    All methods are static for easy use without initialization.
    """

    # -------------------------------------------------------------------------
    # Training Loop Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def step(
        config_key: str,
        step: int,
        loss: float,
        learning_rate: float = 0.0,
        samples_per_second: float = 0.0,
        **extra_labels,
    ) -> None:
        """Publish metrics for a training step.

        Args:
            config_key: Configuration key
            step: Step number
            loss: Training loss
            learning_rate: Current learning rate
            samples_per_second: Training throughput
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        # Counter: steps completed
        publish_counter(TrainingMetricNames.STEPS_TOTAL, 1, **labels)

        # Gauges: current state
        publish_gauge(TrainingMetricNames.CURRENT_LOSS, loss, **labels)
        if learning_rate > 0:
            publish_gauge(TrainingMetricNames.CURRENT_LEARNING_RATE, learning_rate, **labels)

        # Counter: samples processed
        if samples_per_second > 0:
            publish_counter(TrainingMetricNames.SAMPLES_TOTAL, samples_per_second, **labels)

    @staticmethod
    def epoch(
        config_key: str,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float | None = None,
        duration_seconds: float = 0.0,
        **extra_labels,
    ) -> None:
        """Publish metrics for epoch completion.

        Args:
            config_key: Configuration key
            epoch: Epoch number
            total_epochs: Total epochs planned
            train_loss: Training loss for epoch
            val_loss: Validation loss if available
            duration_seconds: Epoch duration
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        # Counter: epochs completed
        publish_counter(TrainingMetricNames.EPOCHS_TOTAL, 1, **labels)

        # Gauge: progress
        progress = epoch / max(total_epochs, 1)
        publish_gauge(TrainingMetricNames.EPOCH_PROGRESS, progress, **labels)

        # Gauge: losses
        publish_gauge(TrainingMetricNames.CURRENT_LOSS, train_loss, **labels)
        if val_loss is not None:
            publish_gauge(f"{TrainingMetricNames.CURRENT_LOSS}_val", val_loss, **labels)

        # Histogram: duration
        if duration_seconds > 0:
            publish_histogram(TrainingMetricNames.EPOCH_DURATION, duration_seconds, **labels)

    @staticmethod
    def epoch_timer(config_key: str, epoch: int, **extra_labels) -> MetricTimer:
        """Get a timer for measuring epoch duration.

        Args:
            config_key: Configuration key
            epoch: Epoch number
            **extra_labels: Additional labels

        Returns:
            MetricTimer context manager

        Example:
            with TrainingMetrics.epoch_timer("square8_2p", epoch=5):
                train_epoch()
        """
        labels = {"config": config_key, "epoch": str(epoch), **extra_labels}
        return time_operation(TrainingMetricNames.EPOCH_DURATION, **labels)

    @staticmethod
    def step_timer(config_key: str, **extra_labels) -> MetricTimer:
        """Get a timer for measuring step duration.

        Args:
            config_key: Configuration key
            **extra_labels: Additional labels

        Returns:
            MetricTimer context manager
        """
        labels = {"config": config_key, **extra_labels}
        return time_operation(TrainingMetricNames.STEP_DURATION, **labels)

    # -------------------------------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def evaluation(
        config_key: str,
        elo: float,
        win_rate: float,
        games_played: int = 0,
        passes_gating: bool = True,
        duration_seconds: float = 0.0,
        **extra_labels,
    ) -> None:
        """Publish evaluation metrics.

        Args:
            config_key: Configuration key
            elo: Current Elo estimate
            win_rate: Win rate against baselines
            games_played: Number of games played
            passes_gating: Whether baseline gating passed
            duration_seconds: Evaluation duration
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        # Counter: evaluations
        publish_counter(TrainingMetricNames.EVALUATIONS_RUN, 1, **labels)

        # Gauges: Elo and win rate
        publish_gauge(TrainingMetricNames.CURRENT_ELO, elo, **labels)
        publish_gauge(TrainingMetricNames.WIN_RATE, win_rate, **labels)

        # Histogram: duration
        if duration_seconds > 0:
            publish_histogram(TrainingMetricNames.EVAL_DURATION, duration_seconds, **labels)

        # Counter: games played
        if games_played > 0:
            publish_counter("eval_games_total", games_played, **labels)

        # Gating status
        publish_gauge("eval_gating_passed", 1.0 if passes_gating else 0.0, **labels)

    @staticmethod
    def elo_update(
        config_key: str,
        current_elo: float,
        best_elo: float,
        **extra_labels,
    ) -> None:
        """Publish Elo update metrics.

        Args:
            config_key: Configuration key
            current_elo: Current Elo
            best_elo: Best Elo achieved
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        publish_gauge(TrainingMetricNames.CURRENT_ELO, current_elo, **labels)
        publish_gauge(TrainingMetricNames.BEST_ELO, best_elo, **labels)

    @staticmethod
    def eval_timer(config_key: str, **extra_labels) -> MetricTimer:
        """Get a timer for measuring evaluation duration."""
        labels = {"config": config_key, **extra_labels}
        return time_operation(TrainingMetricNames.EVAL_DURATION, **labels)

    # -------------------------------------------------------------------------
    # Checkpoint Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def checkpoint(
        config_key: str,
        step: int,
        is_best: bool = False,
        duration_seconds: float = 0.0,
        **extra_labels,
    ) -> None:
        """Publish checkpoint save metrics.

        Args:
            config_key: Configuration key
            step: Training step
            is_best: Whether this is a new best checkpoint
            duration_seconds: Save duration
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        # Counter: checkpoints saved
        publish_counter(TrainingMetricNames.CHECKPOINTS_SAVED, 1, **labels)

        if is_best:
            publish_counter("checkpoints_best_total", 1, **labels)

        # Histogram: duration
        if duration_seconds > 0:
            publish_histogram(TrainingMetricNames.CHECKPOINT_DURATION, duration_seconds, **labels)

    @staticmethod
    def checkpoint_timer(config_key: str, **extra_labels) -> MetricTimer:
        """Get a timer for measuring checkpoint duration."""
        labels = {"config": config_key, **extra_labels}
        return time_operation(TrainingMetricNames.CHECKPOINT_DURATION, **labels)

    # -------------------------------------------------------------------------
    # Selfplay Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def selfplay(
        config_key: str,
        games_generated: int,
        iteration: int = 0,
        duration_seconds: float = 0.0,
        **extra_labels,
    ) -> None:
        """Publish selfplay metrics.

        Args:
            config_key: Configuration key
            games_generated: Number of games generated
            iteration: Training iteration
            duration_seconds: Selfplay duration
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        # Counter: games generated
        publish_counter(TrainingMetricNames.GAMES_GENERATED, games_generated, **labels)

        # Histogram: duration
        if duration_seconds > 0:
            publish_histogram(TrainingMetricNames.SELFPLAY_DURATION, duration_seconds, **labels)

        # Gauge: games per second
        if duration_seconds > 0:
            games_per_sec = games_generated / duration_seconds
            publish_gauge("selfplay_games_per_second", games_per_sec, **labels)

    @staticmethod
    def selfplay_timer(config_key: str, **extra_labels) -> MetricTimer:
        """Get a timer for measuring selfplay duration."""
        labels = {"config": config_key, **extra_labels}
        return time_operation(TrainingMetricNames.SELFPLAY_DURATION, **labels)

    # -------------------------------------------------------------------------
    # Data Loading Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def data_load(
        config_key: str,
        batch_size: int,
        duration_seconds: float = 0.0,
        **extra_labels,
    ) -> None:
        """Publish data loading metrics.

        Args:
            config_key: Configuration key
            batch_size: Batch size loaded
            duration_seconds: Load duration
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        # Counter: samples loaded
        publish_counter(TrainingMetricNames.SAMPLES_TOTAL, batch_size, **labels)

        # Histogram: load duration
        if duration_seconds > 0:
            publish_histogram(TrainingMetricNames.DATA_LOAD_LATENCY, duration_seconds, **labels)

    @staticmethod
    def buffer_status(
        config_key: str,
        buffer_size: int,
        games_count: int = 0,
        **extra_labels,
    ) -> None:
        """Publish data buffer status metrics.

        Args:
            config_key: Configuration key
            buffer_size: Current buffer size
            games_count: Number of games in buffer
            **extra_labels: Additional labels
        """
        labels = {"config": config_key, **extra_labels}

        publish_gauge(TrainingMetricNames.BUFFER_SIZE, buffer_size, **labels)
        if games_count > 0:
            publish_gauge(TrainingMetricNames.GAMES_IN_BUFFER, games_count, **labels)

    @staticmethod
    def data_load_timer(config_key: str, **extra_labels) -> MetricTimer:
        """Get a timer for measuring data load duration."""
        labels = {"config": config_key, **extra_labels}
        return time_operation(TrainingMetricNames.DATA_LOAD_LATENCY, **labels)

    # -------------------------------------------------------------------------
    # Model Promotion Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def promotion(
        config_key: str,
        from_state: str,
        to_state: str,
        **extra_labels,
    ) -> None:
        """Publish model promotion metrics.

        Args:
            config_key: Configuration key
            from_state: Previous state
            to_state: New state
            **extra_labels: Additional labels
        """
        labels = {
            "config": config_key,
            "from_state": from_state,
            "to_state": to_state,
            **extra_labels,
        }

        publish_counter(TrainingMetricNames.MODEL_PROMOTIONS, 1, **labels)

    # -------------------------------------------------------------------------
    # System Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def active_jobs(count: int, **extra_labels) -> None:
        """Publish active training jobs count.

        Args:
            count: Number of active jobs
            **extra_labels: Additional labels
        """
        publish_gauge(TrainingMetricNames.ACTIVE_TRAINING_JOBS, count, **extra_labels)

    @staticmethod
    def inference_latency(latency_seconds: float, model_name: str = "", **extra_labels) -> None:
        """Publish inference latency.

        Args:
            latency_seconds: Inference latency in seconds
            model_name: Model name
            **extra_labels: Additional labels
        """
        labels = extra_labels.copy()
        if model_name:
            labels["model"] = model_name
        publish_histogram(TrainingMetricNames.INFERENCE_LATENCY, latency_seconds, **labels)

    @staticmethod
    def inference_timer(model_name: str = "", **extra_labels) -> MetricTimer:
        """Get a timer for measuring inference latency."""
        labels = extra_labels.copy()
        if model_name:
            labels["model"] = model_name
        return time_operation(TrainingMetricNames.INFERENCE_LATENCY, **labels)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_training_metrics() -> type[TrainingMetrics]:
    """Get the TrainingMetrics class for static method access."""
    return TrainingMetrics


def publish_training_step(
    config_key: str,
    step: int,
    loss: float,
    **kwargs,
) -> None:
    """Convenience function to publish training step metrics."""
    TrainingMetrics.step(config_key, step, loss, **kwargs)


def publish_epoch_completed(
    config_key: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    **kwargs,
) -> None:
    """Convenience function to publish epoch completion metrics."""
    TrainingMetrics.epoch(config_key, epoch, total_epochs, train_loss, **kwargs)


def publish_eval_result(
    config_key: str,
    elo: float,
    win_rate: float,
    **kwargs,
) -> None:
    """Convenience function to publish evaluation metrics."""
    TrainingMetrics.evaluation(config_key, elo, win_rate, **kwargs)


def publish_checkpoint_saved(
    config_key: str,
    step: int,
    **kwargs,
) -> None:
    """Convenience function to publish checkpoint metrics."""
    TrainingMetrics.checkpoint(config_key, step, **kwargs)


def publish_selfplay_completed(
    config_key: str,
    games_generated: int,
    **kwargs,
) -> None:
    """Convenience function to publish selfplay metrics."""
    TrainingMetrics.selfplay(config_key, games_generated, **kwargs)


@contextmanager
def time_training_epoch(config_key: str, epoch: int, **extra_labels):
    """Context manager to time a training epoch and publish metrics.

    Args:
        config_key: Configuration key
        epoch: Epoch number
        **extra_labels: Additional labels

    Yields:
        Start time for manual calculations

    Example:
        with time_training_epoch("square8_2p", epoch=5) as start:
            train_loss = train_epoch()
            # Duration automatically published on exit
    """
    start_time = time.time()
    try:
        yield start_time
    finally:
        duration = time.time() - start_time
        labels = {"config": config_key, "epoch": str(epoch), **extra_labels}
        publish_histogram(TrainingMetricNames.EPOCH_DURATION, duration, **labels)
