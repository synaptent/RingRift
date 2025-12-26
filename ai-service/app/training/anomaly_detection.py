"""Training Anomaly Detection Module.

Provides anomaly detection for training processes:
- AnomalyEvent: Record of detected anomalies
- TrainingAnomalyDetector: Real-time anomaly detection
- TrainingLossAnomalyHandler: Event-driven anomaly response

Extracted from training_enhancements.py (December 2025).

Usage:
    from app.training.anomaly_detection import (
        TrainingAnomalyDetector,
        TrainingLossAnomalyHandler,
        wire_training_loss_anomaly_handler,
    )

    # Create detector
    detector = TrainingAnomalyDetector(loss_spike_threshold=3.0)

    # Check loss during training
    if detector.check_loss(loss.item(), step):
        continue  # Skip anomalous batch

    # Check gradient norm
    if detector.check_gradient_norm(grad_norm, step):
        optimizer.zero_grad()
        continue
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Record of a training anomaly event.

    Attributes:
        timestamp: Unix timestamp when anomaly was detected.
        step: Training step when anomaly occurred.
        anomaly_type: Type of anomaly (nan, inf, loss_spike, gradient_explosion).
        value: The anomalous value that triggered detection.
        threshold: The threshold that was exceeded.
        message: Human-readable description.
    """

    timestamp: float
    step: int
    anomaly_type: str  # "nan", "inf", "loss_spike", "gradient_explosion"
    value: float
    threshold: float
    message: str


class TrainingAnomalyDetector:
    """Detects and handles training anomalies in real-time.

    Monitors for:
    - NaN/Inf in loss or gradients
    - Loss spikes (sudden large increases)
    - Gradient explosions (norm exceeds threshold)

    Features:
    - Rolling window for spike detection
    - Configurable thresholds
    - Event logging for post-analysis
    - Automatic halt option

    Usage:
        detector = TrainingAnomalyDetector(loss_spike_threshold=3.0)

        for batch in dataloader:
            loss = model(batch)

            # Check for anomalies before backward
            if detector.check_loss(loss.item(), step):
                continue

            loss.backward()

            # Check gradients
            grad_norm = compute_grad_norm(model)
            if detector.check_gradient_norm(grad_norm, step):
                optimizer.zero_grad()
                continue
    """

    def __init__(
        self,
        loss_spike_threshold: float = 4.0,
        gradient_norm_threshold: float = 100.0,
        loss_window_size: int = 100,
        halt_on_nan: bool = True,
        halt_on_spike: bool = False,
        halt_on_gradient_explosion: bool = False,
        max_consecutive_anomalies: int = 20,
    ):
        """Initialize the anomaly detector.

        Args:
            loss_spike_threshold: Standard deviations above mean to trigger spike.
            gradient_norm_threshold: Max gradient norm before explosion detection.
            loss_window_size: Rolling window size for loss statistics.
            halt_on_nan: Raise exception on NaN/Inf detection.
            halt_on_spike: Raise exception on loss spike.
            halt_on_gradient_explosion: Raise exception on gradient explosion.
            max_consecutive_anomalies: Max consecutive anomalies before halt.
        """
        self.loss_spike_threshold = loss_spike_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.loss_window_size = loss_window_size
        self.halt_on_nan = halt_on_nan
        self.halt_on_spike = halt_on_spike
        self.halt_on_gradient_explosion = halt_on_gradient_explosion
        self.max_consecutive_anomalies = max_consecutive_anomalies

        self._loss_history: deque = deque(maxlen=loss_window_size)
        self._events: list[AnomalyEvent] = []
        self._consecutive_anomalies = 0
        self._total_anomalies = 0
        self._halted = False

    def check_loss(self, loss: float, step: int) -> bool:
        """Check loss value for anomalies.

        Args:
            loss: Current loss value.
            step: Current training step.

        Returns:
            True if anomaly detected, False otherwise.

        Raises:
            RuntimeError: If halt_on_nan/halt_on_spike is True and anomaly detected.
        """
        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            event = AnomalyEvent(
                timestamp=time.time(),
                step=step,
                anomaly_type="nan" if math.isnan(loss) else "inf",
                value=loss,
                threshold=0.0,
                message=f"Loss is {'NaN' if math.isnan(loss) else 'Inf'} at step {step}",
            )
            self._record_anomaly(event)

            if self.halt_on_nan:
                raise RuntimeError(event.message)
            return True

        # Check for loss spike
        if len(self._loss_history) >= 10:
            mean_loss = np.mean(list(self._loss_history))
            std_loss = np.std(list(self._loss_history))

            if std_loss > 0 and (loss - mean_loss) > self.loss_spike_threshold * std_loss:
                event = AnomalyEvent(
                    timestamp=time.time(),
                    step=step,
                    anomaly_type="loss_spike",
                    value=loss,
                    threshold=mean_loss + self.loss_spike_threshold * std_loss,
                    message=f"Loss spike at step {step}: {loss:.4f} (mean: {mean_loss:.4f}, threshold: {self.loss_spike_threshold}σ)",
                )
                self._record_anomaly(event)

                if self.halt_on_spike:
                    raise RuntimeError(event.message)
                return True

        # Record valid loss
        self._loss_history.append(loss)
        self._consecutive_anomalies = 0
        return False

    def check_gradient_norm(self, grad_norm: float, step: int) -> bool:
        """Check gradient norm for explosion.

        Args:
            grad_norm: Current gradient norm.
            step: Current training step.

        Returns:
            True if anomaly detected, False otherwise.

        Raises:
            RuntimeError: If halt_on_gradient_explosion is True and anomaly detected.
        """
        # Check for NaN/Inf
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            event = AnomalyEvent(
                timestamp=time.time(),
                step=step,
                anomaly_type="nan" if math.isnan(grad_norm) else "inf",
                value=grad_norm,
                threshold=0.0,
                message=f"Gradient norm is {'NaN' if math.isnan(grad_norm) else 'Inf'} at step {step}",
            )
            self._record_anomaly(event)

            if self.halt_on_nan:
                raise RuntimeError(event.message)
            return True

        # Check for explosion
        if grad_norm > self.gradient_norm_threshold:
            event = AnomalyEvent(
                timestamp=time.time(),
                step=step,
                anomaly_type="gradient_explosion",
                value=grad_norm,
                threshold=self.gradient_norm_threshold,
                message=f"Gradient explosion at step {step}: norm={grad_norm:.4f} > threshold={self.gradient_norm_threshold}",
            )
            self._record_anomaly(event)

            if self.halt_on_gradient_explosion:
                raise RuntimeError(event.message)
            return True

        return False

    def _record_anomaly(self, event: AnomalyEvent) -> None:
        """Record an anomaly event."""
        self._events.append(event)
        self._consecutive_anomalies += 1
        self._total_anomalies += 1

        logger.warning(f"[AnomalyDetector] {event.message}")

        # Check for too many consecutive anomalies
        if self._consecutive_anomalies >= self.max_consecutive_anomalies:
            self._halted = True
            raise RuntimeError(
                f"Training halted: {self._consecutive_anomalies} consecutive anomalies detected"
            )

    def reset(self) -> None:
        """Reset detector state (e.g., for new training run)."""
        self._loss_history.clear()
        self._events.clear()
        self._consecutive_anomalies = 0
        self._total_anomalies = 0
        self._halted = False

    @property
    def is_halted(self) -> bool:
        """Check if training should be halted."""
        return self._halted

    @property
    def total_anomalies(self) -> int:
        """Get total number of anomalies detected."""
        return self._total_anomalies

    def get_events(self) -> list[AnomalyEvent]:
        """Get all recorded anomaly events."""
        return self._events.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of detected anomalies."""
        type_counts: dict[str, int] = {}
        for event in self._events:
            type_counts[event.anomaly_type] = type_counts.get(event.anomaly_type, 0) + 1

        return {
            "total_anomalies": self._total_anomalies,
            "consecutive_anomalies": self._consecutive_anomalies,
            "halted": self._halted,
            "anomaly_types": type_counts,
            "recent_events": [
                {
                    "step": e.step,
                    "type": e.anomaly_type,
                    "value": e.value,
                    "message": e.message,
                }
                for e in self._events[-10:]  # Last 10 events
            ],
        }


class TrainingLossAnomalyHandler:
    """Handles TRAINING_LOSS_ANOMALY events by triggering quality checks.

    When training loss spikes are detected, this handler:
    1. Logs the anomaly for investigation
    2. Emits a LOW_QUALITY_DATA_WARNING to trigger quality-aware responses
    3. Optionally pauses or reduces training rate

    This closes the feedback loop: training loss problems -> data quality investigation.
    """

    def __init__(
        self,
        config_key: str,
        anomaly_threshold: float = 2.0,
        cooldown_seconds: float = 300.0,
    ):
        """Initialize the handler.

        Args:
            config_key: Board configuration (e.g., "hex8_2p").
            anomaly_threshold: Factor above average loss to consider anomaly.
            cooldown_seconds: Minimum time between quality check triggers.
        """
        self.config_key = config_key
        self.anomaly_threshold = anomaly_threshold
        self.cooldown_seconds = cooldown_seconds

        self._subscribed = False
        self._last_trigger_time = 0.0
        self._anomaly_count = 0

    def subscribe(self) -> bool:
        """Subscribe to TRAINING_LOSS_ANOMALY events.

        Returns:
            True if subscription was successful.
        """
        if self._subscribed:
            return True

        try:
            from app.distributed.data_events import DataEventType

            try:
                from app.coordination.event_router import subscribe
            except ImportError:
                from app.distributed.data_events import get_event_bus

                def subscribe(event_type, callback):
                    get_event_bus().subscribe(event_type, callback)

            def on_training_loss_anomaly(event):
                """Handle TRAINING_LOSS_ANOMALY event."""
                payload = event.payload
                event_config = payload.get("config_key", "")

                # Only handle events for our config
                if event_config and event_config != self.config_key:
                    return

                self._handle_anomaly(payload)

            subscribe(DataEventType.TRAINING_LOSS_ANOMALY, on_training_loss_anomaly)
            self._subscribed = True
            logger.info(
                f"[TrainingLossAnomalyHandler] Subscribed to TRAINING_LOSS_ANOMALY "
                f"for {self.config_key}"
            )
            return True

        except ImportError as e:
            logger.debug(f"[TrainingLossAnomalyHandler] Event system not available: {e}")
            return False

    def _handle_anomaly(self, payload: dict[str, Any]) -> None:
        """Handle a training loss anomaly.

        Args:
            payload: Event payload with loss details.
        """
        current_time = time.time()
        loss = payload.get("loss", 0.0)
        average_loss = payload.get("average_loss", 0.0)
        epoch = payload.get("epoch", 0)

        self._anomaly_count += 1

        # Check cooldown
        if current_time - self._last_trigger_time < self.cooldown_seconds:
            logger.debug(
                f"[TrainingLossAnomalyHandler] Anomaly #{self._anomaly_count} for "
                f"{self.config_key} (in cooldown, skipping quality check)"
            )
            return

        # Trigger quality warning
        self._last_trigger_time = current_time

        logger.warning(
            f"[TrainingLossAnomalyHandler] Loss anomaly #{self._anomaly_count} for "
            f"{self.config_key}: loss={loss:.4f}, avg={average_loss:.4f}, "
            f"epoch={epoch}. Triggering quality check."
        )

        # Emit LOW_QUALITY_DATA_WARNING to trigger quality investigation
        try:
            from app.core.async_context import fire_and_forget
            from app.distributed.data_events import DataEventType, get_event_bus

            bus = get_event_bus()

            async def emit_warning():
                await bus.publish(
                    DataEventType.LOW_QUALITY_DATA_WARNING,
                    {
                        "config_key": self.config_key,
                        "reason": "training_loss_anomaly",
                        "loss": loss,
                        "average_loss": average_loss,
                        "anomaly_count": self._anomaly_count,
                    },
                )

            fire_and_forget(emit_warning())

        except Exception as e:
            logger.debug(f"[TrainingLossAnomalyHandler] Failed to emit warning: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "config_key": self.config_key,
            "subscribed": self._subscribed,
            "anomaly_count": self._anomaly_count,
            "last_trigger_time": self._last_trigger_time,
        }


def wire_training_loss_anomaly_handler(config_key: str) -> TrainingLossAnomalyHandler:
    """Wire TRAINING_LOSS_ANOMALY events to quality check triggers.

    Args:
        config_key: Board configuration (e.g., "hex8_2p").

    Returns:
        Subscribed handler instance.
    """
    handler = TrainingLossAnomalyHandler(config_key)
    handler.subscribe()
    return handler
