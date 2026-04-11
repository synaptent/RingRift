"""Loss monitoring helpers for training runs."""

from __future__ import annotations

import logging
from typing import Any

try:
    from app.training.train_events import (
        emit_training_loss_anomaly,
        emit_training_loss_trend,
    )

    HAS_TRAINING_EVENTS = True
except ImportError:
    emit_training_loss_anomaly = None
    emit_training_loss_trend = None
    HAS_TRAINING_EVENTS = False


class LossMonitor:
    """Track loss curves and detect learning stalls during training."""

    def __init__(self, patience: int = 5, config_key: str = "unknown"):
        self.patience = patience
        self.config_key = config_key
        self.history: list[dict[str, float]] = []
        self.best_loss = float("inf")
        self.stale_epochs = 0
        self._logger = logging.getLogger(__name__)

    def record(self, epoch: int, train_loss: float, val_loss: float) -> bool:
        """Record epoch losses and return whether training should continue."""
        self.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < self.best_loss * 0.99:
            self.best_loss = val_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1

        if self.stale_epochs >= self.patience:
            self._logger.warning(
                "[LossMonitor] Loss not decreasing for %s epochs! Best: %.4f, Current: %.4f",
                self.patience,
                self.best_loss,
                val_loss,
            )
            if HAS_TRAINING_EVENTS and emit_training_loss_anomaly is not None:
                try:
                    emit_training_loss_anomaly(
                        config_key=self.config_key,
                        anomaly_type="learning_stall",
                        epochs_stale=self.stale_epochs,
                        best_loss=self.best_loss,
                        current_loss=val_loss,
                    )
                except (RuntimeError, TypeError) as exc:
                    self._logger.debug("Failed to emit anomaly event: %s", exc)
            return False

        if epoch % 5 == 0 and HAS_TRAINING_EVENTS and emit_training_loss_trend is not None:
            try:
                emit_training_loss_trend(
                    config_key=self.config_key,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
            except (RuntimeError, TypeError):
                pass

        return True

    def get_summary(self) -> dict[str, Any]:
        """Return a compact summary of monitored losses."""
        return {
            "config_key": self.config_key,
            "epochs_recorded": len(self.history),
            "best_loss": self.best_loss,
            "stale_epochs": self.stale_epochs,
            "is_stalled": self.stale_epochs >= self.patience,
        }


__all__ = ["LossMonitor"]
