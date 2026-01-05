"""Validation Scheduling Module.

Provides validation interval management and early stopping:
- ValidationResult: Result dataclass from validation runs
- ValidationIntervalManager: Configurable validation during training
- EnhancedEarlyStopping: Multi-criteria early stopping with plateau detection

Extracted from training_enhancements.py (December 2025).

Usage:
    from app.training.validation_scheduling import (
        ValidationIntervalManager,
        EnhancedEarlyStopping,
        EarlyStopping,  # Backwards compatible alias
    )

    # Create validation manager
    val_manager = ValidationIntervalManager(
        validation_fn=lambda model: validate(model, val_loader),
        interval_steps=1000,
    )

    # Create early stopping
    early_stopping = EnhancedEarlyStopping(patience=10)

    for epoch in range(epochs):
        # Train...

        if early_stopping.should_stop(val_loss=val_loss, current_elo=elo):
            early_stopping.restore_best_model(model)
            break
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a validation run.

    Attributes:
        step: Training step when validation occurred.
        epoch: Training epoch when validation occurred.
        val_loss: Validation loss value.
        val_metrics: Dictionary of additional validation metrics.
        samples_validated: Number of samples used in validation.
        duration_seconds: Time taken for validation.
        is_improvement: Whether this was a new best result.
    """

    step: int
    epoch: int
    val_loss: float
    val_metrics: dict[str, float]
    samples_validated: int
    duration_seconds: float
    is_improvement: bool = False


class ValidationIntervalManager:
    """Manages configurable validation during training.

    Instead of validating only at epoch boundaries, this allows:
    - Validation every N steps
    - Validation on a subset of data for speed
    - Adaptive validation frequency based on loss trends
    - Early detection of overfitting

    Usage:
        val_manager = ValidationIntervalManager(
            validation_fn=lambda model: validate(model, val_loader),
            interval_steps=1000,
            subset_size=0.1,
        )

        for step, batch in enumerate(train_loader):
            # Train step...

            # Check if validation is due
            if val_manager.should_validate(step, epoch):
                result = val_manager.validate(model, step, epoch)
                if result.is_improvement:
                    save_checkpoint(model)
    """

    def __init__(
        self,
        validation_fn: Callable[[nn.Module], tuple[float, dict[str, float]]] | None = None,
        interval_steps: int = 1000,
        interval_epochs: float | None = None,
        subset_size: float = 1.0,
        adaptive_interval: bool = False,
        min_interval_steps: int = 100,
        max_interval_steps: int = 10000,
        warmup_steps: int = 0,
        track_best: bool = True,
    ):
        """Initialize the validation interval manager.

        Args:
            validation_fn: Function that takes model and returns (loss, metrics_dict).
            interval_steps: Validate every N training steps.
            interval_epochs: Alternative: validate every N epochs (overrides interval_steps).
            subset_size: Fraction of validation data to use (0-1).
            adaptive_interval: Adjust interval based on loss variance.
            min_interval_steps: Minimum interval when adaptive.
            max_interval_steps: Maximum interval when adaptive.
            warmup_steps: Skip validation for first N steps.
            track_best: Track and report best validation loss.
        """
        self.validation_fn = validation_fn
        self.interval_steps = interval_steps
        self.interval_epochs = interval_epochs
        self.subset_size = min(1.0, max(0.01, subset_size))
        self.adaptive_interval = adaptive_interval
        self.min_interval_steps = min_interval_steps
        self.max_interval_steps = max_interval_steps
        self.warmup_steps = warmup_steps
        self.track_best = track_best

        self._last_val_step = -interval_steps  # Allow immediate first validation
        self._last_val_epoch = -1.0
        self._current_interval = interval_steps
        self._results: list[ValidationResult] = []
        self._best_loss = float("inf")
        self._best_step = 0
        self._loss_history: deque = deque(maxlen=10)

    def should_validate(self, step: int, epoch: int = 0, epoch_fraction: float = 0.0) -> bool:
        """Check if validation should be performed.

        Args:
            step: Current training step.
            epoch: Current epoch number.
            epoch_fraction: Fraction through current epoch (0-1).

        Returns:
            True if validation should be performed.
        """
        # Skip during warmup
        if step < self.warmup_steps:
            return False

        # Epoch-based interval
        if self.interval_epochs is not None:
            current_epoch_float = epoch + epoch_fraction
            return current_epoch_float - self._last_val_epoch >= self.interval_epochs

        # Step-based interval
        return (step - self._last_val_step) >= self._current_interval

    def validate(
        self,
        model: nn.Module,
        step: int,
        epoch: int = 0,
        val_loader: torch.utils.data.DataLoader | None = None,
        device: torch.device | None = None,
    ) -> ValidationResult:
        """Perform validation and record results.

        Args:
            model: Model to validate.
            step: Current training step.
            epoch: Current epoch.
            val_loader: Optional validation dataloader (if validation_fn not set).
            device: Device for validation.

        Returns:
            ValidationResult with loss and metrics.
        """
        start_time = time.time()

        if self.validation_fn is not None:
            # Use provided validation function
            val_loss, val_metrics = self.validation_fn(model)
            samples = int(val_metrics.get("samples", 0))
        elif val_loader is not None:
            # Default validation loop
            val_loss, val_metrics, samples = self._default_validate(model, val_loader, device)
        else:
            raise ValueError("Either validation_fn or val_loader must be provided")

        duration = time.time() - start_time

        # Check for improvement
        is_improvement = val_loss < self._best_loss
        if is_improvement and self.track_best:
            self._best_loss = val_loss
            self._best_step = step

        # Record result
        result = ValidationResult(
            step=step,
            epoch=epoch,
            val_loss=val_loss,
            val_metrics=val_metrics,
            samples_validated=samples,
            duration_seconds=duration,
            is_improvement=is_improvement,
        )
        self._results.append(result)

        # Update tracking
        self._last_val_step = step
        self._last_val_epoch = epoch
        self._loss_history.append(val_loss)

        # Adapt interval if enabled
        if self.adaptive_interval and len(self._loss_history) >= 3:
            self._adapt_interval()

        logger.info(
            f"Validation at step {step}: loss={val_loss:.4f} "
            f"({'new best' if is_improvement else f'best={self._best_loss:.4f}'}) "
            f"[{duration:.1f}s, {samples} samples]"
        )

        return result

    def _default_validate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
    ) -> tuple[float, dict[str, float], int]:
        """Default validation loop."""
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        total_loss = 0.0
        total_samples = 0
        metrics: dict[str, float] = {}
        batches_used = 0

        # Determine subset size
        total_batches = len(val_loader)
        batches_to_use = max(1, int(total_batches * self.subset_size))

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= batches_to_use:
                    break

                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device) if len(batch) > 1 else None
                else:
                    inputs = batch.to(device)
                    targets = None

                outputs = model(inputs)

                # Compute loss if targets available
                if targets is not None:
                    if isinstance(outputs, tuple):
                        # Assume (policy, value) outputs
                        loss = nn.functional.mse_loss(outputs[1], targets)
                    else:
                        loss = nn.functional.mse_loss(outputs, targets)
                    total_loss += loss.item() * len(inputs)

                total_samples += len(inputs)
                batches_used = i + 1

        avg_loss = total_loss / max(1, total_samples)
        metrics["samples"] = float(total_samples)
        metrics["batches"] = float(min(batches_to_use, batches_used))

        model.train()
        return avg_loss, metrics, total_samples

    def _adapt_interval(self) -> None:
        """Adapt validation interval based on loss variance."""
        losses = list(self._loss_history)

        # Compute coefficient of variation
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        if mean_loss > 0:
            cv = std_loss / mean_loss
        else:
            cv = 0

        # High variance = validate more frequently
        # Low variance = validate less frequently
        if cv > 0.1:  # High variance
            self._current_interval = max(
                self.min_interval_steps, int(self._current_interval * 0.8)
            )
        elif cv < 0.02:  # Low variance
            self._current_interval = min(
                self.max_interval_steps, int(self._current_interval * 1.2)
            )

    def get_best(self) -> tuple[float, int]:
        """Get best validation loss and step."""
        return self._best_loss, self._best_step

    def get_results(self) -> list[ValidationResult]:
        """Get all validation results."""
        return self._results.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get validation summary statistics."""
        if not self._results:
            return {"num_validations": 0}

        losses = [r.val_loss for r in self._results]
        durations = [r.duration_seconds for r in self._results]

        return {
            "num_validations": len(self._results),
            "best_loss": self._best_loss,
            "best_step": self._best_step,
            "latest_loss": losses[-1],
            "mean_loss": float(np.mean(losses)),
            "mean_duration": float(np.mean(durations)),
            "current_interval": self._current_interval,
            "improvements": sum(1 for r in self._results if r.is_improvement),
        }

    def reset(self) -> None:
        """Reset validation state for new training run."""
        self._last_val_step = -self.interval_steps
        self._last_val_epoch = -1.0
        self._current_interval = self.interval_steps
        self._results.clear()
        self._best_loss = float("inf")
        self._best_step = 0
        self._loss_history.clear()


class EnhancedEarlyStopping:
    """Enhanced early stopping with multiple criteria and plateau detection.

    Features:
    - Track both loss and Elo for stopping criteria
    - Plateau detection with configurable patience
    - Best model state preservation
    - Restore on stop

    Usage:
        early_stopping = EnhancedEarlyStopping(patience=10)

        for epoch in range(epochs):
            val_loss = validate()

            if early_stopping.should_stop(val_loss=val_loss, current_elo=elo):
                early_stopping.restore_best_model(model)
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        elo_patience: int | None = None,
        elo_min_improvement: float = 5.0,
        mode: str = "min",
        restore_best: bool = True,
        min_epochs: int = 3,
        emit_events: bool = True,
        plateau_warning_threshold: float = 0.5,
        config_name: str = "unknown",
    ):
        """Initialize enhanced early stopping.

        Args:
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
            elo_patience: Separate patience for Elo (None = use patience).
            elo_min_improvement: Minimum Elo improvement to reset counter.
            mode: 'min' for loss, 'max' for accuracy.
            restore_best: Whether to restore best model on stop.
            min_epochs: Minimum epochs before early stopping can trigger.
            emit_events: Whether to emit PLATEAU_DETECTED events (default True).
            plateau_warning_threshold: Emit warning when counter reaches this
                fraction of patience (default 0.5 = 50%).
            config_name: Config name for event emission (e.g., "hex8_2p").
        """
        self.patience = patience
        self.min_delta = min_delta
        self.elo_patience = elo_patience or patience
        self.elo_min_improvement = elo_min_improvement
        self.mode = mode
        self.restore_best = restore_best
        self.min_epochs = min_epochs
        self.emit_events = emit_events
        self.plateau_warning_threshold = plateau_warning_threshold
        self.config_name = config_name

        self.best_loss = float("inf") if mode == "min" else float("-inf")
        self.best_elo = float("-inf")
        self.loss_counter = 0
        self.elo_counter = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch = 0
        self._stopped = False
        self._call_epoch = 0  # Epoch counter for __call__ legacy interface
        self._loss_plateau_emitted = False  # Track if loss plateau warning emitted
        self._elo_plateau_emitted = False  # Track if elo plateau warning emitted

    def should_stop(
        self,
        val_loss: float | None = None,
        current_elo: float | None = None,
        model: nn.Module | None = None,
        epoch: int = 0,
    ) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Validation loss.
            current_elo: Current Elo rating.
            model: Model to save best state from.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        improved = False

        # Check loss improvement
        if val_loss is not None:
            if self._is_improvement(val_loss, self.best_loss):
                self.best_loss = val_loss
                self.loss_counter = 0
                improved = True
            else:
                self.loss_counter += 1

        # Check Elo improvement
        if current_elo is not None:
            if current_elo > self.best_elo + self.elo_min_improvement:
                self.best_elo = current_elo
                self.elo_counter = 0
                improved = True
            else:
                self.elo_counter += 1

        # Save best model state (always track, even during min_epochs)
        if improved and model is not None:
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            # Reset plateau warning flags on improvement
            self._loss_plateau_emitted = False
            self._elo_plateau_emitted = False

        # Check and emit plateau warnings (emit even during min_epochs for feedback)
        self._check_and_emit_plateau_warnings(val_loss, current_elo)

        # Don't allow early stopping before min_epochs
        # (but we still track improvements above)
        if epoch < self.min_epochs:
            self._stopped = False
            return False

        # Check if should stop
        loss_stop = val_loss is not None and self.loss_counter >= self.patience
        elo_stop = current_elo is not None and self.elo_counter >= self.elo_patience

        # Stop if both criteria suggest stopping (or one if only one is tracked)
        if val_loss is not None and current_elo is not None:
            self._stopped = loss_stop and elo_stop
        else:
            self._stopped = loss_stop or elo_stop

        if self._stopped:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {self.best_epoch}, "
                f"Best loss: {self.best_loss:.4f}, "
                f"Best Elo: {self.best_elo:.1f}"
            )

        return self._stopped

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def _emit_plateau_event(
        self,
        plateau_type: str,
        current_value: float,
        best_value: float,
        epochs_since_improvement: int,
    ) -> None:
        """Emit PLATEAU_DETECTED event for curriculum feedback.

        Args:
            plateau_type: "loss" or "elo".
            current_value: Current metric value.
            best_value: Best value seen.
            epochs_since_improvement: How many epochs without improvement.
        """
        if not self.emit_events:
            return

        try:
            from app.coordination.event_emission_helpers import safe_emit_event

            safe_emit_event(
                "PLATEAU_DETECTED",
                {
                    "metric_name": f"{self.config_name}_{plateau_type}",
                    "current_value": current_value,
                    "best_value": best_value,
                    "epochs_since_improvement": epochs_since_improvement,
                    "plateau_type": plateau_type,
                    "config_key": self.config_name,
                    "patience": self.patience if plateau_type == "loss" else self.elo_patience,
                    "threshold_pct": self.plateau_warning_threshold,
                },
                log_after=f"[EnhancedEarlyStopping] Emitted PLATEAU_DETECTED for {plateau_type} "
                f"(epochs={epochs_since_improvement}, config={self.config_name})",
                context="early_stopping",
            )
        except Exception as e:
            logger.debug(f"[EnhancedEarlyStopping] Event emission failed: {e}")

    def _check_and_emit_plateau_warnings(
        self,
        val_loss: float | None,
        current_elo: float | None,
    ) -> None:
        """Check if plateau warning threshold reached and emit events."""
        # Loss plateau warning
        if val_loss is not None and not self._loss_plateau_emitted:
            warning_threshold = int(self.patience * self.plateau_warning_threshold)
            if self.loss_counter >= warning_threshold and warning_threshold > 0:
                self._emit_plateau_event(
                    plateau_type="loss",
                    current_value=val_loss,
                    best_value=self.best_loss,
                    epochs_since_improvement=self.loss_counter,
                )
                self._loss_plateau_emitted = True

        # Elo plateau warning
        if current_elo is not None and not self._elo_plateau_emitted:
            warning_threshold = int(self.elo_patience * self.plateau_warning_threshold)
            if self.elo_counter >= warning_threshold and warning_threshold > 0:
                self._emit_plateau_event(
                    plateau_type="elo",
                    current_value=current_elo,
                    best_value=self.best_elo,
                    epochs_since_improvement=self.elo_counter,
                )
                self._elo_plateau_emitted = True

    def restore_best_model(self, model: nn.Module) -> bool:
        """Restore model to best state.

        Returns:
            True if restoration was successful.
        """
        if self.best_state is not None and self.restore_best:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored model to best state from epoch {self.best_epoch}")
            return True
        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_loss = float("inf") if self.mode == "min" else float("-inf")
        self.best_elo = float("-inf")
        self.loss_counter = 0
        self.elo_counter = 0
        self.best_state = None
        self.best_epoch = 0
        self._stopped = False
        self._call_epoch = 0  # Reset epoch counter for legacy interface
        self._loss_plateau_emitted = False
        self._elo_plateau_emitted = False

    # =========================================================================
    # Backwards Compatibility Methods (for drop-in replacement of basic EarlyStopping)
    # =========================================================================

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop (backwards compatible interface).

        This allows EnhancedEarlyStopping to be used as a drop-in replacement
        for the basic EarlyStopping class in train.py.

        Args:
            val_loss: Current validation loss.
            model: Model to save state from if this is best so far.

        Returns:
            True if training should stop, False otherwise.
        """
        # Track epoch internally for legacy interface
        result = self.should_stop(val_loss=val_loss, model=model, epoch=self._call_epoch)
        self._call_epoch += 1
        return result

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best weights to the model (backwards compatible alias).

        This is an alias for restore_best_model() to maintain compatibility
        with code using the basic EarlyStopping class.
        """
        self.restore_best_model(model)

    @property
    def counter(self) -> int:
        """Backwards compatible counter property (returns loss_counter)."""
        return self.loss_counter


# Backwards compatible alias
EarlyStopping = EnhancedEarlyStopping
