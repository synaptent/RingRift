"""
Value Head Calibration Automation for RingRift AI Training.

Monitors calibration metrics during training and triggers recalibration
when predictions deviate too much from outcomes.

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class CalibrationAutomation:
    """
    Automates value head calibration monitoring and triggering.

    Monitors calibration metrics during training and triggers
    recalibration when predictions deviate too much from outcomes.

    Usage:
        calibration_auto = CalibrationAutomation(threshold=0.05)

        for epoch in range(epochs):
            # ... training ...

            # Check calibration
            if calibration_auto.should_recalibrate(predictions, outcomes):
                temperature = calibration_auto.compute_optimal_temperature(
                    model, validation_loader
                )
                apply_temperature_scaling(model, temperature)
    """

    def __init__(
        self,
        deviation_threshold: float = 0.05,
        check_interval: int = 5,
        min_samples: int = 1000,
        window_size: int = 5000,
    ):
        """
        Args:
            deviation_threshold: Trigger recalibration if ECE > this
            check_interval: Check calibration every N epochs
            min_samples: Minimum samples before checking
            window_size: Size of rolling window for calibration check
        """
        self.deviation_threshold = deviation_threshold
        self.check_interval = check_interval
        self.min_samples = min_samples
        self.window_size = window_size

        self._predictions: deque = deque(maxlen=window_size)
        self._outcomes: deque = deque(maxlen=window_size)
        self._epoch = 0
        self._last_calibration_epoch = 0
        self._calibration_history: list[dict[str, Any]] = []

    def add_samples(
        self,
        predictions: list[float] | np.ndarray | torch.Tensor,
        outcomes: list[float] | np.ndarray | torch.Tensor,
    ) -> None:
        """Add prediction-outcome pairs for monitoring."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(outcomes, torch.Tensor):
            outcomes = outcomes.detach().cpu().numpy()

        predictions = np.asarray(predictions).flatten()
        outcomes = np.asarray(outcomes).flatten()

        for p, o in zip(predictions, outcomes, strict=False):
            self._predictions.append(p)
            self._outcomes.append(o)

    def should_recalibrate(self, epoch: int | None = None) -> bool:
        """
        Check if recalibration is needed.

        Args:
            epoch: Current epoch (if None, uses internal counter)

        Returns:
            True if recalibration is recommended
        """
        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1

        # Check interval
        if (self._epoch - self._last_calibration_epoch) < self.check_interval:
            return False

        # Check minimum samples
        if len(self._predictions) < self.min_samples:
            return False

        # Compute ECE
        ece = self._compute_ece()

        # Record
        self._calibration_history.append({
            'epoch': self._epoch,
            'ece': ece,
            'samples': len(self._predictions),
            'triggered': ece > self.deviation_threshold,
        })

        if ece > self.deviation_threshold:
            logger.warning(
                f"Calibration check: ECE={ece:.4f} exceeds threshold "
                f"{self.deviation_threshold}, recalibration recommended"
            )
            return True

        logger.info(f"Calibration check: ECE={ece:.4f}, within threshold")
        return False

    def _compute_ece(self, num_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        predictions = np.array(list(self._predictions))
        outcomes = np.array(list(self._outcomes))

        # Convert from [-1, 1] to [0, 1]
        pred_probs = (predictions + 1) / 2
        outcome_probs = (outcomes + 1) / 2

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]

            if i == num_bins - 1:
                mask = (pred_probs >= lower) & (pred_probs <= upper)
            else:
                mask = (pred_probs >= lower) & (pred_probs < upper)

            if mask.sum() > 0:
                bin_acc = outcome_probs[mask].mean()
                bin_conf = pred_probs[mask].mean()
                bin_weight = mask.sum() / len(pred_probs)
                ece += bin_weight * abs(bin_acc - bin_conf)

        return ece

    def compute_optimal_temperature(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Compute optimal temperature for Platt scaling.

        Args:
            model: Model to calibrate
            dataloader: Validation dataloader
            device: Device for computation
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        if device is None:
            device = next(model.parameters()).device

        # Collect logits and labels
        logits_list = []
        labels_list = []

        model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                # Handle different output formats
                if isinstance(outputs, tuple):
                    value_output = outputs[1]  # Assume (policy, value)
                else:
                    value_output = outputs

                logits_list.append(value_output.cpu())
                labels_list.append(targets.cpu())

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1))
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def eval_temp():
            optimizer.zero_grad()
            scaled = logits / temperature
            loss = nn.functional.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_temp)

        optimal_temp = temperature.item()
        self._last_calibration_epoch = self._epoch

        logger.info(f"Computed optimal temperature: {optimal_temp:.4f}")
        return optimal_temp

    def get_calibration_history(self) -> list[dict[str, Any]]:
        """Get calibration check history."""
        return self._calibration_history
