"""
Learning Rate Scheduling for RingRift AI Training.

Provides adaptive learning rate schedulers:
- AdaptiveLRScheduler: Adjusts LR based on Elo progress and loss trends
- WarmRestartsScheduler: Cosine annealing with warm restarts (SGDR)

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np
import torch.optim as optim

logger = logging.getLogger(__name__)


class AdaptiveLRScheduler:
    """
    Adjusts learning rate based on Elo progress and loss trends.

    Features:
    - Increase LR when Elo is stagnating
    - Decrease LR when loss is oscillating
    - Warm restart on distribution shift

    Usage:
        scheduler = AdaptiveLRScheduler(optimizer, base_lr=0.001)

        for epoch in range(epochs):
            train_loss = train_epoch()
            scheduler.step(
                train_loss=train_loss,
                current_elo=get_current_elo(),
            )
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float = 0.001,
        min_lr: float = 1e-6,
        max_lr: float = 0.01,
        elo_lookback: int = 5,
        loss_lookback: int = 10,
        elo_stagnation_threshold: float = 10.0,
        loss_oscillation_threshold: float = 0.1,
        lr_increase_factor: float = 1.5,
        lr_decrease_factor: float = 0.5,
        warmup_epochs: int = 5,
    ):
        """
        Args:
            optimizer: Optimizer to adjust
            base_lr: Base learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            elo_lookback: Number of Elo updates to consider
            loss_lookback: Number of loss values to consider
            elo_stagnation_threshold: Elo change below this is considered stagnation
            loss_oscillation_threshold: Loss variance ratio above this is oscillating
            lr_increase_factor: Factor to increase LR on stagnation
            lr_decrease_factor: Factor to decrease LR on oscillation
            warmup_epochs: Number of initial epochs to skip adjustment
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.elo_lookback = elo_lookback
        self.loss_lookback = loss_lookback
        self.elo_stagnation_threshold = elo_stagnation_threshold
        self.loss_oscillation_threshold = loss_oscillation_threshold
        self.lr_increase_factor = lr_increase_factor
        self.lr_decrease_factor = lr_decrease_factor
        self.warmup_epochs = warmup_epochs

        self._current_lr = base_lr
        self._elo_history: deque = deque(maxlen=elo_lookback)
        self._loss_history: deque = deque(maxlen=loss_lookback)
        self._epoch = 0

        # Set initial LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

    def step(
        self,
        train_loss: float,
        current_elo: float | None = None,
        val_loss: float | None = None,
    ) -> float:
        """
        Update learning rate based on progress.

        Args:
            train_loss: Training loss for this epoch
            current_elo: Current Elo rating (if available)
            val_loss: Validation loss (optional)

        Returns:
            New learning rate
        """
        self._epoch += 1
        self._loss_history.append(train_loss)

        if current_elo is not None:
            self._elo_history.append(current_elo)

        # Skip adjustment during warmup
        if self._epoch <= self.warmup_epochs:
            return self._current_lr

        # Check for Elo stagnation
        elo_stagnating = self._check_elo_stagnation()

        # Check for loss oscillation
        loss_oscillating = self._check_loss_oscillation()

        # Adjust LR
        new_lr = self._current_lr

        if elo_stagnating and not loss_oscillating:
            # Increase LR to explore more
            new_lr = min(self.max_lr, self._current_lr * self.lr_increase_factor)
            logger.info(f"Elo stagnating, increasing LR: {self._current_lr:.6f} -> {new_lr:.6f}")
        elif loss_oscillating:
            # Decrease LR for stability
            new_lr = max(self.min_lr, self._current_lr * self.lr_decrease_factor)
            logger.info(f"Loss oscillating, decreasing LR: {self._current_lr:.6f} -> {new_lr:.6f}")

        # Apply new LR
        if new_lr != self._current_lr:
            self._current_lr = new_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

        return self._current_lr

    def _check_elo_stagnation(self) -> bool:
        """Check if Elo is stagnating."""
        if len(self._elo_history) < self.elo_lookback:
            return False

        elo_list = list(self._elo_history)
        elo_change = abs(elo_list[-1] - elo_list[0])
        return elo_change < self.elo_stagnation_threshold

    def _check_loss_oscillation(self) -> bool:
        """Check if loss is oscillating."""
        if len(self._loss_history) < self.loss_lookback:
            return False

        losses = np.array(list(self._loss_history))

        # Check variance ratio (high variance relative to mean = oscillation)
        mean_loss = np.mean(losses)
        if mean_loss < 1e-8:
            return False

        variance = np.var(losses)
        variance_ratio = variance / (mean_loss ** 2)

        return variance_ratio > self.loss_oscillation_threshold

    def warm_restart(self, lr_factor: float = 1.0) -> None:
        """
        Perform a warm restart (e.g., on data distribution shift).

        Args:
            lr_factor: Factor to apply to base LR for restart
        """
        new_lr = min(self.max_lr, self.base_lr * lr_factor)
        self._current_lr = new_lr
        self._loss_history.clear()
        self._elo_history.clear()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        logger.info(f"Warm restart with LR: {new_lr:.6f}")

    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return self._current_lr


class WarmRestartsScheduler:
    """
    Cosine annealing with warm restarts (SGDR).

    Implements the learning rate schedule from "SGDR: Stochastic Gradient
    Descent with Warm Restarts" (Loshchilov & Hutter, 2017).

    The learning rate follows a cosine curve from max to min, then "restarts"
    by jumping back to max. The restart periods can grow exponentially.

    Usage:
        scheduler = WarmRestartsScheduler(
            optimizer=optimizer,
            T_0=10,  # Initial restart period (epochs)
            T_mult=2,  # Period multiplier after each restart
        )

        for epoch in range(epochs):
            for batch in dataloader:
                train_step()
                scheduler.step()  # Call every batch

            # Or call once per epoch
            scheduler.step_epoch()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        eta_max: float | None = None,
        last_epoch: int = -1,
        warmup_steps: int = 0,
    ):
        """
        Args:
            optimizer: Optimizer to adjust
            T_0: Initial number of epochs/steps until first restart
            T_mult: Multiplier for T_i after each restart (T_i+1 = T_i * T_mult)
            eta_min: Minimum learning rate
            eta_max: Maximum learning rate (defaults to optimizer's initial LR)
            last_epoch: Index of last epoch (-1 = start fresh)
            warmup_steps: Number of initial steps for linear warmup to eta_max
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps

        # Get initial LR from optimizer
        self.eta_max = eta_max or optimizer.param_groups[0]['lr']

        self._step_count = 0
        self._epoch = last_epoch + 1
        self._restart_count = 0
        self._T_cur = 0  # Current position in restart period
        self._T_i = T_0  # Current restart period length
        self._current_lr = self.eta_max

        # Store restart history
        self._restart_epochs: list[int] = [0]

    def step(self, epoch: int | None = None) -> float:
        """
        Update learning rate (call once per step/batch).

        Args:
            epoch: Current epoch (optional, for epoch-based scheduling)

        Returns:
            Current learning rate
        """
        self._step_count += 1

        if epoch is not None:
            self._epoch = epoch
            self._T_cur = epoch
        else:
            self._T_cur += 1

        # Handle warmup
        if self._step_count <= self.warmup_steps:
            self._current_lr = self.eta_min + (
                (self.eta_max - self.eta_min) * self._step_count / self.warmup_steps
            )
            self._apply_lr()
            return self._current_lr

        # Check for restart
        if self._T_cur >= self._T_i:
            self._restart()

        # Compute cosine annealing LR
        self._current_lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + math.cos(math.pi * self._T_cur / self._T_i)
        )

        self._apply_lr()
        return self._current_lr

    def step_epoch(self) -> float:
        """
        Update learning rate (call once per epoch).

        Returns:
            Current learning rate
        """
        self._epoch += 1
        return self.step(epoch=self._epoch)

    def _restart(self) -> None:
        """Perform a warm restart."""
        self._restart_count += 1
        self._restart_epochs.append(self._epoch)

        # Reset position in period
        self._T_cur = 0

        # Update period length
        self._T_i = self._T_i * self.T_mult

        logger.info(
            f"Warm restart #{self._restart_count} at epoch {self._epoch}, "
            f"new period T_i={self._T_i}"
        )

    def _apply_lr(self) -> None:
        """Apply current LR to optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._current_lr

    def get_last_lr(self) -> list[float]:
        """Get last computed learning rate for each param group."""
        return [self._current_lr] * len(self.optimizer.param_groups)

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state."""
        return {
            'step_count': self._step_count,
            'epoch': self._epoch,
            'restart_count': self._restart_count,
            'T_cur': self._T_cur,
            'T_i': self._T_i,
            'current_lr': self._current_lr,
            'restart_epochs': self._restart_epochs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state."""
        self._step_count = state_dict['step_count']
        self._epoch = state_dict['epoch']
        self._restart_count = state_dict['restart_count']
        self._T_cur = state_dict['T_cur']
        self._T_i = state_dict['T_i']
        self._current_lr = state_dict['current_lr']
        self._restart_epochs = state_dict['restart_epochs']

    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return self._current_lr

    @property
    def num_restarts(self) -> int:
        """Get number of restarts performed."""
        return self._restart_count

    def get_schedule_info(self) -> dict[str, Any]:
        """Get information about current schedule state."""
        return {
            'current_lr': self._current_lr,
            'epoch': self._epoch,
            'step': self._step_count,
            'restart_count': self._restart_count,
            'current_period': self._T_i,
            'position_in_period': self._T_cur,
            'progress_in_period': self._T_cur / self._T_i if self._T_i > 0 else 0,
            'restart_epochs': self._restart_epochs,
        }
