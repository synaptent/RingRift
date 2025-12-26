"""
Gradient Management for RingRift AI Training.

This module provides gradient accumulation and adaptive gradient clipping utilities.

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class GradientAccumulator:
    """
    Handles gradient accumulation for larger effective batch sizes.

    Useful when GPU memory is limited but larger batch sizes are desired.

    Usage:
        accumulator = GradientAccumulator(accumulation_steps=4)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            loss = model(inputs, targets) / accumulator.accumulation_steps
            loss.backward()

            if accumulator.should_step(batch_idx):
                optimizer.step()
                optimizer.zero_grad()
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
    ):
        """
        Args:
            accumulation_steps: Number of batches to accumulate before stepping
            max_grad_norm: Maximum gradient norm for clipping (None to disable)
        """
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self._step_count = 0

    def should_step(self, batch_idx: int) -> bool:
        """Check if optimizer should step at this batch index."""
        return (batch_idx + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation steps."""
        return loss / self.accumulation_steps

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients if max_grad_norm is set. Returns gradient norm."""
        if self.max_grad_norm is not None:
            return torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_grad_norm
            ).item()
        return 0.0

    def step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> float:
        """
        Perform optimizer step with optional gradient clipping and AMP.

        Returns:
            Gradient norm before clipping
        """
        grad_norm = self.clip_gradients(model)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()
        self._step_count += 1

        return grad_norm

    @property
    def effective_batch_size(self) -> int:
        """Get the effective batch size multiplier."""
        return self.accumulation_steps


class AdaptiveGradientClipper:
    """
    Adaptive gradient clipping based on gradient norm history.

    Automatically adjusts clipping threshold based on recent gradient statistics.
    Prevents both gradient explosion and overly aggressive clipping.

    Features:
    - Tracks gradient norm history
    - Adjusts clip threshold based on percentile of recent norms
    - Prevents both explosion (high norms) and over-clipping (low threshold)

    Usage:
        clipper = AdaptiveGradientClipper(initial_max_norm=1.0)

        for batch in dataloader:
            loss.backward()
            grad_norm = clipper.update_and_clip(model.parameters())
            optimizer.step()

            # Optional: log statistics
            stats = clipper.get_stats()
    """

    def __init__(
        self,
        initial_max_norm: float = 1.0,
        percentile: float = 90.0,
        history_size: int = 100,
        min_clip: float = 0.1,
        max_clip: float = 10.0,
        # Backwards compatibility alias
        initial_clip: float | None = None,
    ):
        """
        Args:
            initial_max_norm: Starting gradient clipping threshold
            percentile: Percentile of gradient norms to use for threshold
            history_size: Number of gradient norms to track
            min_clip: Minimum allowed clipping threshold
            max_clip: Maximum allowed clipping threshold
            initial_clip: Alias for initial_max_norm (backwards compatibility)
        """
        # Support backwards compatible parameter name
        if initial_clip is not None:
            initial_max_norm = initial_clip
        self.current_max_norm = initial_max_norm
        self.percentile = percentile
        self.history_size = history_size
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.grad_norms: list[float] = []

    def update_and_clip(self, parameters) -> float:
        """
        Update history and clip gradients.

        Args:
            parameters: Model parameters (from model.parameters())

        Returns:
            The actual gradient norm before clipping
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.grad_norms.append(total_norm)
        if len(self.grad_norms) > self.history_size:
            self.grad_norms.pop(0)

        # Update threshold based on history
        if len(self.grad_norms) >= 10:
            threshold = np.percentile(self.grad_norms, self.percentile)
            self.current_max_norm = np.clip(threshold * 1.5, self.min_clip, self.max_clip)

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(parameters, self.current_max_norm)
        return total_norm

    def get_stats(self) -> dict[str, float]:
        """Get current clipping statistics."""
        return {
            'current_clip_norm': self.current_max_norm,
            'mean_grad_norm': np.mean(self.grad_norms) if self.grad_norms else 0,
            'max_grad_norm': max(self.grad_norms) if self.grad_norms else 0,
            'history_size': len(self.grad_norms),
        }

    def reset(self) -> None:
        """Reset the gradient history."""
        self.grad_norms.clear()
