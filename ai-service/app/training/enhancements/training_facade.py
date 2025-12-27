"""
Unified Training Enhancements Facade.

Provides a single interface to all training enhancement modules:
- Per-sample loss tracking (enables hard example mining)
- Hard example mining (curriculum learning)
- Curriculum LR scaling
- Quality + freshness weighting

This facade consolidates 3,193 lines of orphaned enhancement code into
an easily-integrated interface for train.py.

Usage:
    from app.training.enhancements.training_facade import (
        TrainingEnhancementsFacade,
        FacadeConfig,
    )

    # Initialize with training
    facade = TrainingEnhancementsFacade(config=FacadeConfig())

    # In training loop:
    for batch_idx, batch in enumerate(dataloader):
        policy_logits, value_pred = model(inputs)

        # Compute per-sample losses (enables mining)
        per_sample_losses = facade.compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets
        )

        # Get weighted loss (upweights hard examples)
        weighted_loss = facade.get_weighted_loss(
            per_sample_losses, batch_indices=batch_indices
        )

        # Record for mining
        facade.record_batch(batch_indices, per_sample_losses)

        # Get curriculum LR scale
        lr_scale = facade.get_curriculum_lr_scale(epoch / total_epochs)

    # End of epoch
    stats = facade.get_epoch_statistics()
    facade.on_epoch_end()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from app.training.enhancements.hard_example_mining import HardExampleMiner
from app.training.enhancements.per_sample_loss import (
    PerSampleLossTracker,
    compute_per_sample_loss,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FacadeConfig:
    """Configuration for training enhancements facade."""

    # Hard example mining
    enable_hard_mining: bool = True
    hard_fraction: float = 0.3
    hard_buffer_size: int = 10000
    hard_min_samples_before_mining: int = 1000
    hard_base_weight: float = 1.0
    hard_upweight: float = 2.0

    # Per-sample loss tracking
    track_per_sample_loss: bool = True
    loss_tracker_max_samples: int = 10000

    # Curriculum LR scaling
    enable_curriculum_lr: bool = True
    curriculum_lr_min_scale: float = 0.8
    curriculum_lr_max_scale: float = 1.2
    curriculum_warmup_fraction: float = 0.1

    # Quality/freshness weighting
    enable_freshness_weighting: bool = True
    freshness_decay_hours: float = 24.0
    freshness_weight: float = 0.2

    # Policy weight in combined loss
    policy_weight: float = 1.0


@dataclass
class EpochStatistics:
    """Statistics for a training epoch."""

    mean_loss: float = 0.0
    mean_per_sample_loss: float = 0.0
    hard_examples_fraction: float = 0.0
    mining_active: bool = False
    tracked_samples: int = 0
    curriculum_lr_scale: float = 1.0


class TrainingEnhancementsFacade:
    """
    Unified interface for all training enhancements.

    Integrates:
    - HardExampleMiner: Identifies and upweights difficult samples
    - PerSampleLossTracker: Tracks individual sample losses for analysis
    - Curriculum LR: Scales learning rate based on training progress
    - Freshness weighting: Prioritizes recent training data

    This facade provides a clean API for train.py integration points.
    """

    def __init__(self, config: FacadeConfig | None = None):
        """Initialize the facade with configuration.

        Args:
            config: Facade configuration. Uses defaults if not provided.
        """
        self.config = config or FacadeConfig()
        self._epoch = 0
        self._total_epochs = 1  # Updated via set_total_epochs
        self._total_samples_seen = 0
        self._epoch_loss_sum = 0.0
        self._epoch_sample_count = 0

        # Initialize hard example miner
        self._miner: HardExampleMiner | None = None
        if self.config.enable_hard_mining:
            self._miner = HardExampleMiner(
                buffer_size=self.config.hard_buffer_size,
                hard_fraction=self.config.hard_fraction,
                min_samples_before_mining=self.config.hard_min_samples_before_mining,
            )

        # Initialize per-sample loss tracker
        self._loss_tracker: PerSampleLossTracker | None = None
        if self.config.track_per_sample_loss:
            self._loss_tracker = PerSampleLossTracker(
                max_samples=self.config.loss_tracker_max_samples,
            )

        logger.info(
            f"TrainingEnhancementsFacade initialized: "
            f"hard_mining={self.config.enable_hard_mining}, "
            f"curriculum_lr={self.config.enable_curriculum_lr}, "
            f"freshness={self.config.enable_freshness_weighting}"
        )

    def set_total_epochs(self, total_epochs: int) -> None:
        """Set total epochs for curriculum LR calculation.

        Args:
            total_epochs: Total number of training epochs.
        """
        self._total_epochs = max(1, total_epochs)

    def compute_per_sample_loss(
        self,
        policy_logits: torch.Tensor,
        policy_targets: torch.Tensor,
        value_pred: torch.Tensor,
        value_targets: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Compute per-sample combined loss for policy and value heads.

        Args:
            policy_logits: Model policy output (B, num_actions) - logits
            policy_targets: Policy labels (B, num_actions) - probabilities
            value_pred: Model value output (B,) or (B, num_players)
            value_targets: Value labels (B,) or (B, num_players)
            reduction: "none" for per-sample, "mean"/"sum" for aggregated

        Returns:
            Per-sample losses of shape (B,) if reduction="none", else scalar
        """
        return compute_per_sample_loss(
            policy_logits=policy_logits,
            policy_targets=policy_targets,
            value_pred=value_pred,
            value_targets=value_targets,
            policy_weight=self.config.policy_weight,
            reduction=reduction,
        )

    def get_weighted_loss(
        self,
        per_sample_losses: torch.Tensor,
        batch_indices: torch.Tensor | np.ndarray | list[int] | None = None,
    ) -> torch.Tensor:
        """
        Apply hard example weighting to per-sample losses.

        If hard mining is enabled and sufficient samples have been seen,
        hard examples are upweighted to focus training on difficult cases.

        Args:
            per_sample_losses: Per-sample losses of shape (B,)
            batch_indices: Dataset indices for the batch (optional)

        Returns:
            Weighted mean loss (scalar tensor)
        """
        if batch_indices is None or not self.config.enable_hard_mining:
            return per_sample_losses.mean()

        if self._miner is None:
            return per_sample_losses.mean()

        # Get sample weights from miner
        weights = self._miner.get_sample_weights(
            indices=batch_indices,
            base_weight=self.config.hard_base_weight,
            hard_weight=self.config.hard_upweight,
        )

        # Convert to tensor on same device
        weights_tensor = torch.from_numpy(weights).to(
            per_sample_losses.device, dtype=per_sample_losses.dtype
        )

        # Normalize weights to maintain loss scale
        weights_tensor = weights_tensor / weights_tensor.mean()

        # Weighted mean
        weighted_loss = (per_sample_losses * weights_tensor).mean()
        return weighted_loss

    def record_batch(
        self,
        batch_indices: torch.Tensor | np.ndarray | list[int],
        per_sample_losses: torch.Tensor,
        uncertainties: torch.Tensor | None = None,
    ) -> None:
        """
        Record batch data for hard example mining and loss tracking.

        Should be called after computing per-sample losses.

        Args:
            batch_indices: Dataset indices for the batch
            per_sample_losses: Per-sample losses
            uncertainties: Optional per-sample uncertainty (e.g., policy entropy)
        """
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.detach().cpu()
        if isinstance(per_sample_losses, torch.Tensor):
            losses_np = per_sample_losses.detach().cpu()
        else:
            losses_np = per_sample_losses

        batch_size = len(batch_indices) if hasattr(batch_indices, '__len__') else 1
        self._total_samples_seen += batch_size
        self._epoch_sample_count += batch_size
        self._epoch_loss_sum += float(per_sample_losses.sum().item()) if isinstance(per_sample_losses, torch.Tensor) else sum(per_sample_losses)

        # Record for hard example mining
        if self._miner is not None:
            self._miner.record_batch(
                indices=batch_indices,
                losses=losses_np,
                uncertainties=uncertainties,
            )

        # Record for loss tracking
        if self._loss_tracker is not None:
            self._loss_tracker.record_batch(
                batch_indices=batch_indices,
                losses=per_sample_losses if isinstance(per_sample_losses, torch.Tensor) else torch.tensor(per_sample_losses),
                epoch=self._epoch,
            )

    def get_curriculum_lr_scale(self, progress: float | None = None) -> float:
        """
        Get learning rate scale based on training progress.

        Implements curriculum-aware LR scaling:
        - Warmup phase: Linear ramp from min_scale to 1.0
        - Main phase: Linear interpolation from 1.0 to max_scale
        - This allows higher LR as model becomes more capable

        Args:
            progress: Training progress (0.0-1.0). If None, computed from epoch.

        Returns:
            LR scale factor (typically 0.8-1.2)
        """
        if not self.config.enable_curriculum_lr:
            return 1.0

        if progress is None:
            progress = self._epoch / max(1, self._total_epochs)

        progress = max(0.0, min(1.0, progress))

        warmup_frac = self.config.curriculum_warmup_fraction
        min_scale = self.config.curriculum_lr_min_scale
        max_scale = self.config.curriculum_lr_max_scale

        if progress < warmup_frac:
            # Warmup: linear from min_scale to 1.0
            warmup_progress = progress / warmup_frac
            return min_scale + (1.0 - min_scale) * warmup_progress
        else:
            # Main phase: linear from 1.0 to max_scale
            main_progress = (progress - warmup_frac) / (1.0 - warmup_frac)
            return 1.0 + (max_scale - 1.0) * main_progress

    def compute_freshness_weight(
        self,
        game_timestamp: float,
        current_time: float | None = None,
    ) -> float:
        """
        Compute freshness weight for a training sample.

        Implements exponential decay based on sample age.

        Args:
            game_timestamp: Unix timestamp when game was played
            current_time: Current time (default: time.time())

        Returns:
            Freshness weight (0.0-1.0, where 1.0 = newest)
        """
        if not self.config.enable_freshness_weighting:
            return 1.0

        import time as time_module
        if current_time is None:
            current_time = time_module.time()

        age_hours = (current_time - game_timestamp) / 3600
        if age_hours < 0:
            return 1.0

        # Exponential decay
        freshness = math.exp(-age_hours / self.config.freshness_decay_hours)
        return max(0.0, min(1.0, freshness))

    def get_epoch_statistics(self) -> EpochStatistics:
        """Get statistics for the current epoch."""
        stats = EpochStatistics(
            curriculum_lr_scale=self.get_curriculum_lr_scale(),
        )

        if self._epoch_sample_count > 0:
            stats.mean_loss = self._epoch_loss_sum / self._epoch_sample_count

        if self._miner is not None:
            miner_stats = self._miner.get_statistics()
            stats.mining_active = miner_stats.get('mining_active', False)
            stats.tracked_samples = miner_stats.get('tracked_examples', 0)
            stats.mean_per_sample_loss = miner_stats.get('mean_loss', 0.0)

            # Compute hard examples fraction
            if stats.tracked_samples > 0:
                hard_indices = self._miner.get_hard_indices(num_samples=stats.tracked_samples)
                stats.hard_examples_fraction = len(hard_indices) / stats.tracked_samples

        if self._loss_tracker is not None:
            tracker_stats = self._loss_tracker.get_statistics()
            stats.tracked_samples = tracker_stats.get('tracked_samples', 0)

        return stats

    def on_epoch_end(self) -> dict[str, Any]:
        """
        Called at the end of each epoch.

        Returns epoch statistics and prepares for next epoch.

        Returns:
            Dictionary of epoch statistics for logging
        """
        stats = self.get_epoch_statistics()

        # Log summary
        logger.info(
            f"Epoch {self._epoch} enhancements: "
            f"mean_loss={stats.mean_loss:.4f}, "
            f"mining_active={stats.mining_active}, "
            f"hard_frac={stats.hard_examples_fraction:.2%}, "
            f"lr_scale={stats.curriculum_lr_scale:.3f}"
        )

        # Reset epoch counters
        self._epoch += 1
        self._epoch_loss_sum = 0.0
        self._epoch_sample_count = 0

        # Reset loss tracker for new epoch
        if self._loss_tracker is not None:
            self._loss_tracker.reset_epoch()

        return {
            'mean_loss': stats.mean_loss,
            'mean_per_sample_loss': stats.mean_per_sample_loss,
            'hard_examples_fraction': stats.hard_examples_fraction,
            'mining_active': stats.mining_active,
            'tracked_samples': stats.tracked_samples,
            'curriculum_lr_scale': stats.curriculum_lr_scale,
            'epoch': self._epoch - 1,
            'total_samples_seen': self._total_samples_seen,
        }

    def get_hard_indices(self, num_samples: int) -> np.ndarray:
        """
        Get indices of hard examples for focused training.

        Useful for creating mixed batches or sampling hard examples.

        Args:
            num_samples: Number of hard example indices to return

        Returns:
            Array of dataset indices
        """
        if self._miner is None:
            return np.array([], dtype=np.int64)
        return self._miner.get_hard_indices(num_samples)

    def get_hardest_samples(self, n: int = 100) -> list[tuple[int, float]]:
        """
        Get the n samples with highest loss.

        Useful for debugging and data quality analysis.

        Args:
            n: Number of samples to return

        Returns:
            List of (sample_idx, loss) tuples
        """
        if self._loss_tracker is None:
            return []
        return self._loss_tracker.get_hardest_samples(n)

    def reset(self) -> None:
        """Reset all internal state."""
        self._epoch = 0
        self._total_samples_seen = 0
        self._epoch_loss_sum = 0.0
        self._epoch_sample_count = 0

        if self._miner is not None:
            self._miner.reset()

        if self._loss_tracker is not None:
            self._loss_tracker.reset_epoch()

    @property
    def is_mining_active(self) -> bool:
        """Check if hard example mining is currently active."""
        if self._miner is None:
            return False
        return self._total_samples_seen >= self.config.hard_min_samples_before_mining


__all__ = [
    "FacadeConfig",
    "EpochStatistics",
    "TrainingEnhancementsFacade",
]
