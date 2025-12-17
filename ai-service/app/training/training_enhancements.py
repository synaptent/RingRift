"""
Training Enhancements for RingRift AI.

This module provides advanced training optimizations:
1. Checkpoint averaging for improved final model
2. Gradient accumulation for larger effective batch sizes
3. Data quality scoring for sample prioritization (with freshness weighting)
4. Adaptive learning rate based on Elo progress
5. Early stopping with patience
6. EWC (Elastic Weight Consolidation) for continual learning
7. Model ensemble support for self-play
8. Value head calibration automation
9. Training anomaly detection (NaN/Inf, loss spikes, gradient explosions)
10. Configurable validation intervals (step/epoch-based, adaptive)

Usage:
    from app.training.training_enhancements import (
        TrainingConfig,
        CheckpointAverager,
        GradientAccumulator,
        DataQualityScorer,
        HardExampleMiner,
        AdaptiveLRScheduler,
        WarmRestartsScheduler,
        EWCRegularizer,
        ModelEnsemble,
        EnhancedEarlyStopping,
        TrainingAnomalyDetector,
        ValidationIntervalManager,
        SeedManager,
        create_training_enhancements,
    )
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


# =============================================================================
# 0. Consolidated Training Configuration (Phase 7)
# =============================================================================


@dataclass
class TrainingConfig:
    """
    Consolidated configuration for all training enhancements.

    This dataclass provides a single point of configuration for:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Data quality scoring
    - Hard example mining
    - Anomaly detection
    - Validation intervals
    - Seed management

    Usage:
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=256,
            use_mixed_precision=True,
            validation_interval_steps=500,
        )

        # Convert to dict for create_training_enhancements
        enhancements = create_training_enhancements(model, optimizer, config.to_dict())
    """

    # === Core Training ===
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    weight_decay: float = 0.0001
    seed: Optional[int] = None

    # === Mixed Precision ===
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # "float16" or "bfloat16"

    # === Gradient Accumulation ===
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # === Learning Rate Schedule ===
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "warm_restarts"
    warmup_epochs: int = 5
    warmup_steps: int = 0
    min_lr: float = 1e-6
    max_lr: float = 0.01

    # Warm restarts (cosine annealing with restarts)
    warm_restart_t0: int = 10  # Initial restart period
    warm_restart_t_mult: int = 2  # Period multiplier after each restart
    warm_restart_eta_min: float = 1e-6

    # === Early Stopping ===
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    elo_patience: int = 10
    elo_min_improvement: float = 5.0

    # === Data Quality Scoring ===
    freshness_decay_hours: float = 24.0
    freshness_weight: float = 0.2
    sample_temperature: float = 1.0

    # === Hard Example Mining ===
    use_hard_example_mining: bool = True
    hard_example_buffer_size: int = 10000
    hard_example_fraction: float = 0.3
    hard_example_percentile: float = 80.0
    min_samples_before_mining: int = 1000

    # === Anomaly Detection ===
    loss_spike_threshold: float = 3.0
    gradient_norm_threshold: float = 100.0
    halt_on_nan: bool = True
    halt_on_spike: bool = False
    max_consecutive_anomalies: int = 5

    # === Validation ===
    validation_interval_steps: Optional[int] = 1000
    validation_interval_epochs: Optional[float] = None
    validation_subset_size: float = 1.0
    adaptive_validation_interval: bool = False

    # === Checkpointing ===
    checkpoint_interval_epochs: int = 1
    avg_checkpoints: int = 5
    keep_checkpoints_on_disk: bool = False

    # === EWC (Elastic Weight Consolidation) ===
    use_ewc: bool = False
    lambda_ewc: float = 1000.0
    ewc_num_samples: int = 1000

    # === Calibration ===
    calibration_threshold: float = 0.05
    calibration_check_interval: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for backward compatibility."""
        return {
            # Core
            'base_lr': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'seed': self.seed,

            # Mixed precision
            'use_mixed_precision': self.use_mixed_precision,
            'mixed_precision_dtype': self.mixed_precision_dtype,

            # Gradient accumulation
            'accumulation_steps': self.accumulation_steps,
            'max_grad_norm': self.max_grad_norm,

            # LR schedule
            'lr_scheduler': self.lr_scheduler,
            'warmup_epochs': self.warmup_epochs,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warm_restart_t0': self.warm_restart_t0,
            'warm_restart_t_mult': self.warm_restart_t_mult,
            'warm_restart_eta_min': self.warm_restart_eta_min,

            # Early stopping
            'patience': self.early_stopping_patience,
            'min_delta': self.early_stopping_min_delta,
            'elo_patience': self.elo_patience,
            'elo_min_improvement': self.elo_min_improvement,

            # Data quality
            'freshness_decay_hours': self.freshness_decay_hours,
            'freshness_weight': self.freshness_weight,
            'sample_temperature': self.sample_temperature,

            # Hard example mining
            'use_hard_example_mining': self.use_hard_example_mining,
            'hard_example_buffer_size': self.hard_example_buffer_size,
            'hard_example_fraction': self.hard_example_fraction,
            'hard_example_percentile': self.hard_example_percentile,
            'min_samples_before_mining': self.min_samples_before_mining,

            # Anomaly detection
            'loss_spike_threshold': self.loss_spike_threshold,
            'gradient_norm_threshold': self.gradient_norm_threshold,
            'halt_on_nan': self.halt_on_nan,
            'halt_on_spike': self.halt_on_spike,
            'max_consecutive_anomalies': self.max_consecutive_anomalies,

            # Validation
            'validation_interval_steps': self.validation_interval_steps,
            'validation_interval_epochs': self.validation_interval_epochs,
            'validation_subset_size': self.validation_subset_size,
            'adaptive_validation_interval': self.adaptive_validation_interval,

            # Checkpointing
            'checkpoint_interval_epochs': self.checkpoint_interval_epochs,
            'avg_checkpoints': self.avg_checkpoints,
            'keep_checkpoints_on_disk': self.keep_checkpoints_on_disk,

            # EWC
            'use_ewc': self.use_ewc,
            'lambda_ewc': self.lambda_ewc,
            'ewc_num_samples': self.ewc_num_samples,

            # Calibration
            'calibration_threshold': self.calibration_threshold,
            'calibration_check_interval': self.calibration_check_interval,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Map dictionary keys to dataclass fields
        mapping = {
            'base_lr': 'learning_rate',
            'patience': 'early_stopping_patience',
            'min_delta': 'early_stopping_min_delta',
        }

        kwargs = {}
        for field_info in cls.__dataclass_fields__.values():
            name = field_info.name

            # Check for mapped name first
            dict_key = None
            for k, v in mapping.items():
                if v == name:
                    dict_key = k
                    break

            if dict_key and dict_key in config:
                kwargs[name] = config[dict_key]
            elif name in config:
                kwargs[name] = config[name]

        return cls(**kwargs)

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.learning_rate > self.max_lr:
            warnings.append(f"learning_rate ({self.learning_rate}) > max_lr ({self.max_lr})")

        if self.learning_rate < self.min_lr:
            warnings.append(f"learning_rate ({self.learning_rate}) < min_lr ({self.min_lr})")

        if self.accumulation_steps < 1:
            warnings.append("accumulation_steps must be >= 1")

        if self.hard_example_fraction < 0 or self.hard_example_fraction > 1:
            warnings.append("hard_example_fraction must be between 0 and 1")

        if self.validation_subset_size < 0.01 or self.validation_subset_size > 1:
            warnings.append("validation_subset_size must be between 0.01 and 1")

        if self.lr_scheduler == "warm_restarts" and self.warm_restart_t0 < 1:
            warnings.append("warm_restart_t0 must be >= 1")

        return warnings

    def get_effective_batch_size(self) -> int:
        """Get effective batch size with gradient accumulation."""
        return self.batch_size * self.accumulation_steps

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainingConfig:"]
        for field_info in self.__dataclass_fields__.values():
            name = field_info.name
            value = getattr(self, name)
            lines.append(f"  {name}: {value}")
        return "\n".join(lines)


# =============================================================================
# 1. Checkpoint Averaging
# =============================================================================


class CheckpointAverager:
    """
    Averages model weights from multiple checkpoints for improved performance.

    Checkpoint averaging typically provides +10-20 Elo improvement by reducing
    variance in the final model weights.

    Usage:
        averager = CheckpointAverager(num_checkpoints=5)

        # During training, save checkpoints
        for epoch in range(epochs):
            train_epoch()
            averager.add_checkpoint(model.state_dict())

        # Get averaged weights
        averaged_state = averager.get_averaged_state_dict()
        model.load_state_dict(averaged_state)
    """

    def __init__(
        self,
        num_checkpoints: int = 5,
        checkpoint_dir: Optional[Path] = None,
        keep_on_disk: bool = False,
    ):
        """
        Args:
            num_checkpoints: Number of recent checkpoints to average
            checkpoint_dir: Directory to save checkpoints (if keep_on_disk=True)
            keep_on_disk: Save checkpoints to disk to save memory
        """
        self.num_checkpoints = num_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.keep_on_disk = keep_on_disk
        self._checkpoints: deque = deque(maxlen=num_checkpoints)
        self._checkpoint_paths: deque = deque(maxlen=num_checkpoints)

        if keep_on_disk and checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        epoch: Optional[int] = None,
    ) -> None:
        """Add a checkpoint to the averaging queue."""
        if self.keep_on_disk and self.checkpoint_dir:
            # Save to disk
            path = self.checkpoint_dir / f"avg_ckpt_{epoch or len(self._checkpoint_paths)}.pt"
            torch.save(state_dict, path)

            # Remove old file if queue is full
            if len(self._checkpoint_paths) == self.num_checkpoints:
                old_path = self._checkpoint_paths[0]
                if old_path.exists():
                    old_path.unlink()

            self._checkpoint_paths.append(path)
        else:
            # Keep in memory (deep copy to avoid reference issues)
            self._checkpoints.append(copy.deepcopy(state_dict))

    def get_averaged_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Compute the average of all stored checkpoints.

        Returns:
            Averaged state dict
        """
        if self.keep_on_disk:
            checkpoints = [
                torch.load(p, weights_only=True)
                for p in self._checkpoint_paths
                if p.exists()
            ]
        else:
            checkpoints = list(self._checkpoints)

        if not checkpoints:
            raise ValueError("No checkpoints available for averaging")

        # Initialize with first checkpoint
        averaged = {}
        for key in checkpoints[0]:
            averaged[key] = checkpoints[0][key].clone().float()

        # Add remaining checkpoints
        for ckpt in checkpoints[1:]:
            for key in averaged:
                averaged[key] += ckpt[key].float()

        # Divide by number of checkpoints
        num_ckpts = len(checkpoints)
        for key in averaged:
            averaged[key] /= num_ckpts
            # Restore original dtype
            averaged[key] = averaged[key].to(checkpoints[0][key].dtype)

        logger.info(f"Averaged {num_ckpts} checkpoints")
        return averaged

    def cleanup(self) -> None:
        """Remove checkpoint files from disk."""
        if self.keep_on_disk:
            for path in self._checkpoint_paths:
                if path.exists():
                    path.unlink()
            self._checkpoint_paths.clear()
        self._checkpoints.clear()

    @property
    def num_stored(self) -> int:
        """Number of checkpoints currently stored."""
        if self.keep_on_disk:
            return len(self._checkpoint_paths)
        return len(self._checkpoints)


def average_checkpoints(
    checkpoint_paths: List[Union[str, Path]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Average model weights from multiple checkpoint files.

    Args:
        checkpoint_paths: List of paths to checkpoint files
        device: Device to load checkpoints to

    Returns:
        Averaged state dict
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    averaged = None
    num_ckpts = 0

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        if averaged is None:
            averaged = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in averaged:
                averaged[k] += state_dict[k].float()
        num_ckpts += 1

    for k in averaged:
        averaged[k] /= num_ckpts

    return averaged


# =============================================================================
# 2. Gradient Accumulation
# =============================================================================


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
        max_grad_norm: Optional[float] = 1.0,
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
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
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


# =============================================================================
# 3. Data Quality Scoring
# =============================================================================


@dataclass
class GameQualityScore:
    """Quality score for a training game."""
    game_id: str
    total_score: float
    length_score: float
    elo_score: float
    diversity_score: float
    decisive_score: float
    freshness_score: float = 1.0  # Phase 7: Time-based freshness (1.0 = newest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'game_id': self.game_id,
            'total_score': self.total_score,
            'length_score': self.length_score,
            'elo_score': self.elo_score,
            'diversity_score': self.diversity_score,
            'decisive_score': self.decisive_score,
            'freshness_score': self.freshness_score,
        }


class DataQualityScorer:
    """
    Scores training data quality for sample prioritization.

    Higher quality games get higher sampling weights during training.

    Quality factors:
    - Game length (avoid very short/long games)
    - Elo differential between players
    - Move diversity/entropy
    - Decisive vs drawn games
    - Freshness (Phase 7): Exponential decay based on game age
    """

    def __init__(
        self,
        min_game_length: int = 20,
        max_game_length: int = 500,
        optimal_game_length: int = 100,
        max_elo_diff: float = 400.0,
        decisive_bonus: float = 1.2,
        draw_penalty: float = 0.8,
        freshness_decay_hours: float = 24.0,
        freshness_weight: float = 0.2,
    ):
        """
        Args:
            min_game_length: Minimum acceptable game length
            max_game_length: Maximum acceptable game length
            optimal_game_length: Optimal game length for highest score
            max_elo_diff: Maximum Elo differential for scoring
            decisive_bonus: Bonus multiplier for decisive games
            draw_penalty: Penalty multiplier for drawn games
            freshness_decay_hours: Half-life for freshness decay (default 24h)
            freshness_weight: Weight of freshness in total score (default 0.2)
        """
        self.min_game_length = min_game_length
        self.max_game_length = max_game_length
        self.optimal_game_length = optimal_game_length
        self.max_elo_diff = max_elo_diff
        self.decisive_bonus = decisive_bonus
        self.draw_penalty = draw_penalty
        self.freshness_decay_hours = freshness_decay_hours
        self.freshness_weight = freshness_weight

    def compute_freshness_score(
        self,
        game_timestamp: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> float:
        """
        Compute freshness score using exponential decay.

        Recent games get higher scores, with decay based on freshness_decay_hours.

        Args:
            game_timestamp: Unix timestamp when game was played
            current_time: Current time (default: time.time())

        Returns:
            Freshness score (0-1, where 1 = newest)
        """
        if game_timestamp is None:
            return 0.5  # Neutral if unknown

        if current_time is None:
            current_time = time.time()

        age_hours = (current_time - game_timestamp) / 3600
        if age_hours < 0:
            return 1.0  # Future timestamp = max freshness

        # Exponential decay: score = exp(-age / decay_hours)
        freshness_score = math.exp(-age_hours / self.freshness_decay_hours)
        return max(0.0, min(1.0, freshness_score))

    def score_game(
        self,
        game_id: str,
        game_length: int,
        winner: Optional[int] = None,
        elo_p1: Optional[float] = None,
        elo_p2: Optional[float] = None,
        move_entropy: Optional[float] = None,
        game_timestamp: Optional[float] = None,
    ) -> GameQualityScore:
        """
        Score a game's quality for training.

        Args:
            game_id: Unique game identifier
            game_length: Number of moves in the game
            winner: Winner (1, 2, ..., or None for draw)
            elo_p1: Elo rating of player 1
            elo_p2: Elo rating of player 2
            move_entropy: Average move entropy (policy diversity)
            game_timestamp: Unix timestamp when game was played (Phase 7: freshness)

        Returns:
            GameQualityScore with component scores
        """
        # Length score: Gaussian around optimal length
        if game_length < self.min_game_length:
            length_score = 0.5 * (game_length / self.min_game_length)
        elif game_length > self.max_game_length:
            length_score = 0.5 * (self.max_game_length / game_length)
        else:
            # Gaussian centered at optimal
            sigma = (self.max_game_length - self.min_game_length) / 4
            diff = game_length - self.optimal_game_length
            length_score = math.exp(-(diff ** 2) / (2 * sigma ** 2))

        # Elo score: Prefer balanced games
        if elo_p1 is not None and elo_p2 is not None:
            elo_diff = abs(elo_p1 - elo_p2)
            elo_score = max(0, 1 - (elo_diff / self.max_elo_diff))
        else:
            elo_score = 0.5  # Neutral if unknown

        # Diversity score: Higher entropy = more diverse moves
        if move_entropy is not None:
            # Normalize entropy (typical range 0-4)
            diversity_score = min(1.0, move_entropy / 3.0)
        else:
            diversity_score = 0.5  # Neutral if unknown

        # Decisive score: Bonus for wins, penalty for draws
        if winner is not None and winner > 0:
            decisive_score = self.decisive_bonus
        else:
            decisive_score = self.draw_penalty

        # Phase 7: Freshness score (exponential decay)
        freshness_score = self.compute_freshness_score(game_timestamp)

        # Total score (weighted combination, redistributed with freshness)
        # Original weights: 0.3 length, 0.2 elo, 0.2 diversity, 0.3 decisive
        # New weights: scaled down to make room for freshness
        remaining_weight = 1.0 - self.freshness_weight
        total_score = (
            (0.3 * remaining_weight) * length_score +
            (0.2 * remaining_weight) * elo_score +
            (0.2 * remaining_weight) * diversity_score +
            (0.3 * remaining_weight) * decisive_score +
            self.freshness_weight * freshness_score
        )

        return GameQualityScore(
            game_id=game_id,
            total_score=total_score,
            length_score=length_score,
            elo_score=elo_score,
            diversity_score=diversity_score,
            decisive_score=decisive_score,
            freshness_score=freshness_score,
        )

    def compute_sample_weights(
        self,
        scores: List[GameQualityScore],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Compute sampling weights from quality scores.

        Args:
            scores: List of quality scores
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            Normalized sampling weights
        """
        raw_scores = np.array([s.total_score for s in scores])

        # Apply temperature and normalize
        if temperature > 0:
            scaled = raw_scores / temperature
            weights = np.exp(scaled - np.max(scaled))  # Numerically stable softmax
            weights /= weights.sum()
        else:
            weights = np.ones(len(scores)) / len(scores)

        return weights


class QualityWeightedSampler(Sampler):
    """
    PyTorch sampler that weights samples by quality scores.

    Usage:
        scorer = DataQualityScorer()
        scores = [scorer.score_game(...) for game in games]
        sampler = QualityWeightedSampler(scores)
        dataloader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
        self,
        scores: List[GameQualityScore],
        num_samples: Optional[int] = None,
        replacement: bool = True,
        temperature: float = 1.0,
    ):
        self.scores = scores
        self.num_samples = num_samples or len(scores)
        self.replacement = replacement

        scorer = DataQualityScorer()
        self.weights = torch.from_numpy(
            scorer.compute_sample_weights(scores, temperature)
        ).double()

    def __iter__(self):
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


# =============================================================================
# 3b. Hard Example Mining (Phase 7)
# =============================================================================


@dataclass
class HardExample:
    """A hard example identified during training."""
    index: int
    loss: float
    uncertainty: float
    times_sampled: int = 1
    last_seen_step: int = 0


class HardExampleMiner:
    """
    Identifies and prioritizes hard examples for training.

    Hard examples are samples where the model:
    - Has high loss (prediction error)
    - Has high uncertainty (low confidence)
    - Consistently performs poorly

    This implements curriculum learning by focusing on difficult cases
    while maintaining diversity to prevent overfitting.

    Usage:
        miner = HardExampleMiner(buffer_size=10000, hard_fraction=0.3)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            losses = compute_per_sample_loss(outputs, targets)

            # Record losses for mining
            batch_indices = get_batch_indices(batch_idx, batch_size)
            miner.record_batch(batch_indices, losses)

            # Get indices of hard examples to emphasize
            hard_indices = miner.get_hard_indices(num_samples=batch_size)

            # Optionally create a hard example batch
            if step % hard_batch_interval == 0:
                hard_batch = dataset[hard_indices]
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        hard_fraction: float = 0.3,
        loss_threshold_percentile: float = 80.0,
        uncertainty_weight: float = 0.3,
        decay_rate: float = 0.99,
        min_samples_before_mining: int = 1000,
        max_times_sampled: int = 10,
    ):
        """
        Args:
            buffer_size: Maximum number of examples to track
            hard_fraction: Fraction of hard examples in sampled batches
            loss_threshold_percentile: Percentile above which examples are "hard"
            uncertainty_weight: Weight of uncertainty vs loss in hardness score
            decay_rate: Decay factor for old hardness scores (per step)
            min_samples_before_mining: Minimum samples seen before mining starts
            max_times_sampled: Cap on how many times a hard example can be sampled
        """
        self.buffer_size = buffer_size
        self.hard_fraction = hard_fraction
        self.loss_threshold_percentile = loss_threshold_percentile
        self.uncertainty_weight = uncertainty_weight
        self.decay_rate = decay_rate
        self.min_samples_before_mining = min_samples_before_mining
        self.max_times_sampled = max_times_sampled

        # Track examples: index -> HardExample
        self._examples: Dict[int, HardExample] = {}
        self._total_samples_seen = 0
        self._current_step = 0
        self._loss_history: deque = deque(maxlen=10000)

    def record_batch(
        self,
        indices: Union[List[int], np.ndarray, torch.Tensor],
        losses: Union[List[float], np.ndarray, torch.Tensor],
        uncertainties: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Record losses and uncertainties for a batch of examples.

        Args:
            indices: Dataset indices for the batch
            losses: Per-sample losses
            uncertainties: Per-sample uncertainties (e.g., entropy of policy)
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        if uncertainties is not None and isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.detach().cpu().numpy()

        indices = np.asarray(indices).flatten()
        losses = np.asarray(losses).flatten()

        if uncertainties is None:
            uncertainties = np.zeros_like(losses)
        else:
            uncertainties = np.asarray(uncertainties).flatten()

        self._current_step += 1

        for idx, loss, unc in zip(indices, losses, uncertainties):
            idx = int(idx)
            self._loss_history.append(loss)

            if idx in self._examples:
                # Update existing example with exponential moving average
                ex = self._examples[idx]
                ex.loss = 0.7 * ex.loss + 0.3 * loss
                ex.uncertainty = 0.7 * ex.uncertainty + 0.3 * unc
                ex.times_sampled += 1
                ex.last_seen_step = self._current_step
            else:
                # Add new example
                self._examples[idx] = HardExample(
                    index=idx,
                    loss=loss,
                    uncertainty=unc,
                    times_sampled=1,
                    last_seen_step=self._current_step,
                )

        self._total_samples_seen += len(indices)

        # Prune buffer if too large
        if len(self._examples) > self.buffer_size:
            self._prune_buffer()

    def _prune_buffer(self) -> None:
        """Remove least hard examples to maintain buffer size."""
        if len(self._examples) <= self.buffer_size:
            return

        # Sort by hardness score and keep top buffer_size
        examples = list(self._examples.values())
        examples.sort(key=lambda e: self._compute_hardness(e), reverse=True)

        # Keep hardest examples
        keep_indices = {e.index for e in examples[:self.buffer_size]}
        self._examples = {
            idx: ex for idx, ex in self._examples.items()
            if idx in keep_indices
        }

    def _compute_hardness(self, example: HardExample) -> float:
        """Compute hardness score for an example."""
        # Decay based on staleness
        staleness = self._current_step - example.last_seen_step
        decay = self.decay_rate ** staleness

        # Penalize over-sampled examples
        sample_penalty = 1.0 / (1.0 + example.times_sampled / self.max_times_sampled)

        # Combine loss and uncertainty
        hardness = (
            (1 - self.uncertainty_weight) * example.loss +
            self.uncertainty_weight * example.uncertainty
        )

        return hardness * decay * sample_penalty

    def get_hard_indices(
        self,
        num_samples: int,
        return_scores: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get indices of hard examples for focused training.

        Args:
            num_samples: Number of indices to return
            return_scores: Also return hardness scores

        Returns:
            Array of indices (and optionally scores)
        """
        if self._total_samples_seen < self.min_samples_before_mining:
            # Not enough data yet - return empty
            if return_scores:
                return np.array([], dtype=np.int64), np.array([])
            return np.array([], dtype=np.int64)

        if not self._examples:
            if return_scores:
                return np.array([], dtype=np.int64), np.array([])
            return np.array([], dtype=np.int64)

        # Compute hardness for all examples
        examples = list(self._examples.values())
        hardness_scores = np.array([self._compute_hardness(e) for e in examples])
        indices = np.array([e.index for e in examples])

        # Determine threshold
        threshold = np.percentile(hardness_scores, self.loss_threshold_percentile)
        hard_mask = hardness_scores >= threshold

        hard_indices = indices[hard_mask]
        hard_scores = hardness_scores[hard_mask]

        # Sample from hard examples (weighted by score)
        num_to_sample = min(num_samples, len(hard_indices))
        if num_to_sample == 0:
            if return_scores:
                return np.array([], dtype=np.int64), np.array([])
            return np.array([], dtype=np.int64)

        # Weighted sampling
        probs = hard_scores / hard_scores.sum()
        sampled_positions = np.random.choice(
            len(hard_indices),
            size=num_to_sample,
            replace=False,
            p=probs,
        )

        result_indices = hard_indices[sampled_positions]
        result_scores = hard_scores[sampled_positions]

        if return_scores:
            return result_indices, result_scores
        return result_indices

    def get_sample_weights(
        self,
        indices: Union[List[int], np.ndarray],
        base_weight: float = 1.0,
        hard_weight: float = 2.0,
    ) -> np.ndarray:
        """
        Get sampling weights for a batch, upweighting hard examples.

        Args:
            indices: Dataset indices to get weights for
            base_weight: Weight for normal examples
            hard_weight: Weight for hard examples

        Returns:
            Array of weights for each index
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices)

        weights = np.full(len(indices), base_weight)

        if self._total_samples_seen < self.min_samples_before_mining:
            return weights

        # Get hardness threshold
        if len(self._loss_history) > 100:
            threshold = np.percentile(list(self._loss_history), self.loss_threshold_percentile)
        else:
            return weights

        # Upweight hard examples
        for i, idx in enumerate(indices):
            if idx in self._examples:
                if self._examples[idx].loss >= threshold:
                    weights[i] = hard_weight

        return weights

    def create_mixed_batch_indices(
        self,
        batch_size: int,
        all_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Create a batch mixing random and hard examples.

        Args:
            batch_size: Total batch size
            all_indices: All available dataset indices

        Returns:
            Mixed batch of indices
        """
        num_hard = int(batch_size * self.hard_fraction)
        num_random = batch_size - num_hard

        # Get hard examples
        hard_indices = self.get_hard_indices(num_hard)

        # Get random examples (excluding already selected hard ones)
        hard_set = set(hard_indices)
        available = np.array([i for i in all_indices if i not in hard_set])

        if len(available) >= num_random:
            random_indices = np.random.choice(available, size=num_random, replace=False)
        else:
            random_indices = available

        # Combine and shuffle
        batch_indices = np.concatenate([hard_indices, random_indices])
        np.random.shuffle(batch_indices)

        return batch_indices

    def get_statistics(self) -> Dict[str, Any]:
        """Get mining statistics."""
        if not self._examples:
            return {
                'total_samples_seen': self._total_samples_seen,
                'tracked_examples': 0,
                'mining_active': False,
            }

        losses = [e.loss for e in self._examples.values()]
        times_sampled = [e.times_sampled for e in self._examples.values()]

        return {
            'total_samples_seen': self._total_samples_seen,
            'tracked_examples': len(self._examples),
            'mining_active': self._total_samples_seen >= self.min_samples_before_mining,
            'mean_loss': np.mean(losses),
            'max_loss': np.max(losses),
            'loss_p90': np.percentile(losses, 90),
            'mean_times_sampled': np.mean(times_sampled),
            'max_times_sampled': np.max(times_sampled),
        }

    def reset(self) -> None:
        """Reset miner state."""
        self._examples.clear()
        self._total_samples_seen = 0
        self._current_step = 0
        self._loss_history.clear()

    # =========================================================================
    # Backwards Compatibility Methods (for drop-in replacement of train_nnue.py version)
    # =========================================================================

    def update_errors(
        self,
        indices: Union[List[int], np.ndarray],
        errors: Union[List[float], np.ndarray],
    ) -> None:
        """
        Update error history for given samples (backwards compatible).

        This is an alias for record_batch() to maintain compatibility with
        the train_nnue.py HardExampleMiner implementation.

        Args:
            indices: Dataset indices for the batch
            errors: Per-sample errors (treated as losses)
        """
        self.record_batch(indices, errors, uncertainties=None)

    def get_all_sample_weights(
        self,
        dataset_size: int,
        min_weight: float = 0.5,
        max_weight: float = 3.0,
    ) -> np.ndarray:
        """
        Compute sample weights for the entire dataset (backwards compatible).

        This method returns weights for all samples in the dataset, compatible
        with the train_nnue.py HardExampleMiner.get_sample_weights() method.

        Args:
            dataset_size: Total size of the dataset
            min_weight: Minimum weight for easy samples
            max_weight: Maximum weight for hard samples

        Returns:
            Array of weights for all samples
        """
        weights = np.full(dataset_size, min_weight, dtype=np.float32)

        if self._total_samples_seen < self.min_samples_before_mining:
            return np.ones(dataset_size, dtype=np.float32)

        if len(self._loss_history) < 100:
            return np.ones(dataset_size, dtype=np.float32)

        # Get hardness threshold
        threshold = np.percentile(list(self._loss_history), self.loss_threshold_percentile)

        # Update weights for tracked examples
        for idx, example in self._examples.items():
            if idx < dataset_size:
                # Scale weight based on hardness
                if example.loss >= threshold:
                    # Hard example - higher weight
                    hardness = min(1.0, (example.loss - threshold) / threshold if threshold > 0 else 0)
                    weights[idx] = min_weight + hardness * (max_weight - min_weight)
                else:
                    weights[idx] = min_weight

        return weights

    def get_stats(self) -> Dict[str, Any]:
        """
        Get mining statistics (backwards compatible alias for get_statistics).
        """
        stats = self.get_statistics()
        # Map to train_nnue.py expected format
        return {
            'seen_samples': stats.get('total_samples_seen', 0),
            'seen_ratio': stats.get('tracked_examples', 0) / max(1, self.buffer_size),
            'mean_error': stats.get('mean_loss', 0),
            'max_error': stats.get('max_loss', 0),
            'mining_active': stats.get('mining_active', False),
        }


# =============================================================================
# 4. Adaptive Learning Rate
# =============================================================================


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
        current_elo: Optional[float] = None,
        val_loss: Optional[float] = None,
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


# =============================================================================
# 4a. Warm Restarts Learning Rate Schedule (Phase 7)
# =============================================================================


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
        eta_max: Optional[float] = None,
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
        self._restart_epochs: List[int] = [0]

    def step(self, epoch: Optional[int] = None) -> float:
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

    def get_last_lr(self) -> List[float]:
        """Get last computed learning rate for each param group."""
        return [self._current_lr] * len(self.optimizer.param_groups)

    def state_dict(self) -> Dict[str, Any]:
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
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

    def get_schedule_info(self) -> Dict[str, Any]:
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


# =============================================================================
# 4b. Training Anomaly Detection (Phase 7)
# =============================================================================


@dataclass
class AnomalyEvent:
    """Record of a training anomaly event."""
    timestamp: float
    step: int
    anomaly_type: str  # "nan", "inf", "loss_spike", "gradient_explosion"
    value: float
    threshold: float
    message: str


class TrainingAnomalyDetector:
    """
    Detects and handles training anomalies in real-time.

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
                # Handle anomaly (skip batch, halt, etc.)
                continue

            loss.backward()

            # Check gradients
            grad_norm = compute_grad_norm(model)
            if detector.check_gradient_norm(grad_norm, step):
                # Handle gradient explosion
                optimizer.zero_grad()
                continue
    """

    def __init__(
        self,
        loss_spike_threshold: float = 3.0,
        gradient_norm_threshold: float = 100.0,
        loss_window_size: int = 100,
        halt_on_nan: bool = True,
        halt_on_spike: bool = False,
        halt_on_gradient_explosion: bool = False,
        max_consecutive_anomalies: int = 5,
    ):
        """
        Args:
            loss_spike_threshold: Standard deviations above mean to trigger spike
            gradient_norm_threshold: Max gradient norm before explosion detection
            loss_window_size: Rolling window size for loss statistics
            halt_on_nan: Raise exception on NaN/Inf detection
            halt_on_spike: Raise exception on loss spike
            halt_on_gradient_explosion: Raise exception on gradient explosion
            max_consecutive_anomalies: Max consecutive anomalies before halt
        """
        self.loss_spike_threshold = loss_spike_threshold
        self.gradient_norm_threshold = gradient_norm_threshold
        self.loss_window_size = loss_window_size
        self.halt_on_nan = halt_on_nan
        self.halt_on_spike = halt_on_spike
        self.halt_on_gradient_explosion = halt_on_gradient_explosion
        self.max_consecutive_anomalies = max_consecutive_anomalies

        self._loss_history: deque = deque(maxlen=loss_window_size)
        self._events: List[AnomalyEvent] = []
        self._consecutive_anomalies = 0
        self._total_anomalies = 0
        self._halted = False

    def check_loss(self, loss: float, step: int) -> bool:
        """
        Check loss value for anomalies.

        Args:
            loss: Current loss value
            step: Current training step

        Returns:
            True if anomaly detected, False otherwise

        Raises:
            RuntimeError if halt_on_nan/halt_on_spike is True and anomaly detected
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
        """
        Check gradient norm for explosion.

        Args:
            grad_norm: Current gradient norm
            step: Current training step

        Returns:
            True if anomaly detected, False otherwise

        Raises:
            RuntimeError if halt_on_gradient_explosion is True and anomaly detected
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

    def get_events(self) -> List[AnomalyEvent]:
        """Get all recorded anomaly events."""
        return self._events.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        type_counts = {}
        for event in self._events:
            type_counts[event.anomaly_type] = type_counts.get(event.anomaly_type, 0) + 1

        return {
            'total_anomalies': self._total_anomalies,
            'consecutive_anomalies': self._consecutive_anomalies,
            'halted': self._halted,
            'anomaly_types': type_counts,
            'recent_events': [
                {
                    'step': e.step,
                    'type': e.anomaly_type,
                    'value': e.value,
                    'message': e.message,
                }
                for e in self._events[-10:]  # Last 10 events
            ],
        }


# =============================================================================
# 4c. Configurable Validation Intervals (Phase 7)
# =============================================================================


@dataclass
class ValidationResult:
    """Result from a validation run."""
    step: int
    epoch: int
    val_loss: float
    val_metrics: Dict[str, float]
    samples_validated: int
    duration_seconds: float
    is_improvement: bool = False


class ValidationIntervalManager:
    """
    Manages configurable validation during training.

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
        validation_fn: Optional[Callable[[nn.Module], Tuple[float, Dict[str, float]]]] = None,
        interval_steps: int = 1000,
        interval_epochs: Optional[float] = None,
        subset_size: float = 1.0,
        adaptive_interval: bool = False,
        min_interval_steps: int = 100,
        max_interval_steps: int = 10000,
        warmup_steps: int = 0,
        track_best: bool = True,
    ):
        """
        Args:
            validation_fn: Function that takes model and returns (loss, metrics_dict)
            interval_steps: Validate every N training steps
            interval_epochs: Alternative: validate every N epochs (overrides interval_steps)
            subset_size: Fraction of validation data to use (0-1)
            adaptive_interval: Adjust interval based on loss variance
            min_interval_steps: Minimum interval when adaptive
            max_interval_steps: Maximum interval when adaptive
            warmup_steps: Skip validation for first N steps
            track_best: Track and report best validation loss
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
        self._results: List[ValidationResult] = []
        self._best_loss = float('inf')
        self._best_step = 0
        self._loss_history: deque = deque(maxlen=10)

    def should_validate(self, step: int, epoch: int = 0, epoch_fraction: float = 0.0) -> bool:
        """
        Check if validation should be performed.

        Args:
            step: Current training step
            epoch: Current epoch number
            epoch_fraction: Fraction through current epoch (0-1)

        Returns:
            True if validation should be performed
        """
        # Skip during warmup
        if step < self.warmup_steps:
            return False

        # Epoch-based interval
        if self.interval_epochs is not None:
            current_epoch_float = epoch + epoch_fraction
            if current_epoch_float - self._last_val_epoch >= self.interval_epochs:
                return True
            return False

        # Step-based interval
        return (step - self._last_val_step) >= self._current_interval

    def validate(
        self,
        model: nn.Module,
        step: int,
        epoch: int = 0,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> ValidationResult:
        """
        Perform validation and record results.

        Args:
            model: Model to validate
            step: Current training step
            epoch: Current epoch
            val_loader: Optional validation dataloader (if validation_fn not set)
            device: Device for validation

        Returns:
            ValidationResult with loss and metrics
        """
        start_time = time.time()

        if self.validation_fn is not None:
            # Use provided validation function
            val_loss, val_metrics = self.validation_fn(model)
            samples = int(val_metrics.get('samples', 0))
        elif val_loader is not None:
            # Default validation loop
            val_loss, val_metrics, samples = self._default_validate(
                model, val_loader, device
            )
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
        device: Optional[torch.device] = None,
    ) -> Tuple[float, Dict[str, float], int]:
        """Default validation loop."""
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        total_loss = 0.0
        total_samples = 0
        metrics = {}

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

        avg_loss = total_loss / max(1, total_samples)
        metrics['samples'] = total_samples
        metrics['batches'] = min(batches_to_use, i + 1)

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
                self.min_interval_steps,
                int(self._current_interval * 0.8)
            )
        elif cv < 0.02:  # Low variance
            self._current_interval = min(
                self.max_interval_steps,
                int(self._current_interval * 1.2)
            )

    def get_best(self) -> Tuple[float, int]:
        """Get best validation loss and step."""
        return self._best_loss, self._best_step

    def get_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self._results.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if not self._results:
            return {'num_validations': 0}

        losses = [r.val_loss for r in self._results]
        durations = [r.duration_seconds for r in self._results]

        return {
            'num_validations': len(self._results),
            'best_loss': self._best_loss,
            'best_step': self._best_step,
            'latest_loss': losses[-1],
            'mean_loss': np.mean(losses),
            'mean_duration': np.mean(durations),
            'current_interval': self._current_interval,
            'improvements': sum(1 for r in self._results if r.is_improvement),
        }

    def reset(self) -> None:
        """Reset validation state for new training run."""
        self._last_val_step = -self.interval_steps
        self._last_val_epoch = -1.0
        self._current_interval = self.interval_steps
        self._results.clear()
        self._best_loss = float('inf')
        self._best_step = 0
        self._loss_history.clear()


# =============================================================================
# 5. Enhanced Early Stopping
# =============================================================================


class EnhancedEarlyStopping:
    """
    Enhanced early stopping with multiple criteria and plateau detection.

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
        elo_patience: Optional[int] = None,
        elo_min_improvement: float = 5.0,
        mode: str = 'min',
        restore_best: bool = True,
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            elo_patience: Separate patience for Elo (None = use patience)
            elo_min_improvement: Minimum Elo improvement to reset counter
            mode: 'min' for loss, 'max' for accuracy
            restore_best: Whether to restore best model on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.elo_patience = elo_patience or patience
        self.elo_min_improvement = elo_min_improvement
        self.mode = mode
        self.restore_best = restore_best

        self.best_loss = float('inf') if mode == 'min' else float('-inf')
        self.best_elo = float('-inf')
        self.loss_counter = 0
        self.elo_counter = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = 0
        self._stopped = False

    def should_stop(
        self,
        val_loss: Optional[float] = None,
        current_elo: Optional[float] = None,
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Validation loss
            current_elo: Current Elo rating
            model: Model to save best state from
            epoch: Current epoch number

        Returns:
            True if training should stop
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

        # Save best model state
        if improved and model is not None:
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch

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
        if self.mode == 'min':
            return current < best - self.min_delta
        return current > best + self.min_delta

    def restore_best_model(self, model: nn.Module) -> bool:
        """
        Restore model to best state.

        Returns:
            True if restoration was successful
        """
        if self.best_state is not None and self.restore_best:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored model to best state from epoch {self.best_epoch}")
            return True
        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_loss = float('inf') if self.mode == 'min' else float('-inf')
        self.best_elo = float('-inf')
        self.loss_counter = 0
        self.elo_counter = 0
        self.best_state = None
        self.best_epoch = 0
        self._stopped = False

    # =========================================================================
    # Backwards Compatibility Methods (for drop-in replacement of basic EarlyStopping)
    # =========================================================================

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop (backwards compatible interface).

        This allows EnhancedEarlyStopping to be used as a drop-in replacement
        for the basic EarlyStopping class in train.py.

        Args:
            val_loss: Current validation loss
            model: Model to save state from if this is best so far

        Returns:
            True if training should stop, False otherwise
        """
        return self.should_stop(val_loss=val_loss, model=model)

    def restore_best_weights(self, model: nn.Module) -> None:
        """
        Restore the best weights to the model (backwards compatible alias).

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


# =============================================================================
# 6. EWC (Elastic Weight Consolidation) for Continual Learning
# =============================================================================


class EWCRegularizer:
    """
    Elastic Weight Consolidation for continual learning.

    Prevents catastrophic forgetting when training on new data by
    penalizing changes to important parameters.

    Usage:
        ewc = EWCRegularizer(model)

        # After training on task 1
        ewc.compute_fisher(dataloader_task1)

        # When training on task 2
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        normalize_fisher: bool = True,
    ):
        """
        Args:
            model: Model to apply EWC to
            lambda_ewc: Importance weight for EWC penalty
            normalize_fisher: Normalize Fisher information matrix
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.normalize_fisher = normalize_fisher

        self._fisher: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        self._computed = False

    def compute_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None,
        num_samples: int = 1000,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Compute Fisher information matrix from dataloader.

        Args:
            dataloader: DataLoader for computing Fisher
            criterion: Loss function (default: cross entropy)
            num_samples: Number of samples to use
            device: Device for computation
        """
        if device is None:
            device = next(self.model.parameters()).device

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Store optimal parameters
        self._optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher to zero
        self._fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        samples_seen = 0

        for inputs, targets in dataloader:
            if samples_seen >= num_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            # Use log-softmax for computing Fisher
            log_probs = torch.log_softmax(outputs, dim=-1)

            # Sample from output distribution
            labels = torch.distributions.Categorical(logits=outputs).sample()
            loss = -log_probs[range(len(labels)), labels].mean()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad.pow(2)

            samples_seen += len(inputs)

        # Normalize Fisher
        for name in self._fisher:
            self._fisher[name] /= samples_seen

            if self.normalize_fisher:
                max_val = self._fisher[name].max()
                if max_val > 0:
                    self._fisher[name] /= max_val

        self._computed = True
        logger.info(f"Computed Fisher information from {samples_seen} samples")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty for current model parameters.

        Args:
            model: Model to compute penalty for

        Returns:
            EWC penalty term
        """
        if not self._computed:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self._fisher and param.requires_grad:
                diff = param - self._optimal_params[name]
                penalty += (self._fisher[name] * diff.pow(2)).sum()

        return 0.5 * self.lambda_ewc * penalty

    def save_state(self, path: Union[str, Path]) -> None:
        """Save EWC state to file."""
        state = {
            'fisher': self._fisher,
            'optimal_params': self._optimal_params,
            'lambda_ewc': self.lambda_ewc,
            'computed': self._computed,
        }
        torch.save(state, path)

    def load_state(self, path: Union[str, Path]) -> None:
        """Load EWC state from file."""
        state = torch.load(path, weights_only=False)
        self._fisher = state['fisher']
        self._optimal_params = state['optimal_params']
        self.lambda_ewc = state['lambda_ewc']
        self._computed = state['computed']


# =============================================================================
# 7. Model Ensemble for Self-Play
# =============================================================================


class ModelEnsemble:
    """
    Ensemble of models for diverse self-play opponents.

    Using an ensemble for self-play provides more diverse training data
    and prevents overfitting to a single opponent's weaknesses.

    Usage:
        ensemble = ModelEnsemble(model_class=RingRiftCNN_v2)

        # Add models
        ensemble.add_model(best_model, weight=0.5)
        ensemble.add_model(previous_model, weight=0.3)
        ensemble.add_model(random_model, weight=0.2)

        # Sample opponent for self-play game
        opponent = ensemble.sample_model()
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_class: Class to instantiate models from
            model_kwargs: Arguments for model constructor
            device: Device for models
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.device = device or torch.device('cpu')

        self._models: List[nn.Module] = []
        self._weights: List[float] = []
        self._names: List[str] = []

    def add_model(
        self,
        model_or_path: Union[nn.Module, str, Path],
        weight: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Add a model to the ensemble.

        Args:
            model_or_path: Model instance or path to checkpoint
            weight: Sampling weight (higher = more likely to be chosen)
            name: Optional name for the model
        """
        if isinstance(model_or_path, (str, Path)):
            # Load from checkpoint
            model = self.model_class(**self.model_kwargs).to(self.device)
            ckpt = torch.load(model_or_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict)
            model.eval()
            model_name = name or Path(model_or_path).stem
        else:
            model = model_or_path.to(self.device)
            model.eval()
            model_name = name or f"model_{len(self._models)}"

        self._models.append(model)
        self._weights.append(weight)
        self._names.append(model_name)

        logger.info(f"Added model '{model_name}' to ensemble with weight {weight}")

    def sample_model(self) -> Tuple[nn.Module, str]:
        """
        Sample a model from the ensemble based on weights.

        Returns:
            Tuple of (model, model_name)
        """
        if not self._models:
            raise ValueError("No models in ensemble")

        # Normalize weights
        total = sum(self._weights)
        probs = [w / total for w in self._weights]

        idx = np.random.choice(len(self._models), p=probs)
        return self._models[idx], self._names[idx]

    def get_model(self, name: str) -> Optional[nn.Module]:
        """Get a specific model by name."""
        for i, n in enumerate(self._names):
            if n == name:
                return self._models[i]
        return None

    def update_weight(self, name: str, weight: float) -> None:
        """Update the weight of a specific model."""
        for i, n in enumerate(self._names):
            if n == name:
                self._weights[i] = weight
                return

    @property
    def num_models(self) -> int:
        """Number of models in ensemble."""
        return len(self._models)

    @property
    def model_names(self) -> List[str]:
        """Names of all models in ensemble."""
        return list(self._names)


# =============================================================================
# 8. Value Head Calibration Automation
# =============================================================================


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
        self._calibration_history: List[Dict[str, Any]] = []

    def add_samples(
        self,
        predictions: Union[List[float], np.ndarray, torch.Tensor],
        outcomes: Union[List[float], np.ndarray, torch.Tensor],
    ) -> None:
        """Add prediction-outcome pairs for monitoring."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(outcomes, torch.Tensor):
            outcomes = outcomes.detach().cpu().numpy()

        predictions = np.asarray(predictions).flatten()
        outcomes = np.asarray(outcomes).flatten()

        for p, o in zip(predictions, outcomes):
            self._predictions.append(p)
            self._outcomes.append(o)

    def should_recalibrate(self, epoch: Optional[int] = None) -> bool:
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
        device: Optional[torch.device] = None,
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

    def get_calibration_history(self) -> List[Dict[str, Any]]:
        """Get calibration check history."""
        return self._calibration_history


# =============================================================================
# 9. Seed Management for Reproducibility (Phase 7)
# =============================================================================


class SeedManager:
    """
    Manages random seeds for reproducible training.

    Handles seeding for:
    - Python's random module
    - NumPy's random generator
    - PyTorch (CPU and CUDA)
    - Optional: CuDNN determinism

    Usage:
        # Set global seed for reproducibility
        seed_manager = SeedManager(seed=42)
        seed_manager.set_global_seed()

        # Get worker init function for DataLoader
        dataloader = DataLoader(
            dataset,
            worker_init_fn=seed_manager.get_worker_init_fn(),
        )

        # Log seed info for experiment tracking
        print(seed_manager.get_seed_info())
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        deterministic: bool = False,
        benchmark: bool = True,
    ):
        """
        Args:
            seed: Random seed (None for random seed)
            deterministic: Enable CuDNN deterministic mode (slower but reproducible)
            benchmark: Enable CuDNN benchmark mode (faster but non-deterministic)
        """
        self.seed = seed if seed is not None else self._generate_seed()
        self.deterministic = deterministic
        self.benchmark = benchmark

        self._initial_seed = self.seed
        self._seed_history: List[Dict[str, Any]] = []

    def _generate_seed(self) -> int:
        """Generate a random seed."""
        import random
        return random.randint(0, 2**32 - 1)

    def set_global_seed(self) -> None:
        """Set seed for all random number generators."""
        import random

        # Python random
        random.seed(self.seed)

        # NumPy
        np.random.seed(self.seed)

        # PyTorch CPU
        torch.manual_seed(self.seed)

        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            # CuDNN settings
            torch.backends.cudnn.deterministic = self.deterministic
            torch.backends.cudnn.benchmark = self.benchmark and not self.deterministic

        # Record
        self._seed_history.append({
            'seed': self.seed,
            'timestamp': time.time(),
            'action': 'set_global_seed',
        })

        logger.info(
            f"Set global seed: {self.seed} "
            f"(deterministic={self.deterministic}, benchmark={self.benchmark})"
        )

    def get_worker_init_fn(self) -> Callable[[int], None]:
        """
        Get worker initialization function for DataLoader.

        Each worker gets a unique but reproducible seed based on the
        global seed and worker ID.

        Returns:
            Worker init function for DataLoader
        """
        base_seed = self.seed

        def worker_init_fn(worker_id: int) -> None:
            import random
            worker_seed = base_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return worker_init_fn

    def get_generator(self, offset: int = 0) -> torch.Generator:
        """
        Get a PyTorch Generator with reproducible seed.

        Args:
            offset: Offset to add to base seed (for different generators)

        Returns:
            Seeded torch.Generator
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + offset)
        return generator

    def fork(self, offset: int = 1) -> 'SeedManager':
        """
        Create a new SeedManager with an offset seed.

        Useful for creating reproducible but different seeds for
        different components (e.g., data augmentation vs dropout).

        Args:
            offset: Offset to add to base seed

        Returns:
            New SeedManager with offset seed
        """
        return SeedManager(
            seed=self.seed + offset,
            deterministic=self.deterministic,
            benchmark=self.benchmark,
        )

    def advance(self, steps: int = 1) -> None:
        """
        Advance the seed by a number of steps.

        Useful for resuming training with a different seed progression.

        Args:
            steps: Number of steps to advance
        """
        self.seed = (self.seed + steps) % (2**32)
        self._seed_history.append({
            'seed': self.seed,
            'timestamp': time.time(),
            'action': f'advance({steps})',
        })

    def get_seed_info(self) -> Dict[str, Any]:
        """Get seed information for experiment tracking."""
        return {
            'initial_seed': self._initial_seed,
            'current_seed': self.seed,
            'deterministic': self.deterministic,
            'benchmark': self.benchmark,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__,
        }

    def save_state(self) -> Dict[str, Any]:
        """Save RNG states for checkpointing."""
        import random

        state = {
            'seed': self.seed,
            'initial_seed': self._initial_seed,
            'python_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'seed_history': self._seed_history,
        }

        if torch.cuda.is_available():
            state['cuda_state'] = torch.cuda.get_rng_state_all()

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load RNG states from checkpoint."""
        import random

        self.seed = state['seed']
        self._initial_seed = state['initial_seed']
        self._seed_history = state.get('seed_history', [])

        random.setstate(state['python_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])

        if torch.cuda.is_available() and 'cuda_state' in state:
            torch.cuda.set_rng_state_all(state['cuda_state'])

        logger.info(f"Loaded RNG state from checkpoint (seed={self.seed})")

    def __repr__(self) -> str:
        return (
            f"SeedManager(seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"benchmark={self.benchmark})"
        )


def set_reproducible_seed(seed: int, deterministic: bool = True) -> SeedManager:
    """
    Convenience function to set a reproducible seed.

    Args:
        seed: Random seed to use
        deterministic: Enable full determinism (slower)

    Returns:
        SeedManager instance
    """
    manager = SeedManager(seed=seed, deterministic=deterministic, benchmark=False)
    manager.set_global_seed()
    return manager


# =============================================================================
# Convenience Functions
# =============================================================================


def create_training_enhancements(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: Optional[Dict[str, Any]] = None,
    validation_fn: Optional[Callable[[nn.Module], Tuple[float, Dict[str, float]]]] = None,
) -> Dict[str, Any]:
    """
    Create a suite of training enhancements with default configuration.

    Args:
        model: Model to enhance training for
        optimizer: Optimizer to use
        config: Optional configuration overrides
        validation_fn: Optional validation function for ValidationIntervalManager

    Returns:
        Dictionary of enhancement objects
    """
    config = config or {}

    enhancements = {
        'checkpoint_averager': CheckpointAverager(
            num_checkpoints=config.get('avg_checkpoints', 5),
        ),
        'gradient_accumulator': GradientAccumulator(
            accumulation_steps=config.get('accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
        ),
        'quality_scorer': DataQualityScorer(
            freshness_decay_hours=config.get('freshness_decay_hours', 24.0),
            freshness_weight=config.get('freshness_weight', 0.2),
        ),
        'adaptive_lr': AdaptiveLRScheduler(
            optimizer=optimizer,
            base_lr=config.get('base_lr', 0.001),
        ),
        'early_stopping': EnhancedEarlyStopping(
            patience=config.get('patience', 10),
        ),
        'ewc': EWCRegularizer(
            model=model,
            lambda_ewc=config.get('lambda_ewc', 1000.0),
        ),
        'calibration': CalibrationAutomation(
            deviation_threshold=config.get('calibration_threshold', 0.05),
        ),
        'anomaly_detector': TrainingAnomalyDetector(
            loss_spike_threshold=config.get('loss_spike_threshold', 3.0),
            gradient_norm_threshold=config.get('gradient_norm_threshold', 100.0),
            halt_on_nan=config.get('halt_on_nan', True),
        ),
        'validation_manager': ValidationIntervalManager(
            validation_fn=validation_fn,
            interval_steps=config.get('validation_interval_steps', 1000),
            interval_epochs=config.get('validation_interval_epochs', None),
            subset_size=config.get('validation_subset_size', 1.0),
            adaptive_interval=config.get('adaptive_validation_interval', False),
        ),
        'hard_example_miner': HardExampleMiner(
            buffer_size=config.get('hard_example_buffer_size', 10000),
            hard_fraction=config.get('hard_example_fraction', 0.3),
            loss_threshold_percentile=config.get('hard_example_percentile', 80.0),
            min_samples_before_mining=config.get('min_samples_before_mining', 1000),
        ),
        'seed_manager': SeedManager(
            seed=config.get('seed'),
            deterministic=config.get('deterministic', False),
            benchmark=config.get('benchmark', True),
        ),
    }

    # Optionally add warm restarts scheduler
    if config.get('lr_scheduler') == 'warm_restarts':
        enhancements['warm_restarts_scheduler'] = WarmRestartsScheduler(
            optimizer=optimizer,
            T_0=config.get('warm_restart_t0', 10),
            T_mult=config.get('warm_restart_t_mult', 2),
            eta_min=config.get('warm_restart_eta_min', 1e-6),
            warmup_steps=config.get('warmup_steps', 0),
        )

    return enhancements
