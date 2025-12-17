"""
Training Enhancements for RingRift AI.

This module provides advanced training optimizations:
1. Checkpoint averaging for improved final model
2. Gradient accumulation for larger effective batch sizes
3. Data quality scoring for sample prioritization
4. Adaptive learning rate based on Elo progress
5. Early stopping with patience
6. EWC (Elastic Weight Consolidation) for continual learning
7. Model ensemble support for self-play
8. Value head calibration automation

Usage:
    from app.training.training_enhancements import (
        CheckpointAverager,
        GradientAccumulator,
        DataQualityScorer,
        AdaptiveLRScheduler,
        EWCRegularizer,
        ModelEnsemble,
        EnhancedEarlyStopping,
    )
"""

from __future__ import annotations

import copy
import logging
import math
import os
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'game_id': self.game_id,
            'total_score': self.total_score,
            'length_score': self.length_score,
            'elo_score': self.elo_score,
            'diversity_score': self.diversity_score,
            'decisive_score': self.decisive_score,
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
    """

    def __init__(
        self,
        min_game_length: int = 20,
        max_game_length: int = 500,
        optimal_game_length: int = 100,
        max_elo_diff: float = 400.0,
        decisive_bonus: float = 1.2,
        draw_penalty: float = 0.8,
    ):
        """
        Args:
            min_game_length: Minimum acceptable game length
            max_game_length: Maximum acceptable game length
            optimal_game_length: Optimal game length for highest score
            max_elo_diff: Maximum Elo differential for scoring
            decisive_bonus: Bonus multiplier for decisive games
            draw_penalty: Penalty multiplier for drawn games
        """
        self.min_game_length = min_game_length
        self.max_game_length = max_game_length
        self.optimal_game_length = optimal_game_length
        self.max_elo_diff = max_elo_diff
        self.decisive_bonus = decisive_bonus
        self.draw_penalty = draw_penalty

    def score_game(
        self,
        game_id: str,
        game_length: int,
        winner: Optional[int] = None,
        elo_p1: Optional[float] = None,
        elo_p2: Optional[float] = None,
        move_entropy: Optional[float] = None,
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

        # Total score (weighted combination)
        total_score = (
            0.3 * length_score +
            0.2 * elo_score +
            0.2 * diversity_score +
            0.3 * decisive_score
        )

        return GameQualityScore(
            game_id=game_id,
            total_score=total_score,
            length_score=length_score,
            elo_score=elo_score,
            diversity_score=diversity_score,
            decisive_score=decisive_score,
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
# Convenience Functions
# =============================================================================


def create_training_enhancements(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a suite of training enhancements with default configuration.

    Args:
        model: Model to enhance training for
        optimizer: Optimizer to use
        config: Optional configuration overrides

    Returns:
        Dictionary of enhancement objects
    """
    config = config or {}

    return {
        'checkpoint_averager': CheckpointAverager(
            num_checkpoints=config.get('avg_checkpoints', 5),
        ),
        'gradient_accumulator': GradientAccumulator(
            accumulation_steps=config.get('accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
        ),
        'quality_scorer': DataQualityScorer(),
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
    }
