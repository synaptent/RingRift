"""
Advanced Training Utilities for RingRift AI.

This module provides advanced training features:
1. Learning Rate Finder - Find optimal LR range
2. Gradient Checkpointing - Memory-efficient training
3. PFSP Opponent Pool - Prioritized Fictitious Self-Play
4. CMA-ES Auto-Tuning Hook - Trigger HP search on plateau

Usage:
    from app.training.advanced_training import (
        LRFinder,
        GradientCheckpointing,
        PFSPOpponentPool,
        CMAESAutoTuner,
    )
"""

from __future__ import annotations

import copy
import logging
import math
import os
import subprocess
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
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Learning Rate Finder
# =============================================================================


@dataclass
class LRFinderResult:
    """Results from learning rate finder."""
    lrs: List[float]
    losses: List[float]
    suggested_lr: float
    min_lr: float
    max_lr: float
    best_lr: float  # LR at minimum loss
    steepest_lr: float  # LR at steepest gradient


class LRFinder:
    """
    Learning Rate Finder for optimal LR range detection.

    Implements the technique from "Cyclical Learning Rates for Training
    Neural Networks" (Smith, 2017). Gradually increases LR from min to max
    and records the loss at each step.

    Usage:
        finder = LRFinder(model, optimizer, criterion)
        result = finder.range_test(train_loader, min_lr=1e-7, max_lr=10)

        # Use suggested LR
        suggested_lr = result.suggested_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = suggested_lr
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Model to find LR for
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or next(model.parameters()).device

        # Save initial state for restoration
        self._initial_model_state = copy.deepcopy(model.state_dict())
        self._initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

    def range_test(
        self,
        train_loader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_iter: Optional[int] = None,
        step_mode: str = "exp",
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> LRFinderResult:
        """
        Run LR range test.

        Args:
            train_loader: Training data loader
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_iter: Number of iterations (default: len(train_loader))
            step_mode: "exp" for exponential, "linear" for linear increase
            smooth_factor: Smoothing factor for loss (0-1)
            diverge_threshold: Stop if loss exceeds this multiple of min loss

        Returns:
            LRFinderResult with LRs, losses, and suggestions
        """
        # Set up
        num_iter = num_iter or len(train_loader)
        self.model.train()

        # Initialize tracking
        lrs = []
        losses = []
        best_loss = float('inf')
        smoothed_loss = 0.0

        # Calculate LR multiplier
        if step_mode == "exp":
            lr_mult = (max_lr / min_lr) ** (1 / num_iter)
        else:
            lr_step = (max_lr - min_lr) / num_iter

        # Set initial LR
        current_lr = min_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        # Iterate through data
        data_iter = iter(train_loader)
        for i in range(num_iter):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Forward pass
            loss = self._train_step(batch)

            # Smooth loss
            if i == 0:
                smoothed_loss = loss
            else:
                smoothed_loss = smooth_factor * loss + (1 - smooth_factor) * smoothed_loss

            # Record
            lrs.append(current_lr)
            losses.append(smoothed_loss)

            # Track best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > diverge_threshold * best_loss:
                logger.info(f"LR finder stopped early at iter {i} (loss diverged)")
                break

            # Update LR
            if step_mode == "exp":
                current_lr *= lr_mult
            else:
                current_lr += lr_step

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

        # Restore initial state
        self.model.load_state_dict(self._initial_model_state)
        self.optimizer.load_state_dict(self._initial_optimizer_state)

        # Analyze results
        result = self._analyze_results(lrs, losses, min_lr, max_lr)
        return result

    def _train_step(self, batch: Any) -> float:
        """Execute a single training step."""
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            inputs = batch.get('features', batch.get('input'))
            targets = batch.get('targets', batch.get('labels'))
        else:
            inputs = batch
            targets = None

        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        if targets is not None and isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)

        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        # Handle tuple outputs (policy, value)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use policy for LR finding

        # Compute loss
        if targets is not None:
            loss = self.criterion(outputs, targets)
        else:
            # Self-supervised or unsupervised loss
            loss = outputs.mean() if hasattr(outputs, 'mean') else outputs

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _analyze_results(
        self,
        lrs: List[float],
        losses: List[float],
        min_lr: float,
        max_lr: float,
    ) -> LRFinderResult:
        """Analyze LR finder results to suggest optimal LR."""
        if not lrs or not losses:
            return LRFinderResult(
                lrs=[], losses=[], suggested_lr=1e-3,
                min_lr=min_lr, max_lr=max_lr,
                best_lr=1e-3, steepest_lr=1e-3,
            )

        # Find LR at minimum loss
        min_loss_idx = np.argmin(losses)
        best_lr = lrs[min_loss_idx]

        # Find steepest gradient (maximum loss decrease rate)
        # Use gradient of smoothed loss
        if len(losses) > 5:
            log_lrs = np.log10(lrs)
            gradients = np.gradient(losses, log_lrs)

            # Find minimum gradient (steepest descent)
            # Only consider first 80% to avoid divergence region
            cutoff = int(len(gradients) * 0.8)
            steepest_idx = np.argmin(gradients[:cutoff])
            steepest_lr = lrs[steepest_idx]
        else:
            steepest_lr = best_lr

        # Suggested LR: one order of magnitude below steepest point
        # This is a safe default that usually works well
        suggested_lr = steepest_lr / 10

        # Clamp to valid range
        suggested_lr = max(min_lr, min(suggested_lr, best_lr))

        logger.info(
            f"LR Finder: best_lr={best_lr:.2e}, steepest_lr={steepest_lr:.2e}, "
            f"suggested_lr={suggested_lr:.2e}"
        )

        return LRFinderResult(
            lrs=lrs,
            losses=losses,
            suggested_lr=suggested_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            best_lr=best_lr,
            steepest_lr=steepest_lr,
        )

    def plot(self, result: LRFinderResult, save_path: Optional[str] = None):
        """Plot LR finder results."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(result.lrs, result.losses, 'b-', linewidth=2)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Rate Finder')

            # Mark suggested LR
            ax.axvline(x=result.suggested_lr, color='r', linestyle='--',
                      label=f'Suggested LR: {result.suggested_lr:.2e}')
            ax.axvline(x=result.best_lr, color='g', linestyle=':',
                      label=f'Best LR: {result.best_lr:.2e}')

            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved LR finder plot to {save_path}")

            plt.close()

        except ImportError:
            logger.warning("matplotlib not available for plotting")


# =============================================================================
# 2. Gradient Checkpointing
# =============================================================================


class GradientCheckpointing:
    """
    Memory-efficient training via gradient checkpointing.

    Trades compute for memory by not storing intermediate activations
    and recomputing them during backward pass.

    Usage:
        checkpointing = GradientCheckpointing(model)
        checkpointing.enable()

        # During training
        output = checkpointing.checkpoint_forward(model.layer, input)
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_layers: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Model to apply checkpointing to
            checkpoint_layers: Names of layers to checkpoint (None = auto-detect)
        """
        self.model = model
        self.checkpoint_layers = checkpoint_layers
        self._enabled = False
        self._original_forward = {}

    def enable(self) -> None:
        """Enable gradient checkpointing."""
        if self._enabled:
            return

        # Find layers to checkpoint
        layers = self._find_checkpoint_layers()

        for name, module in layers:
            # Store original forward
            self._original_forward[name] = module.forward

            # Replace with checkpointed version
            module.forward = self._make_checkpointed_forward(module)

            logger.debug(f"Enabled checkpointing for {name}")

        self._enabled = True
        logger.info(f"Gradient checkpointing enabled for {len(layers)} layers")

    def disable(self) -> None:
        """Disable gradient checkpointing."""
        if not self._enabled:
            return

        # Restore original forwards
        for name, original_forward in self._original_forward.items():
            # Find module by name
            module = dict(self.model.named_modules()).get(name)
            if module:
                module.forward = original_forward

        self._original_forward.clear()
        self._enabled = False
        logger.info("Gradient checkpointing disabled")

    def _find_checkpoint_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find layers suitable for checkpointing."""
        if self.checkpoint_layers:
            # Use specified layers
            modules = dict(self.model.named_modules())
            return [(name, modules[name]) for name in self.checkpoint_layers
                    if name in modules]

        # Auto-detect: checkpoint transformer blocks, conv blocks, etc.
        layers = []
        for name, module in self.model.named_modules():
            # Skip container modules
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue

            # Checkpoint large layers
            if any(keyword in name.lower() for keyword in
                   ['block', 'layer', 'encoder', 'decoder', 'transformer']):
                layers.append((name, module))
            elif isinstance(module, (nn.TransformerEncoderLayer,
                                    nn.TransformerDecoderLayer)):
                layers.append((name, module))

        # If no blocks found, checkpoint every N layers
        if not layers:
            all_modules = list(self.model.named_modules())
            # Skip first (model itself) and last few modules
            step = max(1, len(all_modules) // 4)
            for i in range(step, len(all_modules) - 2, step):
                name, module = all_modules[i]
                if hasattr(module, 'forward'):
                    layers.append((name, module))

        return layers

    def _make_checkpointed_forward(
        self,
        module: nn.Module,
    ) -> Callable:
        """Create a checkpointed forward function."""
        original_forward = module.forward

        def checkpointed_forward(*args, **kwargs):
            # Only use checkpointing during training
            if self.model.training:
                # Use PyTorch's checkpoint utility
                return torch.utils.checkpoint.checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                return original_forward(*args, **kwargs)

        return checkpointed_forward

    @staticmethod
    def checkpoint_sequential(
        functions: List[nn.Module],
        segments: int,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Checkpoint a sequential list of functions.

        Args:
            functions: List of modules to run sequentially
            segments: Number of checkpoint segments
            input: Input tensor

        Returns:
            Output after all functions
        """
        return torch.utils.checkpoint.checkpoint_sequential(
            functions, segments, input, use_reentrant=False
        )

    @property
    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self._enabled


def estimate_memory_savings(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, float]:
    """
    Estimate memory savings from gradient checkpointing.

    Args:
        model: Model to analyze
        input_shape: Input tensor shape (without batch)
        device: Device to use

    Returns:
        Dictionary with memory estimates
    """
    import gc

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()

    # Measure without checkpointing
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()

    dummy_input = torch.randn(1, *input_shape, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Forward + backward without checkpointing
    output = model_copy(dummy_input)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()

    if device.type == 'cuda':
        normal_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        normal_memory = 0

    del model_copy, dummy_input, output, loss
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()

    # Measure with checkpointing
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    checkpointing = GradientCheckpointing(model_copy)
    checkpointing.enable()

    dummy_input = torch.randn(1, *input_shape, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    output = model_copy(dummy_input)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()

    if device.type == 'cuda':
        checkpoint_memory = torch.cuda.max_memory_allocated() / 1024**2
    else:
        checkpoint_memory = 0

    savings_pct = (1 - checkpoint_memory / normal_memory) * 100 if normal_memory > 0 else 0

    return {
        'normal_memory_mb': normal_memory,
        'checkpoint_memory_mb': checkpoint_memory,
        'savings_mb': normal_memory - checkpoint_memory,
        'savings_percent': savings_pct,
    }


# =============================================================================
# 3. PFSP Opponent Pool Management
# =============================================================================


@dataclass
class OpponentStats:
    """Statistics for an opponent in the pool."""
    model_path: str
    model_name: str
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_played: Optional[datetime] = None
    generation: int = 0
    priority_score: float = 1.0


class PFSPOpponentPool:
    """
    Prioritized Fictitious Self-Play (PFSP) opponent pool management.

    Implements opponent selection strategies from AlphaStar:
    - Prioritize opponents the current model struggles against
    - Maintain diverse opponent pool
    - Balance exploitation (hard opponents) and exploration (new opponents)

    Usage:
        pool = PFSPOpponentPool()
        pool.add_opponent("models/gen1.pth", elo=1500)
        pool.add_opponent("models/gen2.pth", elo=1600)

        # Get opponent for next game
        opponent = pool.sample_opponent(current_elo=1550)

        # Update after game
        pool.update_stats(opponent.model_path, won=False)
    """

    def __init__(
        self,
        max_pool_size: int = 20,
        hard_opponent_weight: float = 0.7,
        diversity_weight: float = 0.2,
        recency_weight: float = 0.1,
        min_games_for_priority: int = 5,
    ):
        """
        Args:
            max_pool_size: Maximum opponents to keep
            hard_opponent_weight: Weight for prioritizing hard opponents
            diversity_weight: Weight for opponent diversity
            recency_weight: Weight for recently played opponents
            min_games_for_priority: Min games before using win rate for priority
        """
        self.max_pool_size = max_pool_size
        self.hard_opponent_weight = hard_opponent_weight
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.min_games_for_priority = min_games_for_priority

        self._opponents: Dict[str, OpponentStats] = {}
        self._game_history: deque = deque(maxlen=1000)

    def add_opponent(
        self,
        model_path: str,
        elo: float = 1500.0,
        generation: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """Add an opponent to the pool."""
        if model_path in self._opponents:
            logger.debug(f"Opponent {model_path} already in pool")
            return

        # Evict oldest if at capacity
        if len(self._opponents) >= self.max_pool_size:
            self._evict_oldest()

        name = name or Path(model_path).stem
        self._opponents[model_path] = OpponentStats(
            model_path=model_path,
            model_name=name,
            elo=elo,
            generation=generation,
        )

        logger.info(f"Added opponent '{name}' to pool (elo={elo}, gen={generation})")

    def remove_opponent(self, model_path: str) -> None:
        """Remove an opponent from the pool."""
        if model_path in self._opponents:
            del self._opponents[model_path]
            logger.info(f"Removed opponent {model_path} from pool")

    def sample_opponent(
        self,
        current_elo: float = 1500.0,
        exclude: Optional[List[str]] = None,
        strategy: str = "pfsp",
    ) -> Optional[OpponentStats]:
        """
        Sample an opponent from the pool.

        Args:
            current_elo: Current model's Elo rating
            exclude: Paths to exclude from sampling
            strategy: Sampling strategy ("pfsp", "uniform", "elo_based")

        Returns:
            Selected opponent or None if pool is empty
        """
        candidates = [
            opp for path, opp in self._opponents.items()
            if exclude is None or path not in exclude
        ]

        if not candidates:
            return None

        if strategy == "uniform":
            return np.random.choice(candidates)

        elif strategy == "elo_based":
            # Prefer opponents near current Elo
            weights = []
            for opp in candidates:
                elo_diff = abs(opp.elo - current_elo)
                # Gaussian weighting around current Elo
                weight = math.exp(-(elo_diff ** 2) / (2 * 200 ** 2))
                weights.append(weight)

            weights = np.array(weights)
            weights /= weights.sum()
            return np.random.choice(candidates, p=weights)

        else:  # PFSP
            return self._pfsp_sample(candidates, current_elo)

    def _pfsp_sample(
        self,
        candidates: List[OpponentStats],
        current_elo: float,
    ) -> OpponentStats:
        """PFSP sampling: prioritize hard opponents."""
        scores = []

        for opp in candidates:
            # Hard opponent score: lower win rate = higher priority
            if opp.games_played >= self.min_games_for_priority:
                win_rate = opp.wins / opp.games_played if opp.games_played > 0 else 0.5
                # Prioritize opponents we lose to
                hard_score = 1.0 - win_rate
            else:
                # Unknown difficulty, moderate priority
                hard_score = 0.5

            # Diversity score: less played = higher priority
            max_games = max(o.games_played for o in candidates) or 1
            diversity_score = 1.0 - (opp.games_played / max_games)

            # Recency score: recently played = lower priority
            if opp.last_played:
                time_since = (datetime.now() - opp.last_played).total_seconds()
                # Decay over 1 hour
                recency_score = min(1.0, time_since / 3600)
            else:
                recency_score = 1.0

            # Combined score
            score = (
                self.hard_opponent_weight * hard_score +
                self.diversity_weight * diversity_score +
                self.recency_weight * recency_score
            )
            scores.append(score)

        # Sample proportionally to scores
        scores = np.array(scores)
        scores = np.maximum(scores, 0.01)  # Ensure all positive
        probs = scores / scores.sum()

        return np.random.choice(candidates, p=probs)

    def update_stats(
        self,
        model_path: str,
        won: bool,
        drew: bool = False,
        elo_change: float = 0.0,
    ) -> None:
        """Update opponent statistics after a game."""
        if model_path not in self._opponents:
            return

        opp = self._opponents[model_path]
        opp.games_played += 1
        opp.last_played = datetime.now()

        if drew:
            opp.draws += 1
        elif won:
            opp.wins += 1
        else:
            opp.losses += 1

        # Update Elo if provided
        if elo_change != 0:
            opp.elo += elo_change

        # Update priority score
        self._update_priority(opp)

        # Record history
        self._game_history.append({
            'opponent': model_path,
            'won': won,
            'drew': drew,
            'timestamp': datetime.now(),
        })

    def _update_priority(self, opp: OpponentStats) -> None:
        """Update priority score for an opponent."""
        if opp.games_played < self.min_games_for_priority:
            opp.priority_score = 1.0
            return

        # Priority based on win rate against this opponent
        win_rate = opp.wins / opp.games_played
        # Higher priority for lower win rates (harder opponents)
        opp.priority_score = 1.0 - win_rate

    def _evict_oldest(self) -> None:
        """Evict the oldest/least useful opponent."""
        if not self._opponents:
            return

        # Score opponents for eviction
        eviction_scores = {}
        for path, opp in self._opponents.items():
            # Prefer to evict:
            # - High win rate (easy opponents)
            # - Long since played
            # - Low generation (old models)

            win_rate = opp.wins / max(1, opp.games_played)
            age_score = opp.generation / max(o.generation for o in self._opponents.values()) if self._opponents else 0

            eviction_scores[path] = win_rate - 0.3 * age_score

        # Evict highest scoring (easiest/oldest)
        to_evict = max(eviction_scores, key=eviction_scores.get)
        logger.info(f"Evicting opponent {to_evict} from pool")
        del self._opponents[to_evict]

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the opponent pool."""
        if not self._opponents:
            return {'size': 0}

        elos = [o.elo for o in self._opponents.values()]
        games = [o.games_played for o in self._opponents.values()]

        return {
            'size': len(self._opponents),
            'avg_elo': np.mean(elos),
            'min_elo': min(elos),
            'max_elo': max(elos),
            'total_games': sum(games),
            'avg_games_per_opponent': np.mean(games),
        }

    def get_opponents(self) -> List[OpponentStats]:
        """Get all opponents in the pool."""
        return list(self._opponents.values())

    def save_pool(self, path: Union[str, Path]) -> None:
        """Save pool state to file."""
        import json

        data = {
            'opponents': {
                p: {
                    'model_path': o.model_path,
                    'model_name': o.model_name,
                    'elo': o.elo,
                    'games_played': o.games_played,
                    'wins': o.wins,
                    'losses': o.losses,
                    'draws': o.draws,
                    'generation': o.generation,
                    'priority_score': o.priority_score,
                }
                for p, o in self._opponents.items()
            },
            'config': {
                'max_pool_size': self.max_pool_size,
                'hard_opponent_weight': self.hard_opponent_weight,
                'diversity_weight': self.diversity_weight,
                'recency_weight': self.recency_weight,
            },
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_pool(self, path: Union[str, Path]) -> None:
        """Load pool state from file."""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        for p, o in data['opponents'].items():
            self._opponents[p] = OpponentStats(**o)

        config = data.get('config', {})
        self.max_pool_size = config.get('max_pool_size', self.max_pool_size)
        self.hard_opponent_weight = config.get('hard_opponent_weight', self.hard_opponent_weight)


# =============================================================================
# 4. CMA-ES Auto-Tuning Hook
# =============================================================================


@dataclass
class PlateauConfig:
    """Configuration for plateau detection."""
    patience: int = 10  # Epochs without improvement
    min_delta: float = 0.01  # Minimum improvement to reset patience
    metric: str = "elo"  # "elo", "loss", or "win_rate"
    lookback: int = 20  # Epochs to consider for trend


class CMAESAutoTuner:
    """
    Automatically trigger CMA-ES hyperparameter optimization on plateau.

    Monitors training metrics and launches CMA-ES optimization when
    progress stalls, then integrates optimized hyperparameters.

    Usage:
        auto_tuner = CMAESAutoTuner(
            cmaes_script="scripts/run_gpu_cmaes.py",
            board_type="square8",
        )

        for epoch in range(epochs):
            train_epoch()
            auto_tuner.step(current_elo=elo, current_loss=loss)

            if auto_tuner.should_tune():
                result = auto_tuner.run_optimization()
                if result:
                    # Apply new hyperparameters
                    apply_weights(result['weights'])
    """

    def __init__(
        self,
        cmaes_script: str = "scripts/run_gpu_cmaes.py",
        board_type: str = "square8",
        num_players: int = 2,
        plateau_config: Optional[PlateauConfig] = None,
        output_dir: str = "logs/cmaes_auto",
        min_epochs_between_tuning: int = 50,
        max_auto_tunes: int = 5,
    ):
        """
        Args:
            cmaes_script: Path to CMA-ES optimization script
            board_type: Board type for optimization
            num_players: Number of players
            plateau_config: Plateau detection configuration
            output_dir: Output directory for CMA-ES results
            min_epochs_between_tuning: Minimum epochs between auto-tunes
            max_auto_tunes: Maximum auto-tunes per training run
        """
        self.cmaes_script = cmaes_script
        self.board_type = board_type
        self.num_players = num_players
        self.plateau_config = plateau_config or PlateauConfig()
        self.output_dir = Path(output_dir)
        self.min_epochs_between_tuning = min_epochs_between_tuning
        self.max_auto_tunes = max_auto_tunes

        self._metric_history: deque = deque(maxlen=self.plateau_config.lookback)
        self._epochs_without_improvement = 0
        self._best_metric = float('-inf')
        self._last_tune_epoch = -self.min_epochs_between_tuning
        self._tune_count = 0
        self._current_epoch = 0
        self._is_tuning = False

    def step(
        self,
        current_elo: Optional[float] = None,
        current_loss: Optional[float] = None,
        current_win_rate: Optional[float] = None,
    ) -> None:
        """
        Update with current training metrics.

        Args:
            current_elo: Current Elo rating
            current_loss: Current training loss
            current_win_rate: Current win rate against baseline
        """
        self._current_epoch += 1

        # Get metric value
        metric = self.plateau_config.metric
        if metric == "elo" and current_elo is not None:
            value = current_elo
        elif metric == "loss" and current_loss is not None:
            value = -current_loss  # Lower loss is better
        elif metric == "win_rate" and current_win_rate is not None:
            value = current_win_rate
        else:
            return

        self._metric_history.append(value)

        # Check for improvement
        if value > self._best_metric + self.plateau_config.min_delta:
            self._best_metric = value
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

    def should_tune(self) -> bool:
        """Check if CMA-ES optimization should be triggered."""
        if self._is_tuning:
            return False

        if self._tune_count >= self.max_auto_tunes:
            return False

        if (self._current_epoch - self._last_tune_epoch) < self.min_epochs_between_tuning:
            return False

        # Check for plateau
        if self._epochs_without_improvement >= self.plateau_config.patience:
            logger.info(
                f"Plateau detected: {self._epochs_without_improvement} epochs "
                f"without improvement (patience={self.plateau_config.patience})"
            )
            return True

        return False

    def run_optimization(
        self,
        generations: int = 30,
        population_size: int = 15,
        games_per_eval: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Run CMA-ES optimization.

        Args:
            generations: Number of CMA-ES generations
            population_size: Population size
            games_per_eval: Games per fitness evaluation

        Returns:
            Dictionary with optimized weights and metrics, or None on failure
        """
        if self._is_tuning:
            logger.warning("Optimization already in progress")
            return None

        self._is_tuning = True
        self._tune_count += 1
        self._last_tune_epoch = self._current_epoch

        # Create output directory
        run_dir = self.output_dir / f"auto_tune_{self._tune_count}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting auto CMA-ES optimization #{self._tune_count}")

        try:
            # Build command
            cmd = [
                "python", self.cmaes_script,
                "--board", self.board_type,
                "--num-players", str(self.num_players),
                "--generations", str(generations),
                "--population-size", str(population_size),
                "--games-per-eval", str(games_per_eval),
                "--output-dir", str(run_dir),
            ]

            # Run CMA-ES
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"CMA-ES failed: {result.stderr}")
                return None

            # Load results
            import json
            results_file = run_dir / "optimized_weights.json"
            if results_file.exists():
                with open(results_file) as f:
                    weights = json.load(f)

                # Reset improvement tracking
                self._epochs_without_improvement = 0

                logger.info(f"CMA-ES optimization complete. Best fitness: {weights.get('fitness', 'N/A')}")

                return {
                    'weights': weights,
                    'run_dir': str(run_dir),
                    'tune_number': self._tune_count,
                }
            else:
                logger.warning("CMA-ES completed but no weights file found")
                return None

        except subprocess.TimeoutExpired:
            logger.error("CMA-ES optimization timed out")
            return None
        except Exception as e:
            logger.error(f"CMA-ES optimization failed: {e}")
            return None
        finally:
            self._is_tuning = False

    def get_status(self) -> Dict[str, Any]:
        """Get current auto-tuner status."""
        return {
            'current_epoch': self._current_epoch,
            'epochs_without_improvement': self._epochs_without_improvement,
            'patience': self.plateau_config.patience,
            'best_metric': self._best_metric,
            'tune_count': self._tune_count,
            'max_auto_tunes': self.max_auto_tunes,
            'is_tuning': self._is_tuning,
            'epochs_since_last_tune': self._current_epoch - self._last_tune_epoch,
        }

    def reset(self) -> None:
        """Reset the auto-tuner state."""
        self._metric_history.clear()
        self._epochs_without_improvement = 0
        self._best_metric = float('-inf')
        self._tune_count = 0
        self._current_epoch = 0
        self._last_tune_epoch = -self.min_epochs_between_tuning


# =============================================================================
# Integration Helpers
# =============================================================================


def create_advanced_training_suite(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    board_type: str = "square8",
    num_players: int = 2,
    enable_checkpointing: bool = True,
    enable_pfsp: bool = True,
    enable_auto_tuning: bool = True,
) -> Dict[str, Any]:
    """
    Create a suite of advanced training utilities.

    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        board_type: Board type
        num_players: Number of players
        enable_checkpointing: Enable gradient checkpointing
        enable_pfsp: Enable PFSP opponent pool
        enable_auto_tuning: Enable CMA-ES auto-tuning

    Returns:
        Dictionary of utility objects
    """
    suite = {
        'lr_finder': LRFinder(model, optimizer, criterion),
    }

    if enable_checkpointing:
        checkpointing = GradientCheckpointing(model)
        suite['checkpointing'] = checkpointing

    if enable_pfsp:
        pfsp_pool = PFSPOpponentPool()
        suite['pfsp_pool'] = pfsp_pool

    if enable_auto_tuning:
        auto_tuner = CMAESAutoTuner(
            board_type=board_type,
            num_players=num_players,
        )
        suite['auto_tuner'] = auto_tuner

    return suite


# =============================================================================
# Phase 4: Training Stability & Acceleration (2024-12)
# =============================================================================


@dataclass
class StabilityMetrics:
    """Metrics for training stability monitoring."""
    gradient_norm: float
    loss_value: float
    loss_variance: float
    param_update_ratio: float
    is_stable: bool
    warnings: List[str] = field(default_factory=list)


class TrainingStabilityMonitor:
    """
    Monitor training stability and detect issues early.

    Detects:
    - Gradient explosions/vanishing
    - Loss spikes or NaN
    - Parameter update instabilities
    - Learning rate issues
    """

    def __init__(
        self,
        gradient_clip_threshold: float = 10.0,
        loss_spike_threshold: float = 3.0,
        loss_history_size: int = 100,
        param_update_threshold: float = 0.1,
        auto_recover: bool = True,
    ):
        self.gradient_clip_threshold = gradient_clip_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.loss_history_size = loss_history_size
        self.param_update_threshold = param_update_threshold
        self.auto_recover = auto_recover

        self._loss_history: deque = deque(maxlen=loss_history_size)
        self._gradient_history: deque = deque(maxlen=loss_history_size)
        self._last_params: Optional[Dict[str, torch.Tensor]] = None
        self._recovery_triggered = False
        self._stability_score = 1.0

    def check_gradients(self, model: nn.Module) -> Tuple[float, List[str]]:
        """Check gradient health across all parameters."""
        warnings = []
        total_norm = 0.0
        num_params = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1

                if math.isnan(param_norm) or math.isinf(param_norm):
                    warnings.append(f"NaN/Inf gradient in {name}")
                elif param_norm > self.gradient_clip_threshold * 10:
                    warnings.append(f"Extreme gradient in {name}: {param_norm:.4f}")

        total_norm = math.sqrt(total_norm) if num_params > 0 else 0.0
        self._gradient_history.append(total_norm)

        if total_norm > self.gradient_clip_threshold:
            warnings.append(f"Gradient norm {total_norm:.4f} exceeds threshold")
        elif total_norm < 1e-7:
            warnings.append("Vanishing gradients detected")

        return total_norm, warnings

    def check_loss(self, loss: float) -> Tuple[bool, List[str]]:
        """Check if loss is healthy."""
        warnings = []
        is_healthy = True

        if math.isnan(loss) or math.isinf(loss):
            warnings.append(f"Invalid loss value: {loss}")
            is_healthy = False
        else:
            self._loss_history.append(loss)

            if len(self._loss_history) >= 10:
                recent_mean = np.mean(list(self._loss_history)[-10:])
                overall_mean = np.mean(list(self._loss_history))
                overall_std = np.std(list(self._loss_history))

                if overall_std > 0 and abs(loss - overall_mean) > self.loss_spike_threshold * overall_std:
                    warnings.append(f"Loss spike detected: {loss:.4f} vs mean {overall_mean:.4f}")

        return is_healthy, warnings

    def check_param_updates(self, model: nn.Module) -> Tuple[float, List[str]]:
        """Check parameter update ratios."""
        warnings = []
        update_ratios = []

        current_params = {name: param.data.clone() for name, param in model.named_parameters()}

        if self._last_params is not None:
            for name, current in current_params.items():
                if name in self._last_params:
                    last = self._last_params[name]
                    diff = (current - last).norm().item()
                    param_norm = current.norm().item()

                    if param_norm > 0:
                        ratio = diff / param_norm
                        update_ratios.append(ratio)

                        if ratio > self.param_update_threshold:
                            warnings.append(f"Large param update in {name}: {ratio:.4f}")

        self._last_params = current_params
        avg_ratio = np.mean(update_ratios) if update_ratios else 0.0
        return avg_ratio, warnings

    def step(
        self,
        model: nn.Module,
        loss: float,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> StabilityMetrics:
        """Run all stability checks and return metrics."""
        all_warnings = []

        grad_norm, grad_warnings = self.check_gradients(model)
        all_warnings.extend(grad_warnings)

        loss_healthy, loss_warnings = self.check_loss(loss)
        all_warnings.extend(loss_warnings)

        update_ratio, update_warnings = self.check_param_updates(model)
        all_warnings.extend(update_warnings)

        loss_var = np.var(list(self._loss_history)) if len(self._loss_history) > 1 else 0.0

        is_stable = loss_healthy and len(all_warnings) == 0

        # Update stability score
        if is_stable:
            self._stability_score = min(1.0, self._stability_score + 0.01)
        else:
            self._stability_score = max(0.0, self._stability_score - 0.1)

        # Auto-recovery if enabled
        if self.auto_recover and not is_stable and optimizer is not None:
            if len(all_warnings) > 3:
                self._trigger_recovery(optimizer)

        return StabilityMetrics(
            gradient_norm=grad_norm,
            loss_value=loss,
            loss_variance=loss_var,
            param_update_ratio=update_ratio,
            is_stable=is_stable,
            warnings=all_warnings,
        )

    def _trigger_recovery(self, optimizer: optim.Optimizer) -> None:
        """Attempt to recover from instability."""
        if self._recovery_triggered:
            return

        logger.warning("Training instability detected, triggering recovery...")

        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            logger.info(f"Reduced LR to {param_group['lr']:.2e}")

        self._recovery_triggered = True

    @property
    def stability_score(self) -> float:
        """Get current stability score (0-1)."""
        return self._stability_score

    def get_summary(self) -> Dict[str, Any]:
        """Get stability summary."""
        return {
            'stability_score': self._stability_score,
            'avg_gradient_norm': np.mean(list(self._gradient_history)) if self._gradient_history else 0.0,
            'avg_loss': np.mean(list(self._loss_history)) if self._loss_history else 0.0,
            'loss_variance': np.var(list(self._loss_history)) if len(self._loss_history) > 1 else 0.0,
            'recovery_triggered': self._recovery_triggered,
        }


class AdaptivePrecisionManager:
    """
    Dynamically adjust mixed precision based on training stability.

    Switches between FP32, FP16, and BF16 based on:
    - Loss stability
    - Gradient overflow frequency
    - Training progress
    """

    def __init__(
        self,
        initial_precision: str = "fp16",
        stability_window: int = 100,
        overflow_threshold: float = 0.05,
        auto_downgrade: bool = True,
        auto_upgrade: bool = True,
    ):
        self.current_precision = initial_precision
        self.stability_window = stability_window
        self.overflow_threshold = overflow_threshold
        self.auto_downgrade = auto_downgrade
        self.auto_upgrade = auto_upgrade

        self._overflow_count = 0
        self._step_count = 0
        self._precision_history: List[str] = []
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

        self._precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

    def setup(self, device: torch.device) -> Optional[torch.cuda.amp.GradScaler]:
        """Setup precision management."""
        if device.type != "cuda":
            self.current_precision = "fp32"
            return None

        if self.current_precision == "fp16":
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

        return self._scaler

    def get_autocast_dtype(self) -> torch.dtype:
        """Get current autocast dtype."""
        return self._precision_map.get(self.current_precision, torch.float32)

    def record_overflow(self, had_overflow: bool) -> None:
        """Record gradient overflow event."""
        self._step_count += 1
        if had_overflow:
            self._overflow_count += 1

        # Check if we should change precision
        if self._step_count >= self.stability_window:
            overflow_rate = self._overflow_count / self._step_count

            if overflow_rate > self.overflow_threshold and self.auto_downgrade:
                self._downgrade_precision()
            elif overflow_rate < self.overflow_threshold * 0.1 and self.auto_upgrade:
                self._upgrade_precision()

            # Reset counters
            self._step_count = 0
            self._overflow_count = 0

    def _downgrade_precision(self) -> None:
        """Downgrade to more stable precision."""
        if self.current_precision == "fp16":
            self.current_precision = "bf16"
            self._scaler = None
            logger.info("Precision downgraded: FP16 -> BF16")
        elif self.current_precision == "bf16":
            self.current_precision = "fp32"
            logger.info("Precision downgraded: BF16 -> FP32")

        self._precision_history.append(self.current_precision)

    def _upgrade_precision(self) -> None:
        """Upgrade to faster precision."""
        if self.current_precision == "fp32":
            self.current_precision = "bf16"
            logger.info("Precision upgraded: FP32 -> BF16")
        elif self.current_precision == "bf16":
            self.current_precision = "fp16"
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("Precision upgraded: BF16 -> FP16")

        self._precision_history.append(self.current_precision)

    def get_stats(self) -> Dict[str, Any]:
        """Get precision management stats."""
        return {
            'current_precision': self.current_precision,
            'overflow_rate': self._overflow_count / max(1, self._step_count),
            'precision_history': self._precision_history[-10:],
        }


class ProgressiveLayerUnfreezing:
    """
    Gradually unfreeze model layers during training.

    Implements discriminative fine-tuning where:
    - Early layers are frozen initially
    - Layers are progressively unfrozen as training progresses
    - Different layers can have different learning rates
    """

    def __init__(
        self,
        model: nn.Module,
        unfreeze_schedule: Optional[Dict[int, List[str]]] = None,
        lr_multipliers: Optional[Dict[str, float]] = None,
        total_epochs: int = 50,
    ):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule or {}
        self.lr_multipliers = lr_multipliers or {}
        self.total_epochs = total_epochs

        self._frozen_layers: set = set()
        self._unfrozen_at_epoch: Dict[str, int] = {}

    def freeze_all_except(self, layer_names: List[str]) -> None:
        """Freeze all layers except specified ones."""
        for name, param in self.model.named_parameters():
            should_freeze = True
            for unfrozen in layer_names:
                if unfrozen in name:
                    should_freeze = False
                    break

            param.requires_grad = not should_freeze
            if should_freeze:
                self._frozen_layers.add(name)

    def freeze_layers(self, layer_names: List[str]) -> None:
        """Freeze specific layers."""
        for name, param in self.model.named_parameters():
            for frozen_name in layer_names:
                if frozen_name in name:
                    param.requires_grad = False
                    self._frozen_layers.add(name)
                    break

    def unfreeze_layers(self, layer_names: List[str], epoch: int) -> List[str]:
        """Unfreeze specific layers."""
        unfrozen = []
        for name, param in self.model.named_parameters():
            for unfrozen_name in layer_names:
                if unfrozen_name in name and name in self._frozen_layers:
                    param.requires_grad = True
                    self._frozen_layers.discard(name)
                    self._unfrozen_at_epoch[name] = epoch
                    unfrozen.append(name)
                    break
        return unfrozen

    def step(self, epoch: int, optimizer: optim.Optimizer) -> List[str]:
        """Update layer freezing based on epoch."""
        unfrozen = []

        # Check schedule
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            unfrozen = self.unfreeze_layers(layers_to_unfreeze, epoch)

            if unfrozen:
                logger.info(f"Epoch {epoch}: Unfroze {len(unfrozen)} layers")

                # Update optimizer with new parameters
                self._update_optimizer_params(optimizer)

        return unfrozen

    def _update_optimizer_params(self, optimizer: optim.Optimizer) -> None:
        """Update optimizer with trainable parameters and LR multipliers."""
        trainable_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lr_mult = 1.0
                for pattern, mult in self.lr_multipliers.items():
                    if pattern in name:
                        lr_mult = mult
                        break

                trainable_params.append({
                    'params': [param],
                    'lr': optimizer.defaults['lr'] * lr_mult,
                    'name': name,
                })

        # Replace optimizer param groups
        optimizer.param_groups = trainable_params

    def get_status(self) -> Dict[str, Any]:
        """Get layer freezing status."""
        return {
            'frozen_count': len(self._frozen_layers),
            'trainable_count': sum(1 for p in self.model.parameters() if p.requires_grad),
            'unfrozen_at_epoch': self._unfrozen_at_epoch.copy(),
        }

    @staticmethod
    def create_default_schedule(
        model: nn.Module,
        total_epochs: int,
        num_stages: int = 4,
    ) -> Dict[int, List[str]]:
        """Create a default unfreezing schedule."""
        schedule = {}
        layers = [name for name, _ in model.named_modules()
                  if any(kw in name.lower() for kw in ['layer', 'block', 'encoder', 'decoder'])]

        if not layers:
            return schedule

        stage_size = len(layers) // num_stages
        epoch_step = total_epochs // num_stages

        for i in range(num_stages):
            epoch = i * epoch_step
            start_idx = len(layers) - (i + 1) * stage_size
            end_idx = len(layers) - i * stage_size if i > 0 else len(layers)
            schedule[epoch] = layers[max(0, start_idx):end_idx]

        return schedule


class SWAWithRestarts:
    """
    Stochastic Weight Averaging with periodic restarts.

    Combines SWA's generalization benefits with warm restarts
    for better optimization landscape exploration.
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start: float = 0.75,
        swa_lr: Optional[float] = None,
        restart_period: int = 10,
        num_restarts: int = 3,
    ):
        self.model = model
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.restart_period = restart_period
        self.num_restarts = num_restarts

        self._swa_model: Optional[torch.optim.swa_utils.AveragedModel] = None
        self._restart_count = 0
        self._steps_since_restart = 0
        self._is_averaging = False

    def setup(self, device: torch.device) -> None:
        """Initialize SWA model."""
        self._swa_model = torch.optim.swa_utils.AveragedModel(self.model)

    def should_start_averaging(self, progress: float) -> bool:
        """Check if SWA should start based on training progress."""
        return progress >= self.swa_start

    def step(
        self,
        epoch: int,
        total_epochs: int,
        optimizer: optim.Optimizer,
    ) -> bool:
        """Update SWA state and check for restarts."""
        progress = epoch / total_epochs
        did_restart = False

        if self.should_start_averaging(progress):
            if not self._is_averaging:
                self._is_averaging = True
                logger.info(f"SWA started at epoch {epoch}")

                if self.swa_lr is not None:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.swa_lr

            # Update SWA model
            if self._swa_model is not None:
                self._swa_model.update_parameters(self.model)

            # Check for restart
            self._steps_since_restart += 1
            if (self._steps_since_restart >= self.restart_period and
                self._restart_count < self.num_restarts):
                self._trigger_restart(optimizer)
                did_restart = True

        return did_restart

    def _trigger_restart(self, optimizer: optim.Optimizer) -> None:
        """Trigger a warm restart."""
        self._restart_count += 1
        self._steps_since_restart = 0

        # Restore original learning rate (will decay again)
        if self.swa_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.swa_lr * 1.5  # Slight bump

        logger.info(f"SWA restart {self._restart_count}/{self.num_restarts}")

    def get_averaged_model(self) -> Optional[nn.Module]:
        """Get the SWA averaged model."""
        return self._swa_model

    def update_bn(self, loader: DataLoader, device: torch.device) -> None:
        """Update batch normalization statistics."""
        if self._swa_model is not None:
            torch.optim.swa_utils.update_bn(loader, self._swa_model, device=device)

    def get_stats(self) -> Dict[str, Any]:
        """Get SWA statistics."""
        return {
            'is_averaging': self._is_averaging,
            'restart_count': self._restart_count,
            'steps_since_restart': self._steps_since_restart,
        }


class SmartCheckpointManager:
    """
    Intelligent checkpoint management with minimal I/O overhead.

    Features:
    - Adaptive checkpointing frequency based on loss improvement
    - Keep only top-k best checkpoints
    - Async checkpoint saving
    - Checkpoint compression
    """

    def __init__(
        self,
        save_dir: Path,
        top_k: int = 3,
        min_interval_epochs: int = 1,
        max_interval_epochs: int = 10,
        improvement_threshold: float = 0.01,
    ):
        self.save_dir = Path(save_dir)
        self.top_k = top_k
        self.min_interval = min_interval_epochs
        self.max_interval = max_interval_epochs
        self.improvement_threshold = improvement_threshold

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: List[Tuple[float, Path]] = []
        self._last_save_epoch = -1
        self._last_loss = float('inf')
        self._adaptive_interval = min_interval_epochs

    def should_save(self, epoch: int, loss: float) -> bool:
        """Determine if checkpoint should be saved."""
        if epoch - self._last_save_epoch < self.min_interval:
            return False

        # Always save if significant improvement
        improvement = (self._last_loss - loss) / (abs(self._last_loss) + 1e-8)
        if improvement > self.improvement_threshold:
            self._adaptive_interval = self.min_interval
            return True

        # Adaptive interval based on improvement rate
        if epoch - self._last_save_epoch >= self._adaptive_interval:
            self._adaptive_interval = min(
                self._adaptive_interval + 1,
                self.max_interval
            )
            return True

        return False

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch:04d}_loss{loss:.4f}.pt"

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        self._checkpoints.append((loss, checkpoint_path))
        self._checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints if exceeding top-k
        while len(self._checkpoints) > self.top_k:
            _, old_path = self._checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")

        self._last_save_epoch = epoch
        self._last_loss = loss

        return checkpoint_path

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self._checkpoints:
            return self._checkpoints[0][1]
        return None

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
    ) -> Optional[Dict[str, Any]]:
        """Load best checkpoint."""
        best_path = self.get_best_checkpoint()
        if best_path is None or not best_path.exists():
            return None

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint stats."""
        return {
            'num_checkpoints': len(self._checkpoints),
            'best_loss': self._checkpoints[0][0] if self._checkpoints else None,
            'adaptive_interval': self._adaptive_interval,
            'last_save_epoch': self._last_save_epoch,
        }


# Update the suite creation function
def create_phase4_training_suite(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    save_dir: Path,
    device: torch.device,
    total_epochs: int = 50,
    enable_stability_monitor: bool = True,
    enable_adaptive_precision: bool = True,
    enable_progressive_unfreezing: bool = False,
    enable_swa_restarts: bool = True,
    enable_smart_checkpoints: bool = True,
) -> Dict[str, Any]:
    """
    Create Phase 4 training utilities suite.

    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        save_dir: Directory for checkpoints
        device: Training device
        total_epochs: Total training epochs
        enable_*: Flags to enable specific features

    Returns:
        Dictionary of Phase 4 utility objects
    """
    suite = {}

    if enable_stability_monitor:
        suite['stability_monitor'] = TrainingStabilityMonitor()

    if enable_adaptive_precision:
        precision_mgr = AdaptivePrecisionManager()
        suite['precision_manager'] = precision_mgr
        suite['scaler'] = precision_mgr.setup(device)

    if enable_progressive_unfreezing:
        schedule = ProgressiveLayerUnfreezing.create_default_schedule(
            model, total_epochs
        )
        suite['layer_unfreezing'] = ProgressiveLayerUnfreezing(
            model, unfreeze_schedule=schedule, total_epochs=total_epochs
        )

    if enable_swa_restarts:
        swa = SWAWithRestarts(model)
        swa.setup(device)
        suite['swa_restarts'] = swa

    if enable_smart_checkpoints:
        suite['checkpoint_manager'] = SmartCheckpointManager(save_dir)

    return suite
