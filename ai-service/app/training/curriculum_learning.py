"""
DEPRECATED: This module is deprecated. Use app.training.curriculum instead.

All functionality from this module has been merged into the canonical
curriculum.py module:

    from app.training.curriculum import (
        CurriculumStage,
        StageConfig,
        DEFAULT_STAGES,
        STAGE_ORDER,
        estimate_position_complexity,
        filter_samples_by_complexity,
        get_sample_weights_by_complexity,
    )

This file is kept for backwards compatibility but will be removed in
a future version.

---
Original docstring:

Curriculum Learning for RingRift AI Training.

Implements progressive difficulty training where the model starts with
simpler positions/games and gradually advances to harder ones.
"""
import warnings

warnings.warn(
    "curriculum_learning.py is deprecated. Use app.training.curriculum instead. "
    "All stage-based curriculum features have been merged into the canonical module.",
    DeprecationWarning,
    stacklevel=2,
)

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages in order of difficulty."""
    EARLY_GAME = "early_game"       # First 10 moves
    MIDGAME = "midgame"             # Moves 10-30
    LATE_GAME = "late_game"         # Moves 30+
    FULL_GAME = "full_game"         # All positions
    ADVERSARIAL = "adversarial"     # Hard positions


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    name: str
    min_move: int
    max_move: int
    complexity_range: Tuple[float, float]  # (min, max) complexity score
    target_accuracy: float  # Accuracy needed to advance
    min_epochs: int  # Minimum epochs before advancing
    sample_weight: float = 1.0  # Weight for mixing stages


# Default stage configurations
DEFAULT_STAGES = {
    CurriculumStage.EARLY_GAME: StageConfig(
        name="Early Game",
        min_move=0,
        max_move=10,
        complexity_range=(0.0, 0.3),
        target_accuracy=0.70,
        min_epochs=5,
        sample_weight=1.0,
    ),
    CurriculumStage.MIDGAME: StageConfig(
        name="Midgame",
        min_move=10,
        max_move=30,
        complexity_range=(0.3, 0.6),
        target_accuracy=0.60,
        min_epochs=10,
        sample_weight=1.0,
    ),
    CurriculumStage.LATE_GAME: StageConfig(
        name="Late Game",
        min_move=30,
        max_move=999,
        complexity_range=(0.6, 0.8),
        target_accuracy=0.55,
        min_epochs=10,
        sample_weight=1.0,
    ),
    CurriculumStage.FULL_GAME: StageConfig(
        name="Full Game",
        min_move=0,
        max_move=999,
        complexity_range=(0.0, 1.0),
        target_accuracy=0.55,
        min_epochs=20,
        sample_weight=1.0,
    ),
    CurriculumStage.ADVERSARIAL: StageConfig(
        name="Adversarial",
        min_move=0,
        max_move=999,
        complexity_range=(0.8, 1.0),
        target_accuracy=0.50,
        min_epochs=10,
        sample_weight=2.0,  # Higher weight for hard positions
    ),
}

# Stage progression order
STAGE_ORDER = [
    CurriculumStage.EARLY_GAME,
    CurriculumStage.MIDGAME,
    CurriculumStage.LATE_GAME,
    CurriculumStage.FULL_GAME,
    CurriculumStage.ADVERSARIAL,
]


@dataclass
class CurriculumState:
    """Current state of curriculum learning."""
    current_stage: str = CurriculumStage.EARLY_GAME.value
    stage_epochs: int = 0
    total_epochs: int = 0
    stage_best_accuracy: float = 0.0
    stage_best_loss: float = float("inf")
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    advancement_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumState":
        return cls(**data)


def estimate_position_complexity(
    move_idx: int,
    total_moves: int,
    piece_count: int = 0,
    branching_factor: float = 0.0,
) -> float:
    """Estimate complexity score for a position.

    Args:
        move_idx: Move number in the game
        total_moves: Total moves in the game
        piece_count: Number of pieces on board (optional)
        branching_factor: Average legal moves available (optional)

    Returns:
        Complexity score in [0, 1]
    """
    # Base complexity from game phase
    if total_moves == 0:
        phase_complexity = 0.0
    else:
        progress = move_idx / max(total_moves, 1)
        # Complexity peaks in midgame (around 40-60% through game)
        phase_complexity = 1.0 - abs(progress - 0.5) * 2

    # Adjust for move number (early game is simpler)
    move_complexity = min(move_idx / 30, 1.0)

    # Combine factors
    complexity = 0.5 * phase_complexity + 0.5 * move_complexity

    # Boost complexity if high branching factor provided
    if branching_factor > 0:
        branch_factor = min(branching_factor / 100, 1.0)
        complexity = 0.7 * complexity + 0.3 * branch_factor

    return min(max(complexity, 0.0), 1.0)


class CurriculumManager:
    """Manages curriculum learning for RingRift AI training."""

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        stages: Optional[Dict[CurriculumStage, StageConfig]] = None,
        state_path: Optional[Path] = None,
        mixing_enabled: bool = True,
        mixing_ratio: float = 0.2,  # 20% from previous stages
    ):
        """
        Initialize curriculum manager.

        Args:
            board_type: Board type
            num_players: Number of players
            stages: Custom stage configurations
            state_path: Path to save/load state
            mixing_enabled: Whether to mix samples from previous stages
            mixing_ratio: Ratio of samples from previous stages
        """
        self.board_type = board_type
        self.num_players = num_players
        self.stages = stages or DEFAULT_STAGES
        self.state_path = state_path
        self.mixing_enabled = mixing_enabled
        self.mixing_ratio = mixing_ratio

        # Load or initialize state
        self.state = self._load_state() or CurriculumState()

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return CurriculumStage(self.state.current_stage)

    @property
    def current_config(self) -> StageConfig:
        """Get current stage configuration."""
        return self.stages[self.current_stage]

    def _load_state(self) -> Optional[CurriculumState]:
        """Load state from disk."""
        if self.state_path and self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                return CurriculumState.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load curriculum state: {e}")
        return None

    def save_state(self):
        """Save state to disk."""
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)

    def filter_samples_by_stage(
        self,
        samples: List[Dict[str, Any]],
        stage: CurriculumStage,
    ) -> List[Dict[str, Any]]:
        """Filter samples appropriate for a curriculum stage.

        Args:
            samples: List of training samples with 'move_idx' and 'total_moves'
            stage: Stage to filter for

        Returns:
            Filtered samples
        """
        config = self.stages[stage]

        filtered = []
        for sample in samples:
            move_idx = sample.get("move_idx", 0)
            total_moves = sample.get("total_moves", 100)

            # Check move range
            if not (config.min_move <= move_idx <= config.max_move):
                continue

            # Check complexity range
            complexity = estimate_position_complexity(move_idx, total_moves)
            if not (config.complexity_range[0] <= complexity <= config.complexity_range[1]):
                continue

            filtered.append(sample)

        return filtered

    def get_training_batch(
        self,
        all_samples: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> List[Dict[str, Any]]:
        """Get a training batch appropriate for current curriculum stage.

        Args:
            all_samples: All available training samples
            batch_size: Desired batch size

        Returns:
            Batch of samples for training
        """
        current_stage = self.current_stage
        current_idx = STAGE_ORDER.index(current_stage)

        # Get samples for current stage
        current_samples = self.filter_samples_by_stage(all_samples, current_stage)

        if not current_samples:
            logger.warning(f"No samples for stage {current_stage.value}, using all samples")
            current_samples = all_samples

        # Mix in samples from previous stages
        if self.mixing_enabled and current_idx > 0:
            prev_samples = []
            for i in range(current_idx):
                stage = STAGE_ORDER[i]
                stage_samples = self.filter_samples_by_stage(all_samples, stage)
                prev_samples.extend(stage_samples)

            if prev_samples:
                num_prev = int(batch_size * self.mixing_ratio)
                num_current = batch_size - num_prev

                # Sample from each pool
                current_batch = random.sample(
                    current_samples, min(num_current, len(current_samples))
                )
                prev_batch = random.sample(
                    prev_samples, min(num_prev, len(prev_samples))
                )

                return current_batch + prev_batch

        # No mixing or first stage - just sample from current
        return random.sample(current_samples, min(batch_size, len(current_samples)))

    def update_progress(
        self,
        val_loss: float,
        val_accuracy: float,
        policy_accuracy: Optional[float] = None,
    ) -> bool:
        """Update curriculum progress after an epoch.

        Args:
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            policy_accuracy: Policy prediction accuracy (optional)

        Returns:
            True if advanced to next stage
        """
        self.state.stage_epochs += 1
        self.state.total_epochs += 1

        # Track best metrics
        if val_accuracy > self.state.stage_best_accuracy:
            self.state.stage_best_accuracy = val_accuracy
        if val_loss < self.state.stage_best_loss:
            self.state.stage_best_loss = val_loss

        # Record history
        self.state.stage_history.append({
            "epoch": self.state.stage_epochs,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "policy_accuracy": policy_accuracy,
            "timestamp": datetime.now().isoformat(),
        })

        # Check for advancement
        config = self.current_config
        should_advance = (
            self.state.stage_epochs >= config.min_epochs
            and val_accuracy >= config.target_accuracy
        )

        if should_advance:
            return self.advance_stage()

        self.save_state()
        return False

    def advance_stage(self) -> bool:
        """Advance to the next curriculum stage.

        Returns:
            True if successfully advanced, False if already at final stage
        """
        current_idx = STAGE_ORDER.index(self.current_stage)

        if current_idx >= len(STAGE_ORDER) - 1:
            logger.info("Already at final curriculum stage")
            return False

        next_stage = STAGE_ORDER[current_idx + 1]

        # Record advancement
        self.state.advancement_history.append({
            "from_stage": self.current_stage.value,
            "to_stage": next_stage.value,
            "epochs_in_stage": self.state.stage_epochs,
            "best_accuracy": self.state.stage_best_accuracy,
            "best_loss": self.state.stage_best_loss,
            "timestamp": datetime.now().isoformat(),
        })

        # Reset stage tracking
        self.state.current_stage = next_stage.value
        self.state.stage_epochs = 0
        self.state.stage_best_accuracy = 0.0
        self.state.stage_best_loss = float("inf")
        self.state.stage_history = []

        logger.info(f"Advanced curriculum to stage: {next_stage.value}")
        self.save_state()

        return True

    def reset(self):
        """Reset curriculum to initial stage."""
        self.state = CurriculumState()
        self.save_state()

    def get_sample_weights(
        self,
        samples: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Get sample weights based on curriculum stage.

        Weights are higher for samples matching current stage's complexity.

        Args:
            samples: Training samples

        Returns:
            Array of weights for each sample
        """
        config = self.current_config
        weights = np.ones(len(samples))

        for i, sample in enumerate(samples):
            move_idx = sample.get("move_idx", 0)
            total_moves = sample.get("total_moves", 100)
            complexity = estimate_position_complexity(move_idx, total_moves)

            # Higher weight if complexity is in target range
            min_c, max_c = config.complexity_range
            if min_c <= complexity <= max_c:
                weights[i] = config.sample_weight
            else:
                # Gradual falloff for out-of-range complexity
                if complexity < min_c:
                    weights[i] = max(0.1, 1.0 - (min_c - complexity) * 2)
                else:
                    weights[i] = max(0.1, 1.0 - (complexity - max_c) * 2)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        return weights

    def get_status(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            "current_stage": self.current_stage.value,
            "stage_name": self.current_config.name,
            "stage_epochs": self.state.stage_epochs,
            "total_epochs": self.state.total_epochs,
            "stage_best_accuracy": self.state.stage_best_accuracy,
            "target_accuracy": self.current_config.target_accuracy,
            "min_epochs_remaining": max(
                0, self.current_config.min_epochs - self.state.stage_epochs
            ),
            "stages_completed": STAGE_ORDER.index(self.current_stage),
            "total_stages": len(STAGE_ORDER),
        }


class AdaptiveCurriculumManager(CurriculumManager):
    """Curriculum manager that adapts difficulty based on real-time performance."""

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        window_size: int = 5,
        difficulty_step: float = 0.1,
        **kwargs,
    ):
        """
        Initialize adaptive curriculum.

        Args:
            board_type: Board type
            num_players: Number of players
            window_size: Number of epochs to average for difficulty adjustment
            difficulty_step: How much to adjust difficulty per step
        """
        super().__init__(board_type, num_players, **kwargs)
        self.window_size = window_size
        self.difficulty_step = difficulty_step
        self.recent_accuracies: List[float] = []
        self.current_difficulty: float = 0.0  # 0 = easiest, 1 = hardest

    def update_progress(
        self,
        val_loss: float,
        val_accuracy: float,
        policy_accuracy: Optional[float] = None,
    ) -> bool:
        """Update with adaptive difficulty adjustment."""
        # Track recent accuracies
        self.recent_accuracies.append(val_accuracy)
        if len(self.recent_accuracies) > self.window_size:
            self.recent_accuracies.pop(0)

        # Adjust difficulty based on performance
        if len(self.recent_accuracies) >= self.window_size:
            avg_accuracy = np.mean(self.recent_accuracies)

            # If doing well, increase difficulty
            if avg_accuracy > self.current_config.target_accuracy + 0.05:
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
                logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")

            # If struggling, decrease difficulty
            elif avg_accuracy < self.current_config.target_accuracy - 0.10:
                self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_step)
                logger.info(f"Decreased difficulty to {self.current_difficulty:.2f}")

        return super().update_progress(val_loss, val_accuracy, policy_accuracy)

    def get_difficulty_adjusted_complexity(self) -> Tuple[float, float]:
        """Get complexity range adjusted for current difficulty."""
        base_min, base_max = self.current_config.complexity_range

        # Shift complexity range based on difficulty
        shift = self.current_difficulty * 0.3  # Max shift of 0.3
        adjusted_min = min(1.0, base_min + shift)
        adjusted_max = min(1.0, base_max + shift)

        return (adjusted_min, adjusted_max)


def create_curriculum_for_config(
    board_type: str,
    num_players: int,
    log_dir: Optional[Path] = None,
    adaptive: bool = True,
) -> CurriculumManager:
    """Factory function to create appropriate curriculum manager.

    Args:
        board_type: Board type
        num_players: Number of players
        log_dir: Directory for state persistence
        adaptive: Whether to use adaptive difficulty

    Returns:
        CurriculumManager instance
    """
    config_key = f"{board_type}_{num_players}p"

    state_path = None
    if log_dir:
        state_path = log_dir / f"curriculum_{config_key}.json"

    if adaptive:
        return AdaptiveCurriculumManager(
            board_type=board_type,
            num_players=num_players,
            state_path=state_path,
        )
    else:
        return CurriculumManager(
            board_type=board_type,
            num_players=num_players,
            state_path=state_path,
        )
