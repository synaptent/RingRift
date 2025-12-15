"""
Curriculum training for RingRift neural networks.

Implements iterative self-play training with:
- Generation-based training loops
- Model evaluation and promotion
- Historical data mixing
- Board-type-specific training
- Position complexity estimation (merged from curriculum_learning.py)
- Stage-based sample filtering (merged from curriculum_learning.py)
- Adaptive difficulty adjustment (merged from curriculum_learning.py)

Canonical curriculum runs
-------------------------

This module is the single source of truth for **self-play curriculum
configuration**, complementing:

- ``ai-service/app/config/ladder_config.py`` – canonical difficulty ladder
  (D1–D10) per ``(difficulty, board_type, num_players)``.
- ``docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md`` – human-facing calibration
  procedures for D2/D4/D6/D8 on square8 2‑player.

The current canonical curriculum slices are:

- **Square‑8, 2‑player** (primary production focus)
  - ``board_type = BoardType.SQUARE8``
  - ``num_players = 2``
  - Default curriculum hyperparameters come from ``CurriculumConfig``.
- **Square‑19, 2‑player** (experimental)
  - ``board_type = BoardType.SQUARE19``
  - ``num_players = 2``
- **Hexagonal, 2‑player** (experimental)
  - ``board_type = BoardType.HEXAGONAL``
  - ``num_players = 2``

All of these share the same CurriculumConfig knobs (generations, games per
generation, eval_games, max_moves, engine mix, etc.). For auditability,
each run writes its resolved configuration to ``<run_dir>/config.json`` via
``CurriculumTrainer._save_config()`` so that training parameters can be
inspected and reproduced later.

To launch a canonical square8 2‑player curriculum run with the defaults:

.. code-block:: bash

    cd ai-service
    python -m app.training.curriculum \\
      --board-type square8 \\
      --generations 10 \\
      --games-per-gen 1000 \\
      --eval-games 100 \\
      --output-dir curriculum_runs/square8_2p

These defaults are intentionally conservative and match the values used by
initial curriculum experiments; they can be tuned per run by overriding the
corresponding CLI flags, but every change will still be recorded in the
run's ``config.json`` for later analysis.
"""

import json
import logging
import random
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.models import BoardType
from app.training.config import TrainConfig
from app.training.significance import wilson_lower_bound

logger = logging.getLogger(__name__)


# =============================================================================
# Position Complexity & Stage-Based Curriculum (merged from curriculum_learning.py)
# =============================================================================


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


def filter_samples_by_complexity(
    samples: List[Dict[str, Any]],
    stage: CurriculumStage,
    stages: Optional[Dict[CurriculumStage, StageConfig]] = None,
) -> List[Dict[str, Any]]:
    """Filter training samples appropriate for a curriculum stage.

    Args:
        samples: List of training samples with 'move_idx' and 'total_moves'
        stage: Stage to filter for
        stages: Custom stage configurations (uses DEFAULT_STAGES if None)

    Returns:
        Filtered samples
    """
    stages = stages or DEFAULT_STAGES
    config = stages[stage]

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


def get_sample_weights_by_complexity(
    samples: List[Dict[str, Any]],
    stage: CurriculumStage,
    stages: Optional[Dict[CurriculumStage, StageConfig]] = None,
) -> np.ndarray:
    """Get sample weights based on curriculum stage complexity.

    Weights are higher for samples matching current stage's complexity.

    Args:
        samples: Training samples
        stage: Current curriculum stage
        stages: Custom stage configurations

    Returns:
        Array of weights for each sample
    """
    stages = stages or DEFAULT_STAGES
    config = stages[stage]
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


# =============================================================================
# Generation-Based Curriculum Configuration
# =============================================================================


@dataclass
class CurriculumConfig:
    """Configuration for curriculum training."""

    # Board type for this training run (separate models per board type)
    board_type: BoardType = BoardType.SQUARE8

    # Number of curriculum generations to train
    generations: int = 10

    # Self-play games per generation
    games_per_generation: int = 1000

    # Training epochs per generation
    training_epochs: int = 20

    # Evaluation games for model comparison
    eval_games: int = 100

    # Win rate threshold for promoting a candidate model
    # Must win at least this fraction against current best
    promotion_threshold: float = 0.55

    # Confidence for Wilson lower-bound promotion gate.
    promotion_confidence: float = 0.95

    # Number of past generations of data to keep for training
    # Older data is discarded to prevent overfitting to old play patterns
    data_retention: int = 3

    # Exponential decay factor for weighting historical data
    # Recent data gets weight 1.0, older generations get weight * decay^age
    historical_decay: float = 0.8

    # Number of players for self-play games
    num_players: int = 2

    # Maximum moves per self-play game
    max_moves: int = 10000

    # AI configuration for self-play
    think_time_ms: int = 500
    difficulty: int = 10

    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4

    # Random seed (incremented per generation)
    base_seed: int = 42

    # Output directory for curriculum artifacts
    output_dir: str = "curriculum_runs"

    # Model identifier prefix
    model_prefix: str = "ringrift"

    # Engine configuration for self-play data generation
    # Default engine type: "descent" or "mcts"
    engine: str = "descent"

    # Engine mixing strategy: "single", "per_game", or "per_player"
    # - "single": all players use the same engine (specified by `engine`)
    # - "per_game": randomly choose engine per game (ratio controlled by engine_ratio)
    # - "per_player": randomly choose engine per player within a game
    engine_mix: str = "single"

    # Ratio of MCTS usage when engine_mix != "single"
    # 0.0 = all Descent, 1.0 = all MCTS, 0.5 = 50/50
    engine_ratio: float = 0.5

    # Neural network model ID for self-play (curriculum trainer manages this internally)
    # This is typically set by the trainer to point to the current best model
    nn_model_id: Optional[str] = None

    # Model pool competition settings
    # If True, evaluate candidates against a pool of historical models
    # instead of just the current best. This provides more robust evaluation.
    use_model_pool: bool = False

    # Maximum number of models to keep in the pool
    # Older models are removed when the pool exceeds this size
    model_pool_size: int = 5

    # Number of games to play against each model in the pool
    # Total eval games = pool_eval_games * len(pool)
    pool_eval_games: int = 20

    # Minimum win rate against the pool to be promoted
    # This replaces promotion_threshold when use_model_pool=True
    pool_promotion_threshold: float = 0.55

    # Adaptive evaluation settings
    # When enabled, evaluation runs until statistically decisive rather than fixed games
    use_adaptive_eval: bool = True
    adaptive_min_games: int = 30  # Minimum games before early stopping
    adaptive_max_games: int = 300  # Maximum games to play
    adaptive_batch_size: int = 20  # Games per batch
    adaptive_ci_width_target: float = 0.04  # Target CI width (±2%)

    def get_model_id(self) -> str:
        """Get model identifier for this board type."""
        return f"{self.model_prefix}_{self.board_type.value}"


@dataclass
class GenerationResult:
    """Results from a single curriculum generation."""

    generation: int
    promoted: bool
    win_rate: float
    draw_rate: float
    loss_rate: float
    games_played: int
    avg_game_length: float
    training_loss: float
    policy_loss: float
    value_loss: float
    training_time_sec: float
    eval_time_sec: float
    data_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CurriculumTrainer:
    """
    Curriculum training manager for iterative self-play improvement.

    Implements the AlphaZero-style training loop:
    1. Generate self-play data with current best model
    2. Train candidate model on recent + historical data
    3. Evaluate candidate against current best
    4. Promote candidate if it wins majority of games
    5. Repeat for N generations
    """

    def __init__(
        self,
        config: CurriculumConfig,
        base_model_path: Optional[str] = None,
    ):
        """
        Initialize curriculum trainer.

        Parameters
        ----------
        config : CurriculumConfig
            Training configuration.
        base_model_path : Optional[str]
            Path to initial model checkpoint. If None, starts with random weights.
        """
        self.config = config
        self.base_model_path = base_model_path
        self.current_generation = 0
        self.history: List[GenerationResult] = []

        # Setup output directories
        self.run_dir = Path(config.output_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.run_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Current best model path
        if base_model_path:
            self.current_best_path = base_model_path
        else:
            self.current_best_path = None

        # Wire nn_model_id used by generate_data to a stable, curriculum‑scoped
        # identifier so NeuralNetAI can resolve checkpoints via models/<id>.pth.
        # This keeps curriculum models isolated from the default "ringrift_v4_*"
        # lineage used by ladder tiers.
        self.config.nn_model_id = self.config.get_model_id()

        # Ensure the initial best model (when provided) is visible to
        # NeuralNetAI under models/<nn_model_id>.pth so that self‑play uses
        # the intended teacher rather than the global default.
        if self.current_best_path is not None:
            self._sync_current_best_to_models_dir()

        # Save config
        self._save_config()

        # Model pool for pool-based evaluation
        # Each entry is (model_path, generation) - older models at front
        self.model_pool: List[Tuple[str, int]] = []

        # Add initial model to pool if provided and pool mode is enabled
        if config.use_model_pool and base_model_path:
            self.model_pool.append((base_model_path, -1))  # gen=-1 for initial

        logger.info(
            "Initialized curriculum trainer: board=%s, generations=%d, "
            "games_per_gen=%d, eval_games=%d, model_pool=%s",
            config.board_type.value,
            config.generations,
            config.games_per_generation,
            config.eval_games,
            "enabled" if config.use_model_pool else "disabled",
        )

    def _save_config(self) -> None:
        """Save configuration to run directory."""
        config_path = self.run_dir / "config.json"
        config_dict = {
            "board_type": self.config.board_type.value,
            "generations": self.config.generations,
            "games_per_generation": self.config.games_per_generation,
            "training_epochs": self.config.training_epochs,
            "eval_games": self.config.eval_games,
            "promotion_threshold": self.config.promotion_threshold,
            "promotion_confidence": self.config.promotion_confidence,
            "data_retention": self.config.data_retention,
            "historical_decay": self.config.historical_decay,
            "num_players": self.config.num_players,
            "max_moves": self.config.max_moves,
            "think_time_ms": self.config.think_time_ms,
            "difficulty": self.config.difficulty,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "base_seed": self.config.base_seed,
            "base_model_path": self.base_model_path,
            # Engine diversity settings
            "engine": self.config.engine,
            "engine_mix": self.config.engine_mix,
            "engine_ratio": self.config.engine_ratio,
            # Model pool settings
            "use_model_pool": self.config.use_model_pool,
            "model_pool_size": self.config.model_pool_size,
            "pool_eval_games": self.config.pool_eval_games,
            "pool_promotion_threshold": self.config.pool_promotion_threshold,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _models_root_dir(self) -> Path:
        """Return the repository‑level models directory used by NeuralNetAI."""
        # ai-service/app/training/curriculum.py -> training/ -> app/ -> ai-service/
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def _sync_current_best_to_models_dir(self) -> None:
        """
        Copy the current best checkpoint into the global models/ directory.

        NeuralNetAI resolves checkpoints via ``config.nn_model_id`` as
        ``<repo_root>/models/<nn_model_id>.pth`` (or ``*_mps.pth`` for
        MPS‑specific builds). For curriculum self‑play we keep things
        simple and always publish the best model as ``<id>.pth``, where
        ``id`` comes from CurriculumConfig.get_model_id().
        """
        if not self.current_best_path:
            return

        src = Path(self.current_best_path)
        if not src.exists():
            logger.warning(
                "Current best model path does not exist, skipping sync: %s",
                src,
            )
            return

        model_id = self.config.get_model_id()
        dst = self._models_root_dir() / f"{model_id}.pth"

        try:
            shutil.copy(src, dst)
            logger.info(
                "Synced current best model to %s for self‑play (nn_model_id=%s)",
                dst,
                model_id,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to sync current best model to %s: %s",
                dst,
                exc,
            )

    def run(self) -> List[GenerationResult]:
        """
        Run full curriculum training loop.

        Returns
        -------
        List[GenerationResult]
            Results from each generation.
        """
        logger.info("Starting curriculum training for %d generations", self.config.generations)

        for gen in range(self.config.generations):
            logger.info("=" * 60)
            logger.info("GENERATION %d / %d", gen + 1, self.config.generations)
            logger.info("=" * 60)

            result = self.run_generation(gen)
            self.history.append(result)

            # Save progress
            self._save_history()

            if result.promoted:
                logger.info(
                    "Generation %d: PROMOTED (win_rate=%.2f%%)",
                    gen,
                    result.win_rate * 100,
                )
            else:
                logger.info(
                    "Generation %d: NOT PROMOTED (win_rate=%.2f%% < %.2f%%)",
                    gen,
                    result.win_rate * 100,
                    self.config.promotion_threshold * 100,
                )

        logger.info("Curriculum training complete. Final generation: %d", self.current_generation)
        return self.history

    def run_generation(self, generation: int) -> GenerationResult:
        """
        Run a single generation of curriculum training.

        Parameters
        ----------
        generation : int
            Generation number (0-indexed).

        Returns
        -------
        GenerationResult
            Results from this generation.
        """
        gen_seed = self.config.base_seed + generation * 1000

        # 1. Generate self-play data
        logger.info("[Gen %d] Generating self-play data...", generation)
        data_start = time.time()
        data_path, num_samples = self._generate_self_play_data(generation, gen_seed)
        data_time = time.time() - data_start
        logger.info(
            "[Gen %d] Generated %d samples in %.1fs",
            generation,
            num_samples,
            data_time,
        )

        # 2. Combine with historical data
        logger.info("[Gen %d] Combining with historical data...", generation)
        combined_data_path = self._combine_with_history(generation)

        # 3. Train candidate model
        logger.info("[Gen %d] Training candidate model...", generation)
        train_start = time.time()
        candidate_path, train_losses = self._train_candidate(
            generation,
            combined_data_path,
            gen_seed,
        )
        train_time = time.time() - train_start
        logger.info(
            "[Gen %d] Training complete in %.1fs (loss=%.4f)",
            generation,
            train_time,
            train_losses["total"],
        )

        # 4. Evaluate candidate against current best
        logger.info("[Gen %d] Evaluating candidate...", generation)
        eval_start = time.time()
        eval_result = self._evaluate_candidate(candidate_path, gen_seed)
        eval_time = time.time() - eval_start
        logger.info(
            "[Gen %d] Evaluation: win=%.2f%%, draw=%.2f%%, loss=%.2f%%",
            generation,
            eval_result["win_rate"] * 100,
            eval_result["draw_rate"] * 100,
            eval_result["loss_rate"] * 100,
        )

        # 5. Decide on promotion
        # Use pool threshold when pool mode is enabled, otherwise standard threshold
        if self.config.use_model_pool and len(self.model_pool) > 0:
            threshold = self.config.pool_promotion_threshold
        else:
            threshold = self.config.promotion_threshold

        wins = int(eval_result.get("wins", 0))
        losses = int(eval_result.get("losses", 0))
        decisive_games = wins + losses
        ci_low = None
        if decisive_games > 0:
            ci_low = wilson_lower_bound(
                wins,
                decisive_games,
                confidence=self.config.promotion_confidence,
            )

        promoted = eval_result["win_rate"] >= threshold
        # Require statistical confidence when we have a real opponent.
        if decisive_games > 0 and ci_low is not None:
            promoted = promoted and ci_low >= threshold
        if promoted:
            self._promote_candidate(candidate_path, generation)
            self.current_generation = generation + 1

        # Build result
        result = GenerationResult(
            generation=generation,
            promoted=promoted,
            win_rate=eval_result["win_rate"],
            draw_rate=eval_result["draw_rate"],
            loss_rate=eval_result["loss_rate"],
            games_played=eval_result["games_played"],
            avg_game_length=eval_result["avg_game_length"],
            training_loss=train_losses["total"],
            policy_loss=train_losses["policy"],
            value_loss=train_losses["value"],
            training_time_sec=train_time,
            eval_time_sec=eval_time,
            data_samples=num_samples,
        )

        return result

    def _generate_self_play_data(
        self,
        generation: int,
        seed: int,
    ) -> Tuple[Path, int]:
        """Generate self-play training data.

        Uses the engine configuration from CurriculumConfig to support:
        - Single engine mode: all players use the same engine type
        - Per-game mixing: randomly choose engine per game
        - Per-player mixing: randomly choose engine per player
        """
        from app.training.generate_data import generate_dataset

        output_path = self.data_dir / f"gen{generation:03d}_selfplay.npz"

        # Log engine configuration for this generation
        if self.config.engine_mix != "single":
            logger.info(
                "[Gen %d] Engine mixing: %s (ratio=%.2f)",
                generation,
                self.config.engine_mix,
                self.config.engine_ratio,
            )
        else:
            logger.info("[Gen %d] Engine: %s", generation, self.config.engine)

        # Use generate_dataset with engine mixing support.
        # AI instances are created internally based on engine/engine_mix/engine_ratio.
        #
        # Note: For curriculum training, ideally we'd use the current_best model checkpoint.
        # However, the AI classes expect a model_id (not a path). For now, if nn_model_id
        # is set in config, we use it; otherwise AI uses default model loading.
        # TODO: Add proper checkpoint path support to AI classes for curriculum training.
        generate_dataset(
            num_games=self.config.games_per_generation,
            output_file=str(output_path),
            board_type=self.config.board_type,
            seed=seed,
            max_moves=self.config.max_moves,
            num_players=self.config.num_players,
            engine=self.config.engine,
            engine_mix=self.config.engine_mix,
            engine_ratio=self.config.engine_ratio,
            nn_model_id=self.config.nn_model_id,
        )

        # Count samples
        data = np.load(output_path, allow_pickle=True)
        num_samples = len(data["values"])

        return output_path, num_samples

    def _combine_with_history(self, generation: int) -> Path:
        """Combine current generation data with historical data."""
        combined_path = self.data_dir / f"gen{generation:03d}_combined.npz"

        # Find recent generations to include
        start_gen = max(0, generation - self.config.data_retention + 1)
        data_files = []
        weights = []

        for g in range(start_gen, generation + 1):
            path = self.data_dir / f"gen{g:03d}_selfplay.npz"
            if path.exists():
                age = generation - g
                weight = self.config.historical_decay ** age
                data_files.append(path)
                weights.append(weight)
                logger.debug(
                    "Including gen %d data (age=%d, weight=%.2f)",
                    g,
                    age,
                    weight,
                )

        if not data_files:
            logger.warning("No data files found for combination")
            return combined_path

        # Load and combine
        all_features = []
        all_globals = []
        all_values = []
        all_policy_indices = []
        all_policy_values = []
        all_weights = []

        for path, weight in zip(data_files, weights):
            data = np.load(path, allow_pickle=True)
            n = len(data["values"])

            all_features.append(data["features"])
            all_globals.append(data["globals"])
            all_values.append(data["values"])
            all_policy_indices.append(data["policy_indices"])
            all_policy_values.append(data["policy_values"])
            all_weights.append(np.full(n, weight, dtype=np.float32))

        # Concatenate
        combined = {
            "features": np.concatenate(all_features),
            "globals": np.concatenate(all_globals),
            "values": np.concatenate(all_values),
            "policy_indices": np.concatenate(all_policy_indices),
            "policy_values": np.concatenate(all_policy_values),
            "sample_weights": np.concatenate(all_weights),
        }

        np.savez_compressed(combined_path, **combined)

        logger.info(
            "Combined %d samples from %d generations",
            len(combined["values"]),
            len(data_files),
        )

        return combined_path

    def _train_candidate(
        self,
        generation: int,
        data_path: Path,
        seed: int,
    ) -> Tuple[Path, Dict[str, float]]:
        """Train a candidate model on combined data."""
        from app.training.train import train_from_file

        candidate_path = self.models_dir / f"gen{generation:03d}_candidate.pth"

        train_config = TrainConfig(
            board_type=self.config.board_type,
            epochs_per_iter=self.config.training_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            weight_decay=self.config.weight_decay,
            seed=seed,
            model_id=f"gen{generation:03d}_candidate",
        )

        # Initialize from current best if available
        initial_model = self.current_best_path

        losses = train_from_file(
            data_path=str(data_path),
            output_path=str(candidate_path),
            config=train_config,
            initial_model_path=initial_model,
        )

        return candidate_path, losses

    def _evaluate_candidate(
        self,
        candidate_path: Path,
        seed: int,
    ) -> Dict[str, Any]:
        """Evaluate candidate model against current best or model pool."""
        # Use pool-based evaluation if enabled and pool has models
        if self.config.use_model_pool and len(self.model_pool) > 0:
            return self._evaluate_against_pool(candidate_path, seed)

        return self._evaluate_against_single(candidate_path, seed)

    def _evaluate_against_single(
        self,
        candidate_path: Path,
        seed: int,
    ) -> Dict[str, Any]:
        """Evaluate candidate model against current best (standard mode).

        Uses adaptive evaluation when config.use_adaptive_eval is True,
        which runs games until statistically decisive rather than a fixed count.
        This approach:
        - Promotes obvious winners quickly (30-50 games)
        - Rejects obvious losers quickly (30-50 games)
        - Runs more games for marginal cases (200-300)
        """
        from app.training.tournament import run_tournament, run_tournament_adaptive

        # If no current best, candidate auto-wins
        if self.current_best_path is None:
            return {
                "win_rate": 1.0,
                "draw_rate": 0.0,
                "loss_rate": 0.0,
                "games_played": 0,
                "avg_game_length": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
            }

        # Use adaptive evaluation when enabled
        if self.config.use_adaptive_eval:
            results = run_tournament_adaptive(
                model_a_path=str(candidate_path),
                model_b_path=self.current_best_path,
                promotion_threshold=self.config.promotion_threshold,
                confidence=self.config.promotion_confidence,
                min_games=self.config.adaptive_min_games,
                max_games=self.config.adaptive_max_games,
                batch_size=self.config.adaptive_batch_size,
                ci_width_target=self.config.adaptive_ci_width_target,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                max_moves=self.config.max_moves,
                seed=seed,
            )
            logger.info(
                f"Adaptive eval: {results['total_games']} games, "
                f"win_rate={results['win_rate']:.1%}, "
                f"CI=[{results['ci_lower']:.1%}, {results['ci_upper']:.1%}], "
                f"stop_reason={results['stop_reason']}"
            )
        else:
            # Run fixed tournament: candidate vs current_best
            results = run_tournament(
                model_a_path=str(candidate_path),
                model_b_path=self.current_best_path,
                num_games=self.config.eval_games,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                max_moves=self.config.max_moves,
                seed=seed,
            )

        total_games = max(1, results["total_games"])
        wins = results["model_a_wins"]
        losses = results["model_b_wins"]
        draws = results["draws"]
        return {
            "win_rate": wins / total_games,
            "draw_rate": draws / total_games,
            "loss_rate": losses / total_games,
            "games_played": results["total_games"],
            "avg_game_length": results.get("avg_game_length", 0),
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    def _evaluate_against_pool(
        self,
        candidate_path: Path,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate model against a pool of historical models.

        Plays eval_games against each model in the pool and aggregates
        the results. This provides more robust evaluation than comparing
        against a single opponent.
        """
        from app.training.tournament import run_tournament

        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_games = 0
        total_game_length = 0.0
        per_opponent_results: List[Dict[str, Any]] = []

        logger.info(
            "Evaluating candidate against model pool (size=%d)",
            len(self.model_pool),
        )

        for idx, (opponent_path, opponent_gen) in enumerate(self.model_pool):
            # Each opponent gets a different seed offset for variety
            opponent_seed = seed + idx * 100

            results = run_tournament(
                model_a_path=str(candidate_path),
                model_b_path=opponent_path,
                num_games=self.config.pool_eval_games,
                board_type=self.config.board_type,
                num_players=self.config.num_players,
                max_moves=self.config.max_moves,
                seed=opponent_seed,
            )

            wins = results["model_a_wins"]
            draws = results["draws"]
            losses = results["model_b_wins"]
            games = results["total_games"]
            avg_len = results.get("avg_game_length", 0)

            total_wins += wins
            total_draws += draws
            total_losses += losses
            total_games += games
            total_game_length += avg_len * games

            per_opponent_results.append({
                "opponent_gen": opponent_gen,
                "win_rate": wins / max(1, games),
                "games": games,
            })

            logger.info(
                "  vs gen %d: win_rate=%.1f%% (%d/%d games)",
                opponent_gen,
                100 * wins / max(1, games),
                wins,
                games,
            )

        overall_win_rate = total_wins / max(1, total_games)
        overall_draw_rate = total_draws / max(1, total_games)
        overall_loss_rate = total_losses / max(1, total_games)
        overall_avg_len = total_game_length / max(1, total_games)

        logger.info(
            "Pool evaluation complete: overall win_rate=%.1f%% (%d games)",
            100 * overall_win_rate,
            total_games,
        )

        return {
            "win_rate": overall_win_rate,
            "draw_rate": overall_draw_rate,
            "loss_rate": overall_loss_rate,
            "games_played": total_games,
            "avg_game_length": overall_avg_len,
            "pool_size": len(self.model_pool),
            "per_opponent": per_opponent_results,
            "wins": total_wins,
            "losses": total_losses,
            "draws": total_draws,
        }

    def _promote_candidate(self, candidate_path: Path, generation: int) -> None:
        """Promote candidate to current best and update model pool."""
        best_path = self.models_dir / f"best_{self.config.get_model_id()}.pth"

        # Copy candidate to best
        shutil.copy(candidate_path, best_path)
        self.current_best_path = str(best_path)

        # Ensure NeuralNetAI can load the promoted model via nn_model_id.
        self._sync_current_best_to_models_dir()

        # Also save as versioned checkpoint
        versioned_path = self.models_dir / f"gen{generation:03d}_promoted.pth"
        shutil.copy(candidate_path, versioned_path)

        # Add to model pool if pool-based evaluation is enabled
        if self.config.use_model_pool:
            self._add_to_model_pool(str(versioned_path), generation)

        logger.info(
            "Promoted generation %d model to %s",
            generation,
            best_path,
        )

    def _add_to_model_pool(self, model_path: str, generation: int) -> None:
        """
        Add a model to the evaluation pool.

        Removes oldest models if pool exceeds configured size.
        """
        self.model_pool.append((model_path, generation))

        # Trim pool to max size (remove oldest models from front)
        while len(self.model_pool) > self.config.model_pool_size:
            removed = self.model_pool.pop(0)
            logger.info(
                "Removed gen %d from model pool (pool size: %d)",
                removed[1],
                len(self.model_pool),
            )

        logger.info(
            "Added gen %d to model pool (pool size: %d)",
            generation,
            len(self.model_pool),
        )

    def _save_history(self) -> None:
        """Save training history to disk."""
        history_path = self.run_dir / "history.json"
        history_data = [
            {
                "generation": r.generation,
                "promoted": r.promoted,
                "win_rate": r.win_rate,
                "draw_rate": r.draw_rate,
                "loss_rate": r.loss_rate,
                "games_played": r.games_played,
                "avg_game_length": r.avg_game_length,
                "training_loss": r.training_loss,
                "policy_loss": r.policy_loss,
                "value_loss": r.value_loss,
                "training_time_sec": r.training_time_sec,
                "eval_time_sec": r.eval_time_sec,
                "data_samples": r.data_samples,
                "timestamp": r.timestamp,
            }
            for r in self.history
        ]
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)


def run_curriculum_training(
    board_type: BoardType = BoardType.SQUARE8,
    generations: int = 10,
    games_per_generation: int = 1000,
    base_model_path: Optional[str] = None,
    output_dir: str = "curriculum_runs",
) -> List[GenerationResult]:
    """
    Convenience function to run curriculum training.

    Parameters
    ----------
    board_type : BoardType
        Board type to train on.
    generations : int
        Number of curriculum generations.
    games_per_generation : int
        Self-play games per generation.
    base_model_path : Optional[str]
        Initial model to start from.
    output_dir : str
        Output directory for artifacts.

    Returns
    -------
    List[GenerationResult]
        Results from each generation.
    """
    config = CurriculumConfig(
        board_type=board_type,
        generations=generations,
        games_per_generation=games_per_generation,
        output_dir=output_dir,
    )

    trainer = CurriculumTrainer(config, base_model_path)
    return trainer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run curriculum training")
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hexagonal"],
        default="square8",
        help="Board type to train on",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of curriculum generations",
    )
    parser.add_argument(
        "--games-per-gen",
        type=int,
        default=1000,
        help="Self-play games per generation",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="Evaluation games for promotion",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to initial model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="curriculum_runs",
        help="Output directory",
    )
    # Model pool arguments
    parser.add_argument(
        "--use-model-pool",
        action="store_true",
        help="Enable model pool evaluation (evaluate against historical models)",
    )
    parser.add_argument(
        "--model-pool-size",
        type=int,
        default=5,
        help="Maximum models in the evaluation pool (default: 5)",
    )
    parser.add_argument(
        "--pool-eval-games",
        type=int,
        default=20,
        help="Games per opponent in pool evaluation (default: 20)",
    )
    parser.add_argument(
        "--pool-promotion-threshold",
        type=float,
        default=0.55,
        help="Win rate threshold for pool promotion (default: 0.55)",
    )

    args = parser.parse_args()

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }

    config = CurriculumConfig(
        board_type=board_type_map[args.board_type],
        generations=args.generations,
        games_per_generation=args.games_per_gen,
        eval_games=args.eval_games,
        output_dir=args.output_dir,
        # Model pool settings
        use_model_pool=args.use_model_pool,
        model_pool_size=args.model_pool_size,
        pool_eval_games=args.pool_eval_games,
        pool_promotion_threshold=args.pool_promotion_threshold,
    )

    trainer = CurriculumTrainer(config, args.base_model)
    results = trainer.run()

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 60)
    for r in results:
        status = "PROMOTED" if r.promoted else "skipped"
        print(
            f"Gen {r.generation}: {status} (win={r.win_rate:.1%}, "
            f"loss={r.training_loss:.4f})"
        )
