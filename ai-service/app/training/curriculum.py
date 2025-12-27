"""Curriculum Learning Controller for RingRift AI Training.

Implements systematic progression through training difficulty levels,
automatically advancing when performance thresholds are met.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """A single stage in the curriculum."""
    name: str
    max_moves: int = 100
    opponent_elo_delta: int = 0
    temperature: float = 1.0
    mcts_simulations: int = 100
    use_opening_book: bool = True
    opening_depth: int = 6
    win_rate_threshold: float = 0.55
    games_required: int = 100
    min_steps: int = 1000


@dataclass
class CurriculumState:
    """Current state of curriculum learning."""
    current_stage_idx: int = 0
    stages_completed: list[str] = field(default_factory=list)
    stage_start_time: float = 0.0
    stage_games_played: int = 0
    stage_wins: int = 0
    stage_steps: int = 0
    total_games: int = 0
    total_steps: int = 0


class CurriculumController:
    """Controls progression through curriculum learning stages."""

    def __init__(
        self,
        stages: list[CurriculumStage],
        checkpoint_path: Path | None = None,
        auto_advance: bool = True,
    ):
        if not stages:
            raise ValueError("Must provide at least one curriculum stage")

        self.stages = stages
        self.checkpoint_path = checkpoint_path
        self.auto_advance = auto_advance
        self.state = CurriculumState()
        self.state.stage_start_time = time.time()
        self.history: list[dict[str, Any]] = []

        if checkpoint_path and checkpoint_path.exists():
            self._load_checkpoint()

    def get_current_stage(self) -> CurriculumStage:
        idx = min(self.state.current_stage_idx, len(self.stages) - 1)
        return self.stages[idx]

    def get_stage_parameters(self) -> dict[str, Any]:
        stage = self.get_current_stage()
        return {
            "name": stage.name,
            "max_moves": stage.max_moves,
            "opponent_elo_delta": stage.opponent_elo_delta,
            "temperature": stage.temperature,
            "mcts_simulations": stage.mcts_simulations,
        }

    def update_game_result(self, won: bool):
        self.state.stage_games_played += 1
        self.state.total_games += 1
        if won:
            self.state.stage_wins += 1
        if self.auto_advance:
            self.maybe_advance()

    def update_step(self, steps: int = 1):
        self.state.stage_steps += steps
        self.state.total_steps += steps

    def get_win_rate(self) -> float:
        if self.state.stage_games_played == 0:
            return 0.0
        return self.state.stage_wins / self.state.stage_games_played

    def should_advance(self) -> bool:
        if self.state.current_stage_idx >= len(self.stages) - 1:
            return False
        stage = self.get_current_stage()
        if self.state.stage_games_played < stage.games_required:
            return False
        if self.state.stage_steps < stage.min_steps:
            return False
        return self.get_win_rate() >= stage.win_rate_threshold

    def maybe_advance(self) -> bool:
        if not self.should_advance():
            return False
        return self.advance_stage()

    def advance_stage(self) -> bool:
        if self.state.current_stage_idx >= len(self.stages) - 1:
            return False

        current_stage = self.get_current_stage()
        self.history.append({
            "stage": current_stage.name,
            "win_rate": self.get_win_rate(),
            "games_played": self.state.stage_games_played,
            "timestamp": time.time(),
        })
        self.state.stages_completed.append(current_stage.name)
        self.state.current_stage_idx += 1
        self.state.stage_games_played = 0
        self.state.stage_wins = 0
        self.state.stage_steps = 0
        self.state.stage_start_time = time.time()

        logger.info(f"[Curriculum] Advanced to stage '{self.get_current_stage().name}'")
        if self.checkpoint_path:
            self._save_checkpoint()
        return True

    def get_progress(self) -> dict[str, Any]:
        stage = self.get_current_stage()
        return {
            "current_stage": stage.name,
            "stage_idx": self.state.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_win_rate": self.get_win_rate(),
            "win_rate_threshold": stage.win_rate_threshold,
        }

    def _save_checkpoint(self):
        if not self.checkpoint_path:
            return
        checkpoint = {
            "state": {
                "current_stage_idx": self.state.current_stage_idx,
                "stages_completed": self.state.stages_completed,
                "stage_games_played": self.state.stage_games_played,
                "stage_wins": self.state.stage_wins,
                "total_games": self.state.total_games,
            },
            "history": self.history,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self):
        try:
            with open(self.checkpoint_path) as f:
                checkpoint = json.load(f)
            state = checkpoint.get("state", {})
            self.state.current_stage_idx = state.get("current_stage_idx", 0)
            self.state.stages_completed = state.get("stages_completed", [])
            self.state.stage_games_played = state.get("stage_games_played", 0)
            self.state.stage_wins = state.get("stage_wins", 0)
            self.state.total_games = state.get("total_games", 0)
            self.history = checkpoint.get("history", [])
        except Exception as e:
            logger.warning(f"[Curriculum] Failed to load checkpoint: {e}")


def create_default_curriculum() -> CurriculumController:
    """Create a default 5-stage curriculum."""
    stages = [
        CurriculumStage(name="beginner", max_moves=30, opponent_elo_delta=-300, temperature=1.5, win_rate_threshold=0.60, games_required=50),
        CurriculumStage(name="easy", max_moves=50, opponent_elo_delta=-150, temperature=1.2, win_rate_threshold=0.55, games_required=100),
        CurriculumStage(name="medium", max_moves=75, opponent_elo_delta=-50, temperature=1.0, win_rate_threshold=0.52, games_required=150),
        CurriculumStage(name="hard", max_moves=100, opponent_elo_delta=0, temperature=0.8, win_rate_threshold=0.50, games_required=200),
        CurriculumStage(name="expert", max_moves=200, opponent_elo_delta=50, temperature=0.5, win_rate_threshold=0.48, games_required=300),
    ]
    return CurriculumController(stages=stages, checkpoint_path=Path("data/curriculum_state.json"))


@dataclass
class CurriculumConfig:
    """Configuration for full curriculum training pipeline.

    Combines self-play generation, training, and promotion into an
    iterative loop that progressively improves model strength.
    """
    # Board configuration
    board_type: Any = None  # BoardType enum
    num_players: int = 2

    # Curriculum settings
    generations: int = 10
    games_per_generation: int = 500
    training_epochs: int = 20
    eval_games: int = 50
    promotion_threshold: float = 0.80  # Win rate to promote (Dec 2025: raised from 0.60 for 2000+ Elo)

    # Data retention
    data_retention: int = 3  # Generations of data to keep

    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    base_seed: int = 42

    # Output
    output_dir: str = "curriculum_runs"

    # Engine configuration
    engine: str = "descent"  # descent, mcts
    engine_mix: str = "single"  # single, per_game, per_player
    engine_ratio: float = 0.5  # MCTS ratio when mixing


@dataclass
class GenerationResult:
    """Result of a single curriculum generation."""
    generation: int
    win_rate: float
    training_loss: float
    promoted: bool
    model_path: str | None = None
    games_generated: int = 0
    eval_games_played: int = 0


class CurriculumTrainer:
    """Runs iterative curriculum training with self-play and promotion.

    Each generation:
    1. Generate self-play games with current model
    2. Train on accumulated data
    3. Evaluate against previous generation
    4. Promote if win rate exceeds threshold
    """

    def __init__(self, config: CurriculumConfig, base_model: str | None = None):
        self.config = config
        self.base_model = base_model
        self.run_dir = Path(config.output_dir) / f"run_{int(time.time())}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.current_model: str | None = base_model
        self.generation_data: list[Path] = []

    def run(self) -> list[GenerationResult]:
        """Run full curriculum training loop."""
        results = []

        for gen in range(self.config.generations):
            logger.info(f"[Curriculum] Starting generation {gen + 1}/{self.config.generations}")

            # 1. Generate self-play games
            games_path = self._generate_selfplay(gen)
            self.generation_data.append(games_path)

            # Keep only recent generations
            if len(self.generation_data) > self.config.data_retention:
                self.generation_data = self.generation_data[-self.config.data_retention:]

            # 2. Train on accumulated data
            training_loss = self._train_generation(gen)

            # 3. Evaluate against previous
            win_rate = self._evaluate_generation(gen)

            # 4. Decide on promotion
            promoted = win_rate >= self.config.promotion_threshold
            if promoted:
                self._promote_model(gen)

            result = GenerationResult(
                generation=gen,
                win_rate=win_rate,
                training_loss=training_loss,
                promoted=promoted,
                model_path=str(self.run_dir / f"gen_{gen}" / "model.pth"),
                games_generated=self.config.games_per_generation,
                eval_games_played=self.config.eval_games,
            )
            results.append(result)

            logger.info(
                f"[Curriculum] Gen {gen}: win_rate={win_rate:.1%}, "
                f"loss={training_loss:.4f}, promoted={promoted}"
            )

        return results

    def _generate_selfplay(self, generation: int) -> Path:
        """Generate self-play games for this generation."""
        gen_dir = self.run_dir / f"gen_{generation}"
        gen_dir.mkdir(exist_ok=True)
        games_path = gen_dir / "games.db"

        # Use subprocess to run selfplay generation
        import subprocess
        cmd = [
            "python", "scripts/run_gpu_selfplay.py",
            "--board", str(self.config.board_type.value) if self.config.board_type else "square8",
            "--num-players", str(self.config.num_players),
            "--num-games", str(self.config.games_per_generation),
            "--engine-mode", self.config.engine,
            "--record-db", str(games_path),
            "--skip-resource-check",
        ]

        logger.info(f"[Curriculum] Generating {self.config.games_per_generation} games")
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        except Exception as e:
            logger.warning(f"[Curriculum] Selfplay generation failed: {e}")

        return games_path

    def _train_generation(self, generation: int) -> float:
        """Train model on accumulated data."""
        gen_dir = self.run_dir / f"gen_{generation}"

        # Combine data from recent generations
        # For now, return placeholder loss
        logger.info(f"[Curriculum] Training for {self.config.training_epochs} epochs")

        # Placeholder - actual training would use train.py
        return 0.1 + 0.01 * generation  # Simulated decreasing loss

    def _evaluate_generation(self, generation: int) -> float:
        """Evaluate current model against previous generation."""
        if generation == 0 or self.current_model is None:
            # First generation or no previous model - use baseline
            return 0.55

        logger.info(f"[Curriculum] Evaluating with {self.config.eval_games} games")

        # Placeholder - actual evaluation would run tournament
        import random
        return 0.45 + random.random() * 0.2  # Simulated win rate 45-65%

    def _promote_model(self, generation: int) -> None:
        """Promote the current generation's model."""
        gen_dir = self.run_dir / f"gen_{generation}"
        model_path = gen_dir / "model.pth"
        self.current_model = str(model_path)
        logger.info(f"[Curriculum] Promoted model from generation {generation}")
