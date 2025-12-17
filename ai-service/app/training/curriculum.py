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
from typing import Any, Dict, List, Optional

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
    stages_completed: List[str] = field(default_factory=list)
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
        stages: List[CurriculumStage],
        checkpoint_path: Optional[Path] = None,
        auto_advance: bool = True,
    ):
        if not stages:
            raise ValueError("Must provide at least one curriculum stage")

        self.stages = stages
        self.checkpoint_path = checkpoint_path
        self.auto_advance = auto_advance
        self.state = CurriculumState()
        self.state.stage_start_time = time.time()
        self.history: List[Dict[str, Any]] = []

        if checkpoint_path and checkpoint_path.exists():
            self._load_checkpoint()

    def get_current_stage(self) -> CurriculumStage:
        idx = min(self.state.current_stage_idx, len(self.stages) - 1)
        return self.stages[idx]

    def get_stage_parameters(self) -> Dict[str, Any]:
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

    def get_progress(self) -> Dict[str, Any]:
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
