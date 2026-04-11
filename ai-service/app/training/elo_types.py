"""Shared Elo service types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EloBackendType(str, Enum):
    """Elo storage backend type."""

    RAFT = "raft"
    SQLITE = "sqlite"


@dataclass
class EloRating:
    """Elo rating with metadata."""

    participant_id: str
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_update: float = 0.0
    confidence: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games_played


@dataclass
class MatchResult:
    """Result of a single match."""

    match_id: str
    participant_ids: list[str]
    winner_id: str | None
    game_length: int
    duration_sec: float
    board_type: str
    num_players: int
    timestamp: str
    elo_changes: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingFeedback:
    """Feedback signals for training parameter adaptation."""

    board_type: str
    num_players: int
    best_elo: float = 1500.0
    recent_elo_delta: float = 0.0
    elo_stagnating: bool = False
    elo_declining: bool = False
    best_win_rate: float = 0.5
    win_rate_vs_baseline: float = 0.5
    epochs_multiplier: float = 1.0
    lr_multiplier: float = 1.0
    exploration_boost: float = 0.0
    recommended_curriculum_stage: int = 0


@dataclass
class LeaderboardEntry:
    """Entry in the Elo leaderboard."""

    rank: int
    participant_id: str
    name: str
    ai_type: str
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    last_active: str


__all__ = [
    "EloBackendType",
    "EloRating",
    "LeaderboardEntry",
    "MatchResult",
    "TrainingFeedback",
]
