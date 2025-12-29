"""Facade for tournament scripts migrating from EloDatabase to EloService.

This module provides backward-compatible API wrappers for scripts that still use
the deprecated EloDatabase interface. New code should use EloService directly.

Usage:
    from app.tournament.elo_facade import EloServiceFacade

    # Get facade instance (wraps EloService singleton)
    facade = EloServiceFacade()

    # Use EloDatabase-style API
    match_id, new_ratings = facade.record_match_and_update(
        participant_ids=["model_a", "model_b"],
        rankings=[0, 1],  # model_a won
        board_type="square8",
        num_players=2,
        tournament_id="my_tournament",
    )

    # Get leaderboard
    leaders = facade.get_leaderboard(board_type="square8", num_players=2)

Migration from EloDatabase:
    # Old way
    from app.tournament import get_elo_database
    db = get_elo_database()
    db.record_match_and_update(...)

    # New way (via facade)
    from app.tournament.elo_facade import EloServiceFacade
    facade = EloServiceFacade()
    facade.record_match_and_update(...)

    # Best way (direct EloService)
    from app.training.elo_service import get_elo_service
    elo = get_elo_service()
    elo.record_match(...)

December 2025: Created as part of Elo unification initiative.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Import EloService
try:
    from app.training.elo_service import EloService, get_elo_service, EloRating
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None  # type: ignore
    EloService = None  # type: ignore
    EloRating = None  # type: ignore


@dataclass
class LegacyEloRating:
    """EloDatabase-compatible rating dataclass."""

    participant_id: str
    board_type: str
    num_players: int
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_deviation: float = 350.0
    last_update: float | None = None

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "rating": round(self.rating, 1),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 3),
            "rating_deviation": round(self.rating_deviation, 1),
        }


class EloServiceFacade:
    """Facade wrapping EloService with EloDatabase-compatible API.

    This allows scripts using the old EloDatabase interface to work with
    the new unified EloService without major code changes.
    """

    def __init__(self) -> None:
        """Initialize facade with EloService singleton."""
        if not HAS_ELO_SERVICE:
            raise ImportError(
                "EloService not available. Install app.training.elo_service."
            )
        self._service = get_elo_service()

    def record_match_and_update(
        self,
        participant_ids: list[str],
        rankings: list[int],
        board_type: str,
        num_players: int,
        tournament_id: str,
        game_length: int = 0,
        duration_sec: float = 0.0,
        worker: str | None = None,
        metadata: dict | None = None,
        k_factor: float = 32.0,
        game_id: str | None = None,
    ) -> tuple[int, dict[str, float]]:
        """Record a match using EloDatabase-style API.

        Converts ranking-based format to EloService's participant_a/b/winner format.

        Args:
            participant_ids: List of participant IDs in the match
            rankings: Final positions (0=1st, 1=2nd, etc.) for each participant
            board_type: Board type (square8, hex8, etc.)
            num_players: Number of players (2, 3, or 4)
            tournament_id: Tournament identifier
            game_length: Number of moves in the game
            duration_sec: Duration of the game in seconds
            worker: Worker that ran the game (optional)
            metadata: Additional match metadata (optional)
            k_factor: K-factor for Elo calculation (unused - EloService uses its own)
            game_id: Optional game UUID for deduplication

        Returns:
            Tuple of (match_id, dict of participant_id -> new_rating)
        """
        if len(participant_ids) < 2:
            raise ValueError("At least 2 participants required")

        # Generate game_id if not provided
        if game_id is None:
            game_id = str(uuid.uuid4())

        # Build metadata with worker info
        full_metadata = metadata.copy() if metadata else {}
        if worker:
            full_metadata["worker"] = worker
        full_metadata["game_id"] = game_id

        new_ratings: dict[str, float] = {}

        # For 2-player games, use direct record_match
        if len(participant_ids) == 2:
            # Determine winner based on rankings (lower rank = better)
            if rankings[0] < rankings[1]:
                winner = participant_ids[0]
            elif rankings[1] < rankings[0]:
                winner = participant_ids[1]
            else:
                winner = None  # Draw

            result = self._service.record_match(
                participant_a=participant_ids[0],
                participant_b=participant_ids[1],
                winner=winner,
                board_type=board_type,
                num_players=num_players,
                game_length=game_length,
                duration_sec=duration_sec,
                tournament_id=tournament_id,
                metadata=full_metadata,
            )

            # Get new ratings
            for pid in participant_ids:
                rating = self._service.get_rating(pid, board_type, num_players)
                new_ratings[pid] = rating.rating

            return result.match_id if hasattr(result, "match_id") else 0, new_ratings

        # For multiplayer games (3-4 players), record pairwise matches
        # Sort by ranking to get ordered participants
        sorted_participants = sorted(
            zip(participant_ids, rankings, strict=False),
            key=lambda x: x[1]
        )

        # Record pairwise matches for multiplayer
        match_id = 0
        for i, (pid_i, rank_i) in enumerate(sorted_participants):
            for j, (pid_j, rank_j) in enumerate(sorted_participants):
                if i >= j:
                    continue  # Skip self and already-processed pairs

                # Determine winner of this pair
                if rank_i < rank_j:
                    winner = pid_i
                elif rank_j < rank_i:
                    winner = pid_j
                else:
                    winner = None  # Draw (same rank)

                try:
                    result = self._service.record_match(
                        participant_a=pid_i,
                        participant_b=pid_j,
                        winner=winner,
                        board_type=board_type,
                        num_players=num_players,
                        game_length=game_length,
                        duration_sec=duration_sec,
                        tournament_id=tournament_id,
                        metadata=full_metadata,
                    )
                    if hasattr(result, "match_id"):
                        match_id = result.match_id
                except Exception as e:
                    logger.warning(f"Failed to record pairwise match {pid_i} vs {pid_j}: {e}")

        # Get final ratings
        for pid in participant_ids:
            rating = self._service.get_rating(pid, board_type, num_players)
            new_ratings[pid] = rating.rating

        return match_id, new_ratings

    def get_rating(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
    ) -> LegacyEloRating:
        """Get rating for a participant.

        Args:
            participant_id: The participant ID
            board_type: Board type
            num_players: Number of players

        Returns:
            LegacyEloRating with EloDatabase-compatible fields
        """
        rating = self._service.get_rating(participant_id, board_type, num_players)
        return LegacyEloRating(
            participant_id=participant_id,
            board_type=board_type,
            num_players=num_players,
            rating=rating.rating,
            games_played=rating.games_played,
            wins=rating.wins,
            losses=rating.losses,
            draws=rating.draws,
            rating_deviation=rating.confidence if hasattr(rating, "confidence") else 350.0,
        )

    def get_leaderboard(
        self,
        board_type: str,
        num_players: int,
        limit: int = 50,
    ) -> list[LegacyEloRating]:
        """Get leaderboard for a configuration.

        Args:
            board_type: Board type
            num_players: Number of players
            limit: Maximum entries to return

        Returns:
            List of LegacyEloRating sorted by rating descending
        """
        entries = self._service.get_leaderboard(
            board_type=board_type,
            num_players=num_players,
            limit=limit,
        )
        return [
            LegacyEloRating(
                participant_id=e.participant_id,
                board_type=board_type,
                num_players=num_players,
                rating=e.rating,
                games_played=e.games_played,
                wins=e.wins,
                losses=e.losses,
                draws=e.draws,
            )
            for e in entries
        ]

    def ensure_participant(
        self,
        participant_id: str,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> None:
        """Ensure a participant exists in the database.

        Args:
            participant_id: The participant ID
            board_type: Optional board type (defaults to square8)
            num_players: Optional player count (defaults to 2)
        """
        # EloService auto-registers on first use, but we can explicitly register
        board = board_type or "square8"
        players = num_players or 2
        self._service.register_model(participant_id, board, players)

    def get_participant_elo(
        self,
        participant_id: str,
        board_type: str,
        num_players: int,
    ) -> float:
        """Get Elo rating for a participant.

        Args:
            participant_id: The participant ID
            board_type: Board type
            num_players: Number of players

        Returns:
            Current Elo rating
        """
        rating = self._service.get_rating(participant_id, board_type, num_players)
        return rating.rating

    def get_match_count(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> int:
        """Get total match count.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count

        Returns:
            Number of matches recorded
        """
        # EloService provides this via get_stats or direct query
        stats = self._service.get_stats(board_type=board_type, num_players=num_players)
        return stats.get("total_matches", 0) if isinstance(stats, dict) else 0

    def get_participant_count(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> int:
        """Get total participant count.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count

        Returns:
            Number of unique participants
        """
        stats = self._service.get_stats(board_type=board_type, num_players=num_players)
        return stats.get("total_participants", 0) if isinstance(stats, dict) else 0


# Singleton instance
_facade_instance: EloServiceFacade | None = None


def get_elo_facade() -> EloServiceFacade:
    """Get singleton EloServiceFacade instance.

    Returns:
        EloServiceFacade instance wrapping EloService
    """
    global _facade_instance
    if _facade_instance is None:
        _facade_instance = EloServiceFacade()
    return _facade_instance


# Backward-compatible alias
EloFacade = EloServiceFacade


__all__ = [
    "EloServiceFacade",
    "EloFacade",
    "LegacyEloRating",
    "get_elo_facade",
]
