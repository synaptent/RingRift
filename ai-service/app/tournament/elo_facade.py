"""Facade for tournament scripts migrating from EloDatabase to EloService.

This module provides backward-compatible API wrappers for scripts that still use
the deprecated EloDatabase interface. New code should use EloService directly.

Additionally provides architecture performance ranking for the unified NN/NNUE
multi-harness evaluation system (December 2025).

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

    # NEW: Architecture performance tracking
    from app.tournament.elo_facade import (
        get_architecture_rankings,
        ArchitecturePerformance,
    )

    # Get performance by architecture version
    rankings = get_architecture_rankings(board_type="square8", num_players=2)
    for perf in rankings:
        print(f"{perf.model_type}: avg={perf.avg_elo:.0f}, best={perf.best_elo:.0f}")

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
December 2025: Added architecture performance ranking for unified NN/NNUE tracking.
January 2026: Added harness_type extraction from composite participant IDs.
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

# Import harness type extraction (January 2026)
try:
    from app.training.composite_participant import extract_harness_type
except ImportError:
    def extract_harness_type(participant_id: str) -> str | None:  # type: ignore
        """Fallback if composite_participant not available."""
        return None


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
        harness_type: str | None = None,
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
            harness_type: AI harness type (e.g., "gumbel_mcts", "brs") for per-harness Elo (Jan 2026)

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

        # January 2026: Try to extract harness_type from composite participant IDs if not provided
        effective_harness_type = harness_type
        if effective_harness_type is None and participant_ids:
            effective_harness_type = extract_harness_type(participant_ids[0])

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
                harness_type=effective_harness_type,
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
                        harness_type=effective_harness_type,
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


def reset_elo_facade() -> None:
    """Reset the singleton EloServiceFacade (for testing).

    Dec 29, 2025: Added to fix tournament test class leak issue.
    Ensures test isolation by clearing the cached facade instance.
    Also resets the underlying EloService singleton to fully clear state.

    Usage in tests:
        @pytest.fixture(autouse=True)
        def cleanup_elo():
            yield
            reset_elo_facade()
    """
    global _facade_instance

    # Reset the facade instance
    _facade_instance = None

    # Also reset the underlying EloService singleton for full cleanup
    try:
        from app.training.elo_service import reset_elo_service

        reset_elo_service()
    except ImportError:
        pass  # EloService not available, nothing to reset


# Backward-compatible alias
EloFacade = EloServiceFacade


# ==============================================================================
# Architecture Performance Tracking (December 2025)
# ==============================================================================


@dataclass
class ArchitecturePerformance:
    """Performance metrics for a model architecture version.

    Used for:
    - Comparing efficiency across NN architectures (v2, v3, v4, v5, v6)
    - Tracking NNUE vs full NN performance
    - Allocating training compute to better-performing architectures
    """

    model_type: str       # e.g., "nn_v5", "nnue_v1", "nnue_mp_v1"
    avg_elo: float        # Average Elo across all harnesses
    best_elo: float       # Best Elo for this architecture
    worst_elo: float      # Worst Elo (useful for consistency check)
    games_evaluated: int  # Total games played by models of this type
    configs_trained: int  # Number of board configurations with models
    harnesses_tested: int  # Number of harness variants evaluated
    best_harness: str | None  # Which harness gave best results
    elo_variance: float   # Variance in Elo across harnesses (consistency)

    @property
    def elo_range(self) -> float:
        """Elo range (best - worst) as measure of harness sensitivity."""
        return self.best_elo - self.worst_elo

    @property
    def is_reliable(self) -> bool:
        """Check if we have enough data for reliable comparison."""
        return self.games_evaluated >= 50 and self.harnesses_tested >= 2


def get_architecture_rankings(
    board_type: str | None = None,
    num_players: int | None = None,
    min_games: int = 10,
) -> list[ArchitecturePerformance]:
    """Get performance rankings by model architecture.

    Aggregates Elo data across all model+harness combinations, grouped by
    architecture version. Useful for deciding which architectures to train.

    Args:
        board_type: Optional filter by board type
        num_players: Optional filter by player count
        min_games: Minimum games required for inclusion

    Returns:
        List of ArchitecturePerformance sorted by avg_elo descending
    """
    try:
        from app.training.composite_participant import (
            ModelType,
            extract_model_type,
            is_composite_id,
        )
    except ImportError:
        logger.warning("composite_participant not available")
        return []

    if not HAS_ELO_SERVICE:
        return []

    service = get_elo_service()

    # Get all participants
    try:
        leaderboard = service.get_leaderboard(
            board_type=board_type,
            num_players=num_players,
            limit=1000,
        )
    except Exception as e:
        logger.warning(f"Failed to get leaderboard: {e}")
        return []

    # Group by model type
    type_data: dict[str, list[tuple[float, int, str]]] = {}  # model_type -> [(elo, games, harness), ...]

    for entry in leaderboard:
        pid = entry.participant_id
        elo = entry.rating
        games = entry.games_played

        if games < min_games:
            continue

        # Extract model type
        model_type = extract_model_type(pid)
        if model_type is None:
            continue

        type_key = model_type.value

        # Extract harness from composite ID
        harness = "unknown"
        if is_composite_id(pid):
            parts = pid.split(":")
            if len(parts) >= 2:
                harness = parts[1]

        if type_key not in type_data:
            type_data[type_key] = []
        type_data[type_key].append((elo, games, harness))

    # Compute aggregate stats per architecture
    results: list[ArchitecturePerformance] = []

    for model_type, entries in type_data.items():
        if not entries:
            continue

        elos = [e[0] for e in entries]
        games = [e[1] for e in entries]
        harnesses = set(e[2] for e in entries)

        avg_elo = sum(elos) / len(elos)
        best_elo = max(elos)
        worst_elo = min(elos)
        total_games = sum(games)

        # Find best harness
        best_idx = elos.index(best_elo)
        best_harness = entries[best_idx][2] if best_idx < len(entries) else None

        # Compute variance
        variance = sum((e - avg_elo) ** 2 for e in elos) / len(elos) if len(elos) > 1 else 0.0

        results.append(ArchitecturePerformance(
            model_type=model_type,
            avg_elo=avg_elo,
            best_elo=best_elo,
            worst_elo=worst_elo,
            games_evaluated=total_games,
            configs_trained=len(entries),  # Approximation
            harnesses_tested=len(harnesses),
            best_harness=best_harness,
            elo_variance=variance,
        ))

    # Sort by average Elo descending
    results.sort(key=lambda x: x.avg_elo, reverse=True)
    return results


def get_harness_rankings(
    model_id: str,
    board_type: str,
    num_players: int,
) -> list[tuple[str, float, int]]:
    """Get performance by harness for a specific model.

    Args:
        model_id: Model identifier (e.g., "ringrift_v5_sq8_2p")
        board_type: Board type
        num_players: Number of players

    Returns:
        List of (harness, elo, games) tuples sorted by elo descending
    """
    try:
        from app.training.composite_participant import (
            get_all_harness_variants,
            parse_composite_participant_id,
        )
    except ImportError:
        logger.warning("composite_participant not available")
        return []

    if not HAS_ELO_SERVICE:
        return []

    service = get_elo_service()

    # Get all harness variants for this model
    variants = get_all_harness_variants(model_id)

    results: list[tuple[str, float, int]] = []
    for pid in variants:
        try:
            _, harness, _ = parse_composite_participant_id(pid)
            rating = service.get_rating(pid, board_type, num_players)
            results.append((harness, rating.rating, rating.games_played))
        except (ValueError, AttributeError):
            continue

    # Sort by Elo descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def compute_training_allocation(
    board_type: str | None = None,
    num_players: int | None = None,
    temperature: float = 0.5,
) -> dict[str, float]:
    """Compute training compute allocation based on architecture performance.

    Uses softmax with temperature to convert Elo rankings into allocation
    weights. Higher-performing architectures get more training compute.

    Args:
        board_type: Optional filter by board type
        num_players: Optional filter by player count
        temperature: Softmax temperature (lower = more winner-take-all)

    Returns:
        Dict mapping model_type to allocation weight (sums to 1.0)

    Example:
        >>> compute_training_allocation("square8", 2, temperature=0.5)
        {"nn_v5": 0.4, "nn_v4": 0.3, "nn_v3": 0.2, "nn_v2": 0.1}
    """
    import math

    rankings = get_architecture_rankings(board_type, num_players)
    if not rankings:
        return {}

    # Filter to reliable entries only
    reliable = [r for r in rankings if r.is_reliable]
    if not reliable:
        reliable = rankings[:5]  # Fall back to top 5

    # Compute softmax weights
    elos = [r.avg_elo for r in reliable]
    max_elo = max(elos)

    # Normalize Elo (subtract max for numerical stability)
    exp_elos = [math.exp((elo - max_elo) / (temperature * 100)) for elo in elos]
    total = sum(exp_elos)

    allocations = {}
    for perf, exp_elo in zip(reliable, exp_elos, strict=False):
        allocations[perf.model_type] = exp_elo / total

    return allocations


def register_nnue_participant(
    nnue_version: str,
    board_config: str,
    ai_type: str = "minimax",
    model_path: str | None = None,
) -> str:
    """Register an NNUE model as a participant.

    Convenience function for registering NNUE models with proper
    composite participant ID format.

    Args:
        nnue_version: NNUE version (e.g., "v1")
        board_config: Board configuration (e.g., "sq8_2p")
        ai_type: Algorithm type (defaults to "minimax")
        model_path: Optional path to model file

    Returns:
        Composite participant ID that was registered
    """
    try:
        from app.training.composite_participant import make_nnue_participant_id
    except ImportError:
        raise ImportError("composite_participant module required")

    pid = make_nnue_participant_id(nnue_version, board_config, ai_type)

    # Extract board_type and num_players from config
    parts = board_config.split("_")
    board_type = parts[0] if parts else "square8"
    num_players = 2
    if len(parts) > 1 and parts[-1].endswith("p"):
        try:
            num_players = int(parts[-1][:-1])
        except ValueError:
            pass

    if HAS_ELO_SERVICE:
        service = get_elo_service()
        service.register_model(
            model_id=pid,
            board_type=board_type,
            num_players=num_players,
            model_path=model_path,
            validate_file=model_path is not None,
        )

    return pid


def register_nnue_mp_participant(
    nnue_version: str,
    board_config: str,
    ai_type: str = "maxn",
    model_path: str | None = None,
) -> str:
    """Register a multi-player NNUE model as a participant.

    Args:
        nnue_version: NNUE version (e.g., "v1")
        board_config: Board configuration (e.g., "hex8_4p")
        ai_type: Algorithm type (defaults to "maxn")
        model_path: Optional path to model file

    Returns:
        Composite participant ID that was registered
    """
    try:
        from app.training.composite_participant import make_nnue_mp_participant_id
    except ImportError:
        raise ImportError("composite_participant module required")

    pid = make_nnue_mp_participant_id(nnue_version, board_config, ai_type)

    # Extract board_type and num_players from config
    parts = board_config.split("_")
    board_type = parts[0] if parts else "hex8"
    num_players = 4
    if len(parts) > 1 and parts[-1].endswith("p"):
        try:
            num_players = int(parts[-1][:-1])
        except ValueError:
            pass

    if HAS_ELO_SERVICE:
        service = get_elo_service()
        service.register_model(
            model_id=pid,
            board_type=board_type,
            num_players=num_players,
            model_path=model_path,
            validate_file=model_path is not None,
        )

    return pid


__all__ = [
    # Legacy facade
    "EloServiceFacade",
    "EloFacade",
    "LegacyEloRating",
    "get_elo_facade",
    "reset_elo_facade",
    # Architecture performance (December 2025)
    "ArchitecturePerformance",
    "get_architecture_rankings",
    "get_harness_rankings",
    "compute_training_allocation",
    "register_nnue_participant",
    "register_nnue_mp_participant",
]
