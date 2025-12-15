"""Tournament system for AI agent evaluation with Elo ratings.

This module provides a unified tournament framework for running AI
agent competitions with Elo rating tracking.

Quick Start:
    from app.tournament import run_quick_tournament

    # Run a quick tournament
    results = run_quick_tournament(
        agent_ids=["random", "heuristic", "mcts_100"],
        board_type="square8",
        num_players=2,
        games_per_pairing=10,
    )
    print(f"Winner: {results.final_ratings}")

For more control, use the lower-level APIs:
    from app.tournament import TournamentRunner, AIAgentRegistry, RoundRobinScheduler

    registry = AIAgentRegistry()
    scheduler = RoundRobinScheduler()
    runner = TournamentRunner(registry, scheduler)
    results = runner.run_tournament(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .agents import AIAgent, AIAgentRegistry, AgentType
from .elo import EloRating, EloCalculator
from .scheduler import Match, MatchStatus, TournamentScheduler, RoundRobinScheduler, SwissScheduler
from .runner import MatchResult, TournamentRunner, TournamentResults
from .unified_elo_db import (
    EloDatabase,
    UnifiedEloRating,
    MatchRecord,
    get_elo_database,
    reset_elo_database,
)

# Canonical Elo service (preferred for new code)
# unified_elo_db is maintained for backward compatibility
from app.training.elo_service import (
    EloService,
    get_elo_service,
    ELO_DB_PATH,
    get_head_to_head,
    get_database_stats,
    get_match_history,
    get_rating_history,
)
from .orchestrator import (
    TournamentOrchestrator,
    TournamentSummary,
    EvaluationResult,
    run_quick_evaluation,
    run_elo_calibration,
)

logger = logging.getLogger(__name__)

__all__ = [
    # High-level API
    "run_quick_tournament",
    "create_tournament_runner",
    # Core classes
    "AIAgent",
    "AIAgentRegistry",
    "AgentType",
    "EloRating",
    "EloCalculator",
    "Match",
    "MatchStatus",
    "TournamentScheduler",
    "RoundRobinScheduler",
    "SwissScheduler",
    "MatchResult",
    "TournamentRunner",
    "TournamentResults",
    # Unified Elo database (legacy - use EloService for new code)
    "EloDatabase",
    "UnifiedEloRating",
    "MatchRecord",
    "get_elo_database",
    "reset_elo_database",
    # Canonical Elo service (preferred)
    "EloService",
    "get_elo_service",
    "ELO_DB_PATH",
    "get_head_to_head",
    "get_database_stats",
    "get_match_history",
    "get_rating_history",
    # Tournament orchestrator
    "TournamentOrchestrator",
    "TournamentSummary",
    "EvaluationResult",
    "run_quick_evaluation",
    "run_elo_calibration",
]


# Singleton registry for convenience
_default_registry: Optional[AIAgentRegistry] = None


def get_default_registry() -> AIAgentRegistry:
    """Get the default agent registry (singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = AIAgentRegistry()
    return _default_registry


def create_tournament_runner(
    scheduler_type: str = "round_robin",
    max_workers: int = 4,
    persist_to_unified_elo: bool = True,
    registry: Optional[AIAgentRegistry] = None,
) -> TournamentRunner:
    """Create a configured tournament runner.

    Args:
        scheduler_type: Type of scheduler ("round_robin" or "swiss")
        max_workers: Maximum parallel workers for local execution
        persist_to_unified_elo: Whether to persist results to unified Elo database
        registry: Agent registry (uses default if None)

    Returns:
        Configured TournamentRunner instance
    """
    # Get config for defaults
    try:
        from app.config.unified_config import get_config
        config = get_config()
        max_workers = max_workers or config.tournament.default_games_per_matchup
    except ImportError:
        pass

    # Create scheduler
    if scheduler_type == "swiss":
        scheduler = SwissScheduler()
    else:
        scheduler = RoundRobinScheduler()

    # Use provided or default registry
    if registry is None:
        registry = get_default_registry()

    return TournamentRunner(
        agent_registry=registry,
        scheduler=scheduler,
        max_workers=max_workers,
        persist_to_unified_elo=persist_to_unified_elo,
    )


def run_quick_tournament(
    agent_ids: List[str],
    board_type: Union[str, "BoardType"] = "square8",
    num_players: int = 2,
    games_per_pairing: int = 10,
    scheduler_type: str = "round_robin",
    max_workers: int = 4,
    persist_to_unified_elo: bool = True,
    progress_callback: Optional[Any] = None,
) -> TournamentResults:
    """Run a quick tournament with minimal setup.

    This is the easiest way to run a tournament. For more control,
    use create_tournament_runner() or TournamentRunner directly.

    Args:
        agent_ids: List of agent IDs to compete. Use built-in IDs like
                   "random", "baseline_v1", "aggressive_v1", "defensive_v1"
                   or register custom agents first.
        board_type: Board type (e.g., "square8", "square19", "hexagonal")
        num_players: Number of players per match (2, 3, or 4)
        games_per_pairing: Number of games per agent pairing
        scheduler_type: "round_robin" or "swiss"
        max_workers: Maximum parallel workers
        persist_to_unified_elo: Persist results to unified Elo database
        progress_callback: Optional callback(completed, total) for progress

    Returns:
        TournamentResults with final ratings and statistics

    Example:
        results = run_quick_tournament(
            agent_ids=["random", "baseline_v1", "aggressive_v1"],
            board_type="square8",
            num_players=2,
            games_per_pairing=20,
        )

        for agent_id, rating in results.final_ratings.items():
            stats = results.agent_stats.get(agent_id, {})
            print(f"{agent_id}: {rating:.0f} Elo ({stats.get('win_rate', 0)*100:.0f}% win rate)")
    """
    from app.models import BoardType as BT

    # Convert board type if string
    if isinstance(board_type, str):
        board_type_enum = BT(board_type)
    else:
        board_type_enum = board_type

    # Create runner
    runner = create_tournament_runner(
        scheduler_type=scheduler_type,
        max_workers=max_workers,
        persist_to_unified_elo=persist_to_unified_elo,
    )

    # Run tournament
    return runner.run_tournament(
        agent_ids=agent_ids,
        board_type=board_type_enum,
        num_players=num_players,
        games_per_pairing=games_per_pairing,
        progress_callback=progress_callback,
    )
