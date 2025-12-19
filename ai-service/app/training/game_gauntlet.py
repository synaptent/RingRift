"""Unified Game Gauntlet Module for RingRift AI Evaluation.

Consolidates game-playing logic that was duplicated across:
- scripts/select_best_checkpoint_by_elo.py
- app/training/tier_eval_runner.py
- app/training/background_eval.py

This module provides:
- play_single_game(): Play one game between two AIs
- run_baseline_gauntlet(): Evaluate a model against baseline opponents
- BaselineOpponent: Enum of standard baseline types

Usage:
    from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent

    results = run_baseline_gauntlet(
        model_path="models/my_model.pth",
        board_type=BoardType.SQUARE8,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        games_per_opponent=20,
    )
    print(f"Win rate vs random: {results['random']['win_rate']:.1%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies and heavy imports at module load
_torch_loaded = False
_game_modules_loaded = False


def _ensure_game_modules():
    """Lazy load game-related modules."""
    global _game_modules_loaded
    if _game_modules_loaded:
        return

    global BoardType, AIType, AIConfig, GameStatus
    global HeuristicAI, RandomAI, PolicyOnlyAI
    global create_initial_state, DefaultRulesEngine

    from app.models import BoardType, AIType, AIConfig, GameStatus
    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.random_ai import RandomAI
    from app.ai.policy_only_ai import PolicyOnlyAI
    from app.training.generate_data import create_initial_state
    from app.rules.default_engine import DefaultRulesEngine

    _game_modules_loaded = True


class BaselineOpponent(Enum):
    """Standard baseline opponents for evaluation."""
    RANDOM = "random"
    HEURISTIC = "heuristic"


# Import baseline Elo estimates from centralized config
try:
    from app.config.thresholds import (
        BASELINE_ELO_RANDOM,
        BASELINE_ELO_HEURISTIC,
        MIN_WIN_RATE_VS_RANDOM,
        MIN_WIN_RATE_VS_HEURISTIC,
    )
except ImportError:
    BASELINE_ELO_RANDOM = 400
    BASELINE_ELO_HEURISTIC = 1200
    MIN_WIN_RATE_VS_RANDOM = 0.85
    MIN_WIN_RATE_VS_HEURISTIC = 0.60

BASELINE_ELOS = {
    BaselineOpponent.RANDOM: BASELINE_ELO_RANDOM,
    BaselineOpponent.HEURISTIC: BASELINE_ELO_HEURISTIC,
}

MIN_WIN_RATES = {
    BaselineOpponent.RANDOM: MIN_WIN_RATE_VS_RANDOM,
    BaselineOpponent.HEURISTIC: MIN_WIN_RATE_VS_HEURISTIC,
}


@dataclass
class GameResult:
    """Result of a single game."""
    winner: Optional[int]  # Player number who won, or None for draw
    move_count: int
    victory_reason: str
    candidate_player: int  # Which player was the candidate
    candidate_won: bool


@dataclass
class GauntletResult:
    """Aggregated results from a gauntlet evaluation."""
    total_games: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    win_rate: float = 0.0

    # Per-opponent results
    opponent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Baseline gating
    passes_baseline_gating: bool = True
    failed_baselines: List[str] = field(default_factory=list)

    # Elo estimate
    estimated_elo: float = 1500.0


def create_baseline_ai(
    baseline: BaselineOpponent,
    player: int,
    board_type: Any,  # BoardType
    difficulty: Optional[int] = None,
) -> Any:
    """Create an AI instance for a baseline opponent.

    Args:
        baseline: Which baseline to create
        player: Player number (1 or 2)
        board_type: Board type enum
        difficulty: Optional difficulty override

    Returns:
        AI instance ready to play
    """
    _ensure_game_modules()

    if baseline == BaselineOpponent.RANDOM:
        config = AIConfig(
            ai_type=AIType.RANDOM,
            board_type=board_type,
            difficulty=difficulty or 1,
        )
        return RandomAI(player, config)

    elif baseline == BaselineOpponent.HEURISTIC:
        config = AIConfig(
            ai_type=AIType.HEURISTIC,
            board_type=board_type,
            difficulty=difficulty or 5,
        )
        return HeuristicAI(player, config)

    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def create_neural_ai(
    player: int,
    board_type: Any,  # BoardType
    model_path: Optional[Union[str, Path]] = None,
    model_getter: Optional[Callable[[], Any]] = None,
    temperature: float = 0.5,
) -> Any:
    """Create a neural network AI instance.

    Args:
        player: Player number
        board_type: Board type enum
        model_path: Path to model checkpoint (for file-based loading)
        model_getter: Callable that returns model weights (for in-memory loading)
        temperature: Policy temperature for move selection

    Returns:
        PolicyOnlyAI instance
    """
    _ensure_game_modules()

    if model_path is not None:
        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_model_id=str(model_path),
            policy_temperature=temperature,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    elif model_getter is not None:
        # In-memory model loading for BackgroundEvaluator (zero disk I/O)
        model_info = model_getter()

        # Extract state_dict from model_info
        if isinstance(model_info, dict):
            if 'state_dict' in model_info:
                state_dict = model_info['state_dict']
            else:
                # Assume the dict is the state_dict itself
                state_dict = model_info
        elif hasattr(model_info, 'state_dict'):
            # It's an nn.Module - extract state_dict
            state_dict = model_info.state_dict()
        else:
            raise ValueError(
                f"model_getter must return a dict with 'state_dict', a state_dict, "
                f"or an nn.Module. Got: {type(model_info)}"
            )

        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_state_dict=state_dict,
            policy_temperature=temperature,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    else:
        raise ValueError("Must provide either model_path or model_getter")


def play_single_game(
    candidate_ai: Any,
    opponent_ai: Any,
    board_type: Any,  # BoardType
    num_players: int = 2,
    candidate_player: int = 1,
    max_moves: int = 500,
    seed: Optional[int] = None,
) -> GameResult:
    """Play a single game between candidate and opponent.

    Args:
        candidate_ai: The AI being evaluated
        opponent_ai: The baseline/opponent AI
        board_type: Board type for the game
        num_players: Number of players
        candidate_player: Which player number is the candidate
        max_moves: Maximum moves before draw
        seed: Optional random seed

    Returns:
        GameResult with game outcome
    """
    _ensure_game_modules()

    engine = DefaultRulesEngine()
    state = create_initial_state(board_type, num_players)

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player

        if current_player == candidate_player:
            move = candidate_ai.select_move(state)
        else:
            move = opponent_ai.select_move(state)

        if move:
            state = engine.apply_move(state, move)
        move_count += 1

    # Determine outcome
    victory_reason = getattr(state, 'victory_reason', 'unknown')
    if hasattr(victory_reason, 'value'):
        victory_reason = victory_reason.value

    winner = state.winner
    candidate_won = winner == candidate_player if winner is not None else False

    return GameResult(
        winner=winner,
        move_count=move_count,
        victory_reason=str(victory_reason) if victory_reason else "max_moves",
        candidate_player=candidate_player,
        candidate_won=candidate_won,
    )


def run_baseline_gauntlet(
    model_path: Optional[Union[str, Path]] = None,
    board_type: Any = None,  # BoardType
    opponents: Optional[List[BaselineOpponent]] = None,
    games_per_opponent: int = 20,
    num_players: int = 2,
    check_baseline_gating: bool = True,
    verbose: bool = False,
    model_getter: Optional[Callable[[], Any]] = None,
) -> GauntletResult:
    """Run a gauntlet evaluation against baseline opponents.

    Args:
        model_path: Path to the model checkpoint (file-based loading)
        board_type: Board type for games
        opponents: List of baselines to test against (default: RANDOM, HEURISTIC)
        games_per_opponent: Number of games per opponent
        num_players: Number of players in each game
        check_baseline_gating: Whether to check minimum win rate thresholds
        verbose: Whether to log per-game results
        model_getter: Callable returning model weights (in-memory loading, zero disk I/O)

    Returns:
        GauntletResult with aggregated statistics
    """
    if model_path is None and model_getter is None:
        raise ValueError("Must provide either model_path or model_getter")
    _ensure_game_modules()

    if opponents is None:
        opponents = [BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC]

    result = GauntletResult()

    for baseline in opponents:
        baseline_name = baseline.value
        opponent_stats = {"wins": 0, "games": 0, "win_rate": 0.0}

        for game_num in range(games_per_opponent):
            # Alternate who plays first
            candidate_player = 1 if game_num % 2 == 0 else 2
            opponent_player = 2 if game_num % 2 == 0 else 1

            try:
                candidate_ai = create_neural_ai(
                    candidate_player, board_type,
                    model_path=model_path,
                    model_getter=model_getter,
                )
                opponent_ai = create_baseline_ai(
                    baseline, opponent_player, board_type
                )

                game_result = play_single_game(
                    candidate_ai=candidate_ai,
                    opponent_ai=opponent_ai,
                    board_type=board_type,
                    num_players=num_players,
                    candidate_player=candidate_player,
                )

                result.total_games += 1
                opponent_stats["games"] += 1

                if game_result.candidate_won:
                    result.total_wins += 1
                    opponent_stats["wins"] += 1
                elif game_result.winner is not None:
                    result.total_losses += 1
                else:
                    result.total_draws += 1

                if verbose:
                    outcome = "WIN" if game_result.candidate_won else "LOSS"
                    logger.info(
                        f"[gauntlet] Game {game_num+1}/{games_per_opponent} vs {baseline_name}: "
                        f"{outcome} ({game_result.victory_reason}, {game_result.move_count} moves)"
                    )

            except Exception as e:
                logger.error(f"Error in game {game_num} vs {baseline_name}: {e}")
                continue

        # Calculate win rate for this opponent
        if opponent_stats["games"] > 0:
            opponent_stats["win_rate"] = opponent_stats["wins"] / opponent_stats["games"]

        result.opponent_results[baseline_name] = opponent_stats

        # Check baseline gating
        if check_baseline_gating:
            min_required = MIN_WIN_RATES.get(baseline, 0.0)
            if opponent_stats["win_rate"] < min_required:
                result.passes_baseline_gating = False
                result.failed_baselines.append(baseline_name)
                logger.warning(
                    f"[gauntlet] Failed baseline gating vs {baseline_name}: "
                    f"{opponent_stats['win_rate']:.1%} < {min_required:.0%} required"
                )

    # Calculate overall win rate
    if result.total_games > 0:
        result.win_rate = result.total_wins / result.total_games

    # Estimate Elo from win rates
    result.estimated_elo = _estimate_elo_from_results(result.opponent_results)

    return result


def _estimate_elo_from_results(
    opponent_results: Dict[str, Dict[str, Any]]
) -> float:
    """Estimate Elo rating from gauntlet results.

    Uses weighted average of Elo estimates from each opponent.
    """
    import math

    total_elo = 0.0
    total_weight = 0.0

    for baseline_name, stats in opponent_results.items():
        baseline = BaselineOpponent(baseline_name)
        opponent_elo = BASELINE_ELOS.get(baseline, 1000)
        win_rate = stats.get("win_rate", 0.5)
        games = stats.get("games", 0)

        if games == 0:
            continue

        # Elo formula: E = 1 / (1 + 10^((Rb-Ra)/400))
        # Solving for Ra: Ra = Rb - 400 * log10(1/E - 1)
        if win_rate <= 0:
            estimated = opponent_elo - 400
        elif win_rate >= 1:
            estimated = opponent_elo + 400
        else:
            estimated = opponent_elo - 400 * math.log10(1/win_rate - 1)

        total_elo += estimated * games
        total_weight += games

    if total_weight > 0:
        return total_elo / total_weight
    return 1500.0


# Convenience function for quick evaluation
def quick_evaluate(
    model_path: Union[str, Path],
    games: int = 10,
) -> Dict[str, float]:
    """Quick evaluation against baselines.

    Args:
        model_path: Path to model checkpoint
        games: Games per opponent

    Returns:
        Dict with win rates per opponent
    """
    _ensure_game_modules()

    result = run_baseline_gauntlet(
        model_path=model_path,
        board_type=BoardType.SQUARE8,
        games_per_opponent=games,
        verbose=True,
    )

    return {
        name: stats["win_rate"]
        for name, stats in result.opponent_results.items()
    }
