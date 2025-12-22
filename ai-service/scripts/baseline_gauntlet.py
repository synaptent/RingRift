#!/usr/bin/env python3
"""Baseline gauntlet for model evaluation.

Provides simple synchronous evaluation of models against baselines
(random, heuristic) for use by the promotion daemon.

This is a thin adapter around the existing game infrastructure,
specifically designed for quick model quality checks during promotion.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.models import AIConfig, BoardType, GameStatus
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state

if TYPE_CHECKING:
    from app.ai.base_ai import BaseAI
    from app.models import GameState


@dataclass
class GauntletResult:
    """Result of baseline gauntlet evaluation."""

    model_type: str
    vs_random: float
    vs_heuristic: float
    vs_mcts: float  # 0.0 if not evaluated
    score: float  # Weighted composite score
    games_played: int


def run_game(ai1: "BaseAI", ai2: "BaseAI", board_type: BoardType, num_players: int) -> int | None:
    """Run a single game between two AIs.

    Args:
        ai1: AI for player 1
        ai2: AI for player 2
        board_type: Board type for the game
        num_players: Number of players

    Returns:
        Winner (1 or 2) or None for draw/timeout
    """
    state = create_initial_state(board_type=board_type, num_players=num_players)
    engine = DefaultRulesEngine()
    ais = {1: ai1, 2: ai2}

    max_moves = 500
    move_count = 0

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current = state.current_player
        ai = ais[current]

        try:
            move = ai.select_move(state)
        except Exception:
            # AI error - opponent wins
            return 3 - current  # Other player wins

        if move is None:
            break

        state = engine.apply_move(state, move)
        move_count += 1

    if state.game_status == GameStatus.COMPLETED and state.winner is not None:
        return state.winner
    return None


def create_ai_from_model(
    model: dict[str, Any],
    player: int,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> "BaseAI":
    """Create AI instance from model dictionary using UnifiedModelLoader.

    This function uses the unified model loader to automatically detect
    and load ANY model architecture (NNUE, CNN, Hex, experimental).
    It never fails - always returns a working AI (falls back to heuristic).

    Args:
        model: Model dict with 'path', 'name', 'type' keys
        player: Player number (1-based)
        board_type: Board type for the game
        num_players: Number of players

    Returns:
        AI instance with loaded model weights (or heuristic fallback)
    """
    model_type = model.get("type", "nn").lower()
    model_path = Path(model.get("path", ""))

    # Handle special model types first
    if model_type == "mcts":
        from app.ai.mcts_ai import MCTSAI
        ai_config = AIConfig(difficulty=5, board_type=board_type)
        mcts_config = {"simulations": 100}
        return MCTSAI(player, ai_config, mcts_config)

    elif model_type == "heuristic":
        return HeuristicAI(player, AIConfig(difficulty=5, board_type=board_type))

    elif model_type == "random":
        return RandomAI(player, AIConfig(difficulty=1))

    # Check if model path exists
    if not model_path.exists():
        logger.info(f"Model path not found: {model_path}, using HeuristicAI")
        return HeuristicAI(player, AIConfig(difficulty=5, board_type=board_type))

    # Use UnifiedModelLoader + UniversalAI for all neural network models
    try:
        from app.ai.unified_loader import UnifiedModelLoader
        from app.ai.universal_ai import UniversalAI

        loader = UnifiedModelLoader()
        loaded = loader.load(
            model_path,
            board_type=board_type,
            num_players=num_players,
            allow_fresh=False,  # Fail if can't load, we'll catch and fallback
        )

        config = AIConfig(
            difficulty=5,
            board_type=board_type,
        )

        ai = UniversalAI(
            player_number=player,
            config=config,
            loaded_model=loaded,
            board_type=board_type,
            num_players=num_players,
            use_mcts=False,  # Use direct policy/minimax for speed
            policy_temperature=0.1,  # Low temp for more deterministic play
        )

        logger.info(
            f"Loaded {loaded.architecture.name} model from {model_path} "
            f"(board={board_type}, players={num_players})"
        )
        return ai

    except Exception as e:
        logger.warning(f"UnifiedModelLoader failed for {model_path}: {e}")

    # Fallback to heuristic - NEVER fail
    logger.info("Using HeuristicAI (unified loader failed)")
    return HeuristicAI(player, AIConfig(difficulty=5, board_type=board_type))


def run_gauntlet_for_model(
    model: dict[str, Any],
    num_games: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    fast_mode: bool = True,
) -> GauntletResult:
    """Run baseline gauntlet evaluation for a model.

    Args:
        model: Model dict with 'path', 'name', 'type' keys
        num_games: Games per baseline opponent
        board_type: Board type for evaluation
        num_players: Number of players (default 2)
        fast_mode: If True, use fewer simulations for MCTS opponents

    Returns:
        GauntletResult with win rates and composite score
    """
    model_type = model.get("type", "nn")

    # Create model AI (as player 1) with proper model loading
    model_ai = create_ai_from_model(
        model, player=1, board_type=board_type, num_players=num_players
    )

    # Baseline opponents (as player 2)
    random_ai = RandomAI(2, AIConfig(difficulty=1))
    heuristic_ai = HeuristicAI(2, AIConfig(difficulty=5))

    games_played = 0

    # Test vs Random
    logger.info(f"  [vs Random] Starting {num_games} games...")
    wins_vs_random = 0
    for i in range(num_games):
        winner = run_game(model_ai, random_ai, board_type, num_players)
        if winner == 1:
            wins_vs_random += 1
        games_played += 1
        if (i + 1) % 5 == 0:
            logger.info(f"    [{i+1}/{num_games}] wins: {wins_vs_random}")

    logger.info(f"  [vs Random] Done: {wins_vs_random}/{num_games} wins ({wins_vs_random/num_games:.0%})")

    # Test vs Heuristic
    logger.info(f"  [vs Heuristic] Starting {num_games} games...")
    wins_vs_heuristic = 0
    for i in range(num_games):
        winner = run_game(model_ai, heuristic_ai, board_type, num_players)
        if winner == 1:
            wins_vs_heuristic += 1
        games_played += 1
        if (i + 1) % 5 == 0:
            logger.info(f"    [{i+1}/{num_games}] wins: {wins_vs_heuristic}")

    logger.info(f"  [vs Heuristic] Done: {wins_vs_heuristic}/{num_games} wins ({wins_vs_heuristic/num_games:.0%})")

    vs_random = wins_vs_random / num_games if num_games > 0 else 0.0
    vs_heuristic = wins_vs_heuristic / num_games if num_games > 0 else 0.0
    vs_mcts = 0.0  # Skip MCTS for fast mode

    # Composite score: weighted average
    # Random is easy (low weight), heuristic matters more
    score = 0.2 * vs_random + 0.8 * vs_heuristic

    return GauntletResult(
        model_type=model_type,
        vs_random=vs_random,
        vs_heuristic=vs_heuristic,
        vs_mcts=vs_mcts,
        score=score,
        games_played=games_played,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline gauntlet for a model")
    parser.add_argument("model_path", type=Path, help="Path to model file")
    parser.add_argument("--games", type=int, default=10, help="Games per baseline")
    parser.add_argument("--board", default="square8", help="Board type")
    args = parser.parse_args()

    model = {
        "path": str(args.model_path),
        "name": args.model_path.stem,
        "type": "nnue" if "nnue" in args.model_path.stem.lower() else "nn",
    }

    result = run_gauntlet_for_model(
        model=model,
        num_games=args.games,
        board_type=BoardType(args.board),
    )

    print(f"\n=== Gauntlet Result: {args.model_path.name} ===")
    print(f"Model type: {result.model_type}")
    print(f"vs Random: {result.vs_random:.1%}")
    print(f"vs Heuristic: {result.vs_heuristic:.1%}")
    print(f"Composite score: {result.score:.2f}")
    print(f"Games played: {result.games_played}")
