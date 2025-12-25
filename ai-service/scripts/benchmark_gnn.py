#!/usr/bin/env python3
"""GNN Model Benchmark Script.

Evaluates GNN and Hybrid models against baseline opponents to verify
training quality and compare against CNN models.

Usage:
    # Quick benchmark (10 games each)
    python scripts/benchmark_gnn.py --board hex8 --num-players 2

    # Full benchmark (100 games each)
    python scripts/benchmark_gnn.py --board hex8 --num-players 2 --games 100

    # Compare GNN vs CNN
    python scripts/benchmark_gnn.py --board hex8 --num-players 2 \
        --gnn-model models/gnn_hex8_2p/gnn_policy_best.pt \
        --cnn-model models/canonical_hex8_2p.pth

    # Benchmark hybrid model
    python scripts/benchmark_gnn.py --board hex8 --num-players 2 --model-type hybrid

Thresholds (from app/config/thresholds.py):
    - vs RANDOM: 85% win rate required (50% for 4-player)
    - vs HEURISTIC: 60% win rate required (35% for 4-player)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_type: str
    board_type: str
    num_players: int
    games_played: int
    vs_random_wins: int
    vs_random_rate: float
    vs_heuristic_wins: int
    vs_heuristic_rate: float
    elapsed_seconds: float
    passed_thresholds: bool


def play_gnn_vs_opponent(
    gnn_ai,
    opponent_ai,
    board_type,
    num_players: int,
    num_games: int,
    verbose: bool = False,
) -> tuple[int, int]:
    """Play games between GNN AI and an opponent.

    Returns:
        (gnn_wins, total_games)
    """
    from app.training.initial_state import create_initial_state
    from app.rules.default_engine import DefaultRulesEngine
    from app.models import GameStatus

    wins = 0
    engine = DefaultRulesEngine()

    for game_idx in range(num_games):
        state = create_initial_state(board_type, num_players)

        # Alternate who plays first
        if game_idx % 2 == 0:
            players = {1: gnn_ai, 2: opponent_ai}
            gnn_player = 1
        else:
            players = {1: opponent_ai, 2: gnn_ai}
            gnn_player = 2

        # Handle 3-4 player games
        for p in range(3, num_players + 1):
            players[p] = opponent_ai

        move_count = 0
        max_moves = 500

        while state.game_status != GameStatus.COMPLETED and move_count < max_moves:
            current = state.current_player
            ai = players.get(current, opponent_ai)

            # Get move from appropriate AI
            if hasattr(ai, 'select_move'):
                move = ai.select_move(state)
            elif hasattr(ai, 'get_move'):
                move = ai.get_move(state, current)
            else:
                move = ai.choose_action(state)

            if move is None:
                break

            # Apply move
            try:
                state = engine.apply_move(state, move)
            except Exception as e:
                logger.warning(f"Move failed: {e}")
                break

            move_count += 1

        # Check winner
        if state.game_status == GameStatus.COMPLETED and state.winner == gnn_player:
            wins += 1

        if verbose and (game_idx + 1) % 10 == 0:
            logger.info(f"  Game {game_idx + 1}/{num_games}: GNN wins = {wins}")

    return wins, num_games


def benchmark_gnn_model(
    board_type,
    num_players: int,
    model_path: str | None = None,
    model_type: str = "gnn",
    games_per_opponent: int = 20,
    device: str = "cpu",
    verbose: bool = False,
) -> BenchmarkResult:
    """Run full benchmark for a GNN model."""
    from app.models import BoardType
    from app.ai.gnn_ai import create_gnn_ai
    from app.ai.random_ai import RandomAI
    from app.ai.heuristic_ai import HeuristicAI
    from app.models import AIConfig

    # Parse board type
    if isinstance(board_type, str):
        board_type = BoardType(board_type)

    start_time = time.time()

    # Create GNN AI
    logger.info(f"Creating {model_type.upper()} AI for {board_type.value}_{num_players}p...")
    gnn_ai = create_gnn_ai(
        player_number=1,  # Will be reassigned per game
        model_path=model_path,
        device=device,
    )

    # Create baseline opponents
    config = AIConfig(difficulty=6)
    random_ai = RandomAI(2, config)
    heuristic_ai = HeuristicAI(2, config)

    # Benchmark vs RANDOM
    logger.info(f"Benchmarking vs RANDOM ({games_per_opponent} games)...")
    random_wins, random_games = play_gnn_vs_opponent(
        gnn_ai, random_ai, board_type, num_players, games_per_opponent, verbose
    )
    random_rate = random_wins / random_games if random_games > 0 else 0

    # Benchmark vs HEURISTIC
    logger.info(f"Benchmarking vs HEURISTIC ({games_per_opponent} games)...")
    heuristic_wins, heuristic_games = play_gnn_vs_opponent(
        gnn_ai, heuristic_ai, board_type, num_players, games_per_opponent, verbose
    )
    heuristic_rate = heuristic_wins / heuristic_games if heuristic_games > 0 else 0

    elapsed = time.time() - start_time

    # Check thresholds
    try:
        from app.config.thresholds import (
            get_min_win_rate_vs_random,
            get_min_win_rate_vs_heuristic,
        )
        random_threshold = get_min_win_rate_vs_random(num_players)
        heuristic_threshold = get_min_win_rate_vs_heuristic(num_players)
    except ImportError:
        random_threshold = 0.50 if num_players >= 4 else 0.85
        heuristic_threshold = 0.35 if num_players >= 4 else 0.60

    passed = random_rate >= random_threshold and heuristic_rate >= heuristic_threshold

    return BenchmarkResult(
        model_type=model_type,
        board_type=board_type.value,
        num_players=num_players,
        games_played=random_games + heuristic_games,
        vs_random_wins=random_wins,
        vs_random_rate=random_rate,
        vs_heuristic_wins=heuristic_wins,
        vs_heuristic_rate=heuristic_rate,
        elapsed_seconds=elapsed,
        passed_thresholds=passed,
    )


def compare_gnn_vs_cnn(
    board_type,
    num_players: int,
    gnn_path: str | None,
    cnn_path: str,
    games: int = 20,
    device: str = "cpu",
) -> dict:
    """Direct comparison between GNN and CNN models."""
    from app.models import BoardType
    from app.ai.gnn_ai import create_gnn_ai
    from app.ai.universal_ai import UniversalAI

    if isinstance(board_type, str):
        board_type = BoardType(board_type)

    logger.info(f"GNN vs CNN comparison ({games} games)...")

    # Create models
    gnn_ai = create_gnn_ai(player_number=1, model_path=gnn_path, device=device)
    cnn_ai = UniversalAI.from_checkpoint(
        cnn_path, player_number=2, board_type=board_type, num_players=num_players
    )

    gnn_wins, total = play_gnn_vs_opponent(
        gnn_ai, cnn_ai, board_type, num_players, games, verbose=True
    )

    return {
        "gnn_wins": gnn_wins,
        "cnn_wins": total - gnn_wins,
        "total_games": total,
        "gnn_win_rate": gnn_wins / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GNN models against baselines"
    )
    parser.add_argument(
        "--board", "-b", required=True,
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type"
    )
    parser.add_argument(
        "--num-players", "-n", type=int, default=2,
        choices=[2, 3, 4],
        help="Number of players"
    )
    parser.add_argument(
        "--games", "-g", type=int, default=20,
        help="Games per opponent"
    )
    parser.add_argument(
        "--model-type", "-t", default="gnn",
        choices=["gnn", "hybrid"],
        help="Model type to benchmark"
    )
    parser.add_argument(
        "--gnn-model", type=str, default=None,
        help="Path to GNN model checkpoint"
    )
    parser.add_argument(
        "--cnn-model", type=str, default=None,
        help="Path to CNN model for comparison"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to use (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GNN Benchmark")
    logger.info("=" * 60)
    logger.info(f"Board: {args.board}")
    logger.info(f"Players: {args.num_players}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Games per opponent: {args.games}")
    logger.info("=" * 60)

    # Run benchmark
    result = benchmark_gnn_model(
        board_type=args.board,
        num_players=args.num_players,
        model_path=args.gnn_model,
        model_type=args.model_type,
        games_per_opponent=args.games,
        device=args.device,
        verbose=args.verbose,
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {result.model_type.upper()}")
    logger.info(f"Config: {result.board_type}_{result.num_players}p")
    logger.info(f"Games: {result.games_played}")
    logger.info(f"Duration: {result.elapsed_seconds:.1f}s")
    logger.info("")
    logger.info(f"vs RANDOM:    {result.vs_random_wins}/{result.games_played//2} = {result.vs_random_rate:.1%}")
    logger.info(f"vs HEURISTIC: {result.vs_heuristic_wins}/{result.games_played//2} = {result.vs_heuristic_rate:.1%}")
    logger.info("")

    if result.passed_thresholds:
        logger.info("✓ PASSED threshold requirements")
    else:
        logger.info("✗ FAILED threshold requirements")

    # Optional GNN vs CNN comparison
    if args.cnn_model:
        logger.info("")
        logger.info("-" * 60)
        comparison = compare_gnn_vs_cnn(
            board_type=args.board,
            num_players=args.num_players,
            gnn_path=args.gnn_model,
            cnn_path=args.cnn_model,
            games=args.games,
            device=args.device,
        )
        logger.info(f"GNN vs CNN: {comparison['gnn_wins']}/{comparison['total_games']} = {comparison['gnn_win_rate']:.1%}")

    logger.info("=" * 60)

    return 0 if result.passed_thresholds else 1


if __name__ == "__main__":
    sys.exit(main())
