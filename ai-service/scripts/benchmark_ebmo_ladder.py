#!/usr/bin/env python3
"""Benchmark EBMO against ladder AIs (Heuristic, PolicyOnly, MCTS)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass
from collections.abc import Callable

from app.ai.ebmo_ai import EBMO_AI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.ai.base import BaseAI
from app.models import AIConfig, BoardType, GameStatus
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    total_moves: int = 0
    games: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0

    @property
    def avg_moves(self) -> float:
        return self.total_moves / self.games if self.games > 0 else 0.0


def play_game(ai1: BaseAI, ai2: BaseAI, max_moves: int = 500) -> tuple[int | None, int]:
    """Play a single game."""
    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()
    move_count = 0
    ais = {1: ai1, 2: ai2}

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_ai = ais[state.current_player]
        move = current_ai.select_move(state)
        if move is None:
            break
        state = engine.apply_move(state, move)
        move_count += 1

    return state.winner, move_count


def run_match(
    ai1_factory: Callable[[int], BaseAI],
    ai2_factory: Callable[[int], BaseAI],
    ai1_name: str,
    ai2_name: str,
    num_games: int = 20,
) -> MatchResult:
    """Run a match between two AI types."""
    result = MatchResult()

    for game_num in range(num_games):
        # Alternate colors
        if game_num % 2 == 0:
            ai1 = ai1_factory(1)
            ai2 = ai2_factory(2)
            ai1_player = 1
        else:
            ai1 = ai1_factory(2)
            ai2 = ai2_factory(1)
            ai1_player = 2

        winner, moves = play_game(
            ai1 if ai1_player == 1 else ai2,
            ai2 if ai1_player == 1 else ai1,
            max_moves=1000,
        )

        result.games += 1
        result.total_moves += moves

        if winner == ai1_player:
            result.wins += 1
            outcome = f"{ai1_name} wins"
        elif winner is not None:
            result.losses += 1
            outcome = f"{ai2_name} wins"
        else:
            outcome = "No winner"

        logger.info(f"  Game {game_num + 1}: {outcome} ({moves} moves)")

    return result


def main():
    model_path = "models/ebmo_56ch/ebmo_quality_best.pt"
    games_per_match = 20

    logger.info("=" * 70)
    logger.info("EBMO Ladder Benchmark")
    logger.info("=" * 70)

    config = AIConfig(difficulty=5)

    # Factory functions
    def ebmo_factory(player: int) -> BaseAI:
        return EBMO_AI(player, config, model_path)

    def random_factory(player: int) -> BaseAI:
        return RandomAI(player, config)

    def heuristic_factory(player: int) -> BaseAI:
        return HeuristicAI(player, config)

    # Run benchmarks
    results = {}

    # 1. EBMO vs Random (baseline confirmation)
    logger.info("\n[1/3] EBMO vs Random AI")
    logger.info("-" * 40)
    results["random"] = run_match(ebmo_factory, random_factory, "EBMO", "Random", games_per_match)

    # 2. EBMO vs Heuristic (critical test)
    logger.info("\n[2/3] EBMO vs Heuristic AI (D2)")
    logger.info("-" * 40)
    results["heuristic"] = run_match(ebmo_factory, heuristic_factory, "EBMO", "Heuristic", games_per_match)

    # 3. Heuristic vs Random (calibration)
    logger.info("\n[3/3] Heuristic vs Random (calibration)")
    logger.info("-" * 40)
    results["heuristic_random"] = run_match(heuristic_factory, random_factory, "Heuristic", "Random", games_per_match)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nEBMO vs Random:     {results['random'].wins}-{results['random'].losses} "
                f"({results['random'].win_rate*100:.1f}% win rate, {results['random'].avg_moves:.1f} avg moves)")

    logger.info(f"EBMO vs Heuristic:  {results['heuristic'].wins}-{results['heuristic'].losses} "
                f"({results['heuristic'].win_rate*100:.1f}% win rate, {results['heuristic'].avg_moves:.1f} avg moves)")

    logger.info(f"Heuristic vs Random: {results['heuristic_random'].wins}-{results['heuristic_random'].losses} "
                f"({results['heuristic_random'].win_rate*100:.1f}% win rate)")

    # Assessment
    logger.info("\n" + "-" * 70)
    logger.info("LADDER PLACEMENT ASSESSMENT:")

    if results['heuristic'].win_rate >= 0.5:
        logger.info("  EBMO >= Heuristic (D2) - Can place at D3 or higher")
    else:
        logger.info("  EBMO < Heuristic (D2) - Needs improvement before D3")

    if results['random'].win_rate >= 0.8:
        logger.info("  EBMO >> Random - Strong baseline performance")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
