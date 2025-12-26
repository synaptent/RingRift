#!/usr/bin/env python3
"""Quick benchmark harness for AI baselines (heuristic vs random)."""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameStatus
from app.training.initial_state import create_initial_state

AI_FACTORY = Callable[[int], object]


def run_game(ai1_factory: AI_FACTORY, ai2_factory: AI_FACTORY, board_type: BoardType, num_players: int, game_idx: int = 0):
    """Run a single game. Returns winner (1-indexed) or None for draw."""
    state = create_initial_state(board_type=board_type, num_players=num_players)
    ais = {
        1: ai1_factory(1),
        **{player: ai2_factory(player) for player in range(2, num_players + 1)},
    }

    # Reset AIs with unique seeds per game for variety
    for p, ai in ais.items():
        if hasattr(ai, 'reset_for_new_game'):
            ai.reset_for_new_game(rng_seed=game_idx * 1000 + p)

    max_moves = 500
    for _ in range(max_moves):
        if state.status == GameStatus.COMPLETED:
            break
        current = state.current_player
        ai = ais[current]
        move = ai.select_move(state)
        if move is None:
            break
        state = GameEngine.apply_move(state, move)

    return state.winner


def benchmark_ai(
    test_ai_factory: AI_FACTORY,
    test_name: str,
    opponent_ai_factory: AI_FACTORY,
    opponent_name: str,
    board_type: BoardType,
    num_players: int,
    games: int = 10,
):
    """Benchmark an AI against an opponent."""
    wins = 0
    for i in range(games):
        if i % 2 == 0:
            winner = run_game(test_ai_factory, opponent_ai_factory, board_type, num_players, game_idx=i)
            if winner == 1:
                wins += 1
        else:
            winner = run_game(opponent_ai_factory, test_ai_factory, board_type, num_players, game_idx=i)
            if winner == 2:
                wins += 1

    win_rate = wins / games * 100
    print(f"  {test_name} vs {opponent_name}: {wins}/{games} ({win_rate:.0f}%)")
    return win_rate


if __name__ == "__main__":
    config = AIConfig(difficulty=5)
    def heuristic_factory(player):
        return HeuristicAI(player, config)
    def random_factory(player):
        return RandomAI(player, config)

    print("\n=== Square8 3-Player Baseline Benchmark ===")
    benchmark_ai(heuristic_factory, "Heuristic", random_factory, "Random", BoardType.SQUARE8, 3, games=5)

    print("\n=== Square19 2-Player Baseline Benchmark ===")
    benchmark_ai(heuristic_factory, "Heuristic", random_factory, "Random", BoardType.SQUARE19, 2, games=3)

    print("\n=== Hexagonal 3-Player Baseline Benchmark ===")
    benchmark_ai(heuristic_factory, "Heuristic", random_factory, "Random", BoardType.HEXAGONAL, 3, games=3)

    print("\nBaseline benchmarks complete!")
