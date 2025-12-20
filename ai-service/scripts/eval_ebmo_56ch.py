#!/usr/bin/env python3
"""Evaluate EBMO 56-channel model against Random AI."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.ai.ebmo_ai import EBMO_AI
from app.ai.random_ai import RandomAI
from app.models import AIConfig, BoardType, GameStatus
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def play_game(ai1, ai2, max_moves: int = 500) -> tuple[int | None, int]:
    """Play a single game between two AIs."""
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


def main():
    model_path = "models/ebmo_56ch/ebmo_quality_best.pt"
    num_games = 10

    logger.info("=" * 60)
    logger.info("EBMO 56-Channel Model Evaluation")
    logger.info("=" * 60)

    config = AIConfig(difficulty=5)

    # Track results
    ebmo_wins = 0
    random_wins = 0
    total_moves = 0

    for game_num in range(num_games):
        # Alternate who plays first
        ebmo_player = 1 if game_num % 2 == 0 else 2
        random_player = 2 if game_num % 2 == 0 else 1

        ebmo_ai = EBMO_AI(ebmo_player, config, model_path)
        random_ai = RandomAI(random_player, config)

        if ebmo_player == 1:
            ai1, ai2 = ebmo_ai, random_ai
        else:
            ai1, ai2 = random_ai, ebmo_ai

        winner, moves = play_game(ai1, ai2, max_moves=1000)
        total_moves += moves

        if winner == ebmo_player:
            ebmo_wins += 1
            result = "EBMO wins"
        elif winner == random_player:
            random_wins += 1
            result = "Random wins"
        else:
            result = f"No winner (moves={moves})"

        logger.info(f"Game {game_num + 1}: {result} (moves={moves}, EBMO as P{ebmo_player})")

    logger.info("=" * 60)
    logger.info(f"Results: EBMO {ebmo_wins} - {random_wins} Random")
    logger.info(f"Win rate: {100 * ebmo_wins / num_games:.1f}%")
    logger.info(f"Average moves: {total_moves / num_games:.1f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
