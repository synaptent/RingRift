#!/usr/bin/env python3
"""Generate self-play training data for EBMO improvement.

Uses EBMO vs EBMO games to generate higher-quality training data,
labeling moves with game outcomes (winner's moves = positive, loser's = negative).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import ActionFeatureExtractor
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.training.feature_extractor import FeatureExtractor
from app.models.core import AIConfig, BoardType, GameStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_ebmo_selfplay")


def play_selfplay_game(
    ai1: EBMO_AI,
    ai2: EBMO_AI,
    feature_extractor: FeatureExtractor,
    action_extractor: ActionFeatureExtractor,
    max_moves: int = 500,
) -> tuple[list, int | None]:
    """Play a self-play game and collect training data.

    Returns:
        (samples, winner) where samples is a list of (board_features, action_features, player)
    """
    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    samples = []
    ais = {1: ai1, 2: ai2}
    move_count = 0

    # Track frame history for 56-channel features
    frame_history = []

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player
        current_ai = ais[current_player]

        # Extract features before move
        features = feature_extractor.extract(state, current_player)
        frame_history.append(features)
        if len(frame_history) > 4:
            frame_history.pop(0)

        # Stack frames for 56-channel input
        if len(frame_history) >= 4:
            stacked = np.concatenate(frame_history[-4:], axis=0)
        else:
            # Pad with zeros if not enough history
            padding = [np.zeros_like(features)] * (4 - len(frame_history))
            stacked = np.concatenate(padding + frame_history, axis=0)

        # Get move
        move = current_ai.select_move(state)
        if move is None:
            break

        # Extract action features
        action_features = action_extractor.extract([move])[0]

        # Store sample
        samples.append({
            'board': stacked,
            'action': action_features,
            'player': current_player,
            'move_num': move_count,
        })

        state = engine.apply_move(state, move)
        move_count += 1

    return samples, state.winner


def generate_data(
    num_games: int = 100,
    model_path: str = "models/ebmo_56ch/ebmo_quality_best.pt",
    output_path: str = "data/training/ebmo_selfplay.npz",
) -> None:
    """Generate self-play training data."""

    config = AIConfig(difficulty=5)
    action_extractor = ActionFeatureExtractor(8)
    feature_extractor = FeatureExtractor(board_size=8, history_length=1)

    all_boards = []
    all_actions = []
    all_labels = []  # 1 for winner's moves, 0 for loser's moves
    all_move_progress = []  # Move number / total moves (for temporal weighting)

    games_completed = 0
    p1_wins = 0
    p2_wins = 0

    logger.info(f"Generating {num_games} self-play games...")

    while games_completed < num_games:
        # Create fresh AIs for each game
        ai1 = EBMO_AI(1, config, model_path)
        ai2 = EBMO_AI(2, config, model_path)

        samples, winner = play_selfplay_game(
            ai1, ai2, feature_extractor, action_extractor
        )

        if winner is None:
            logger.warning(f"Game {games_completed + 1}: No winner, skipping")
            continue

        if winner == 1:
            p1_wins += 1
        else:
            p2_wins += 1

        # Label samples based on outcome
        total_moves = len(samples)
        for sample in samples:
            all_boards.append(sample['board'])
            all_actions.append(sample['action'])

            # Winner's moves are positive examples
            label = 1.0 if sample['player'] == winner else 0.0
            all_labels.append(label)

            # Move progress for potential temporal weighting
            progress = sample['move_num'] / max(total_moves, 1)
            all_move_progress.append(progress)

        games_completed += 1

        if games_completed % 10 == 0:
            logger.info(f"Completed {games_completed}/{num_games} games "
                       f"(P1: {p1_wins}, P2: {p2_wins})")

    # Convert to arrays
    boards = np.array(all_boards, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    progress = np.array(all_move_progress, dtype=np.float32)

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        boards=boards,
        actions=actions,
        labels=labels,
        progress=progress,
        metadata={
            'num_games': num_games,
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'model_path': model_path,
        }
    )

    logger.info(f"\nSaved {len(boards)} samples to {output_path}")
    logger.info(f"  Board shape: {boards.shape}")
    logger.info(f"  Action shape: {actions.shape}")
    logger.info(f"  Winner samples: {int(labels.sum())}")
    logger.info(f"  Loser samples: {len(labels) - int(labels.sum())}")
    logger.info(f"  P1 wins: {p1_wins}, P2 wins: {p2_wins}")


def main():
    parser = argparse.ArgumentParser(description="Generate EBMO self-play data")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--model", type=str,
                        default="models/ebmo_56ch/ebmo_quality_best.pt",
                        help="EBMO model path")
    parser.add_argument("--output", type=str,
                        default="data/training/ebmo_selfplay.npz",
                        help="Output path")
    args = parser.parse_args()

    generate_data(
        num_games=args.games,
        model_path=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
