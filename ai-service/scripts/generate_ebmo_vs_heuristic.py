#!/usr/bin/env python3
"""Generate training data from EBMO vs Heuristic games.

EBMO learns by seeing what moves Heuristic makes in winning games.
Uses contrastive labeling: winner's moves = positive, loser's = negative.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import numpy as np

from app.ai.ebmo_ai import EBMO_AI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.neural_net import NeuralNetAI
from app.ai.ebmo_network import ActionFeatureExtractor
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models.core import AIConfig, BoardType, GameStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_ebmo_vs_heuristic")


def play_game(
    ai1, ai2,
    nn: NeuralNetAI,
    action_extractor: ActionFeatureExtractor,
    max_moves: int = 500,
):
    """Play a game and collect training data."""
    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    samples = []
    ais = {1: ai1, 2: ai2}
    move_count = 0
    frame_history = []

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player
        current_ai = ais[current_player]

        # Extract features using NeuralNetAI
        try:
            board_feat, _ = nn._extract_features(state)
            frame_history.append(board_feat)
            if len(frame_history) > 4:
                frame_history.pop(0)
        except Exception:
            move = current_ai.select_move(state)
            if move:
                state = engine.apply_move(state, move)
            move_count += 1
            continue

        # Stack frames for 56-channel input
        if len(frame_history) >= 4:
            stacked = np.concatenate(frame_history[-4:], axis=0)
        else:
            padding = [frame_history[0]] * (4 - len(frame_history))
            stacked = np.concatenate(padding + frame_history, axis=0)

        # Get move
        move = current_ai.select_move(state)
        if move is None:
            break

        # Extract action features
        action_features = action_extractor.extract_features(move)

        # Record which AI made this move
        ai_type = "ebmo" if isinstance(current_ai, EBMO_AI) else "heuristic"

        samples.append({
            'board': stacked,
            'action': action_features,
            'player': current_player,
            'ai_type': ai_type,
            'move_num': move_count,
        })

        state = engine.apply_move(state, move)
        move_count += 1

    return samples, state.winner


def generate_data(
    num_games: int = 100,
    model_path: str = "models/ebmo_56ch/ebmo_quality_best.pt",
    output_path: str = "data/training/ebmo_vs_heuristic.npz",
):
    """Generate training data from EBMO vs Heuristic games."""

    config = AIConfig(difficulty=5)
    action_extractor = ActionFeatureExtractor(8)
    nn = NeuralNetAI(1, config)  # For feature extraction

    all_boards = []
    all_actions = []
    all_labels = []  # 1 for winner's moves, 0 for loser's moves
    all_ai_types = []  # "ebmo" or "heuristic"

    ebmo_wins = 0
    heuristic_wins = 0
    games_completed = 0

    logger.info(f"Generating {num_games} EBMO vs Heuristic games...")

    while games_completed < num_games:
        # Alternate who plays first
        ebmo_player = 1 if games_completed % 2 == 0 else 2
        heuristic_player = 2 if games_completed % 2 == 0 else 1

        ai1 = EBMO_AI(1, config, model_path)
        ai2 = HeuristicAI(2, config)

        if ebmo_player == 1:
            ais = {1: ai1, 2: ai2}
        else:
            ais = {1: ai2, 2: ai1}

        samples, winner = play_game(
            ais[1], ais[2], nn, action_extractor
        )

        if winner is None:
            logger.warning(f"Game {games_completed + 1}: No winner")
            continue

        if winner == ebmo_player:
            ebmo_wins += 1
        else:
            heuristic_wins += 1

        # Label samples
        for sample in samples:
            all_boards.append(sample['board'])
            all_actions.append(sample['action'])
            all_ai_types.append(1.0 if sample['ai_type'] == 'heuristic' else 0.0)

            # Winner's moves are positive
            label = 1.0 if sample['player'] == winner else 0.0
            all_labels.append(label)

        games_completed += 1

        if games_completed % 10 == 0:
            logger.info(f"Completed {games_completed}/{num_games} "
                       f"(EBMO: {ebmo_wins}, Heuristic: {heuristic_wins})")

    # Convert and save
    boards = np.array(all_boards, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    ai_types = np.array(all_ai_types, dtype=np.float32)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        boards=boards,
        actions=actions,
        labels=labels,
        ai_types=ai_types,
        metadata={
            'num_games': num_games,
            'ebmo_wins': ebmo_wins,
            'heuristic_wins': heuristic_wins,
            'model_path': model_path,
        }
    )

    logger.info(f"\nSaved {len(boards)} samples to {output_path}")
    logger.info(f"  EBMO wins: {ebmo_wins}, Heuristic wins: {heuristic_wins}")
    logger.info(f"  Winner samples: {int(labels.sum())}")
    logger.info(f"  Heuristic move samples: {int(ai_types.sum())}")


def main():
    parser = argparse.ArgumentParser(description="Generate EBMO vs Heuristic data")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--model", type=str,
                        default="models/ebmo_56ch/ebmo_quality_best.pt",
                        help="EBMO model path")
    parser.add_argument("--output", type=str,
                        default="data/training/ebmo_vs_heuristic.npz",
                        help="Output path")
    args = parser.parse_args()

    generate_data(
        num_games=args.games,
        model_path=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
