#!/usr/bin/env python3
"""Generate 56-channel MCTS-labeled training data for EBMO.

Uses Gumbel MCTS to label moves with quality scores, and stacks
4 frames of 14-channel features to create 56-channel input.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.neural_net import NeuralNetAI
from app.ai.ebmo_network import ActionFeatureExtractor
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models.core import AIConfig, BoardType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_mcts_56ch")


def extract_mcts_quality(mcts_ai: GumbelMCTSAI, valid_moves: list) -> list[tuple[int, float]]:
    """Extract move quality labels from MCTS search results."""
    if mcts_ai._last_search_actions is None:
        return []

    total_visits = sum(a.visit_count for a in mcts_ai._last_search_actions)
    if total_visits == 0:
        return []

    # Build move key mapping
    move_to_idx = {}
    for i, move in enumerate(valid_moves):
        key = f"{move.type.value}"
        if hasattr(move, 'from_pos') and move.from_pos:
            key += f"_{move.from_pos.x},{move.from_pos.y}"
        if hasattr(move, 'to') and move.to:
            key += f"_{move.to.x},{move.to.y}"
        move_to_idx[key] = i

    results = []
    for action in mcts_ai._last_search_actions:
        key = f"{action.move.type.value}"
        if hasattr(action.move, 'from_pos') and action.move.from_pos:
            key += f"_{action.move.from_pos.x},{action.move.from_pos.y}"
        if hasattr(action.move, 'to') and action.move.to:
            key += f"_{action.move.to.x},{action.move.to.y}"

        if key in move_to_idx:
            idx = move_to_idx[key]
            visit_fraction = action.visit_count / total_visits
            results.append((idx, visit_fraction))

    results.sort(key=lambda x: -x[1])
    return results


def generate_data(
    num_games: int = 100,
    mcts_simulations: int = 100,
    board_type: BoardType = BoardType.SQUARE8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate 56-channel training data with MCTS labels."""
    engine = GameEngine()
    nn = NeuralNetAI(1, AIConfig(difficulty=5))
    action_extractor = ActionFeatureExtractor(8)

    all_boards = []
    all_globals = []
    all_actions = []
    all_is_best = []
    all_relative_scores = []

    games_completed = 0
    total_samples = 0

    while games_completed < num_games:
        state = create_initial_state(board_type=board_type, num_players=2)

        # Create MCTS players
        mcts_config = AIConfig(
            difficulty=7,
            extra={
                'num_sampled_actions': 16,
                'simulation_budget': mcts_simulations,
            }
        )
        player1 = GumbelMCTSAI(1, mcts_config)
        player2 = GumbelMCTSAI(2, mcts_config)
        ais = {1: player1, 2: player2}

        # Store frame history for 56-channel stacking
        frame_history = []

        for move_num in range(100):
            if state.winner is not None:
                break

            current_player = state.current_player
            mcts_ai = ais[current_player]

            valid_moves = mcts_ai.get_valid_moves(state)
            if not valid_moves or len(valid_moves) < 2:
                move = mcts_ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            # Extract features
            try:
                board_feat, global_feat = nn._extract_features(state)
            except Exception:
                move = mcts_ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            # Update frame history
            frame_history.append(board_feat)
            if len(frame_history) > 4:
                frame_history.pop(0)

            # Create 56-channel input by stacking history
            if len(frame_history) < 4:
                # Pad with current frame if not enough history
                stacked = np.concatenate([frame_history[0]] * (4 - len(frame_history)) + frame_history, axis=0)
            else:
                stacked = np.concatenate(frame_history, axis=0)

            # Run MCTS search
            selected_move = mcts_ai.select_move(state)
            if selected_move is None:
                continue

            # Extract quality labels
            quality_results = extract_mcts_quality(mcts_ai, valid_moves)
            if len(quality_results) < 2:
                state = engine.apply_move(state, selected_move)
                continue

            # Normalize visit fractions
            visit_fractions = [vf for _, vf in quality_results]
            max_visits = max(visit_fractions)
            min_visits = min(visit_fractions)
            visit_range = max_visits - min_visits

            if visit_range < 1e-6:
                state = engine.apply_move(state, selected_move)
                continue

            # Add samples
            for rank, (move_idx, visit_fraction) in enumerate(quality_results):
                if move_idx >= len(valid_moves):
                    continue

                move = valid_moves[move_idx]
                action_feat = action_extractor.extract_features(move)
                is_best = 1.0 if rank == 0 else 0.0
                relative_score = (visit_fraction - min_visits) / visit_range

                all_boards.append(stacked)
                all_globals.append(global_feat)
                all_actions.append(action_feat)
                all_is_best.append(is_best)
                all_relative_scores.append(relative_score)
                total_samples += 1

            state = engine.apply_move(state, selected_move)

        games_completed += 1
        if games_completed % 10 == 0:
            logger.info(f"Completed {games_completed}/{num_games} games, {total_samples} samples")

    return (
        np.array(all_boards, dtype=np.float32),
        np.array(all_globals, dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
        np.array(all_is_best, dtype=np.float32),
        np.array(all_relative_scores, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate 56-channel MCTS training data")
    parser.add_argument("--output", type=str, default="data/training/ebmo_mcts_56ch.npz")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--mcts-simulations", type=int, default=100)
    args = parser.parse_args()

    logger.info(f"Generating {args.num_games} games with MCTS labels (56 channels)...")

    boards, globals_arr, actions, is_best, relative_scores = generate_data(
        num_games=args.num_games,
        mcts_simulations=args.mcts_simulations,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        boards=boards,
        globals=globals_arr,
        actions=actions,
        is_best=is_best,
        relative_scores=relative_scores,
    )

    logger.info(f"Saved {len(boards)} samples to {output_path}")
    logger.info(f"Board shape: {boards.shape}")


if __name__ == "__main__":
    main()
