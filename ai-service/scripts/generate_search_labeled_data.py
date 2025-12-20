#!/usr/bin/env python
"""Generate EBMO training data with move-level quality labels from search.

Instead of labeling moves by game outcome (noisy), this script uses
Gumbel MCTS or minimax evaluation to score individual moves:

1. Play games using Gumbel MCTS (with neural network guidance)
2. At each position, extract search statistics (visit counts, Q-values)
3. Label moves by their search quality (most-visited = best, others = negatives)

This provides direct supervision on move quality from principled search.

Usage:
    python scripts/generate_search_labeled_data.py --output data/training/ebmo_search_500.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.factory import AIFactory
from app.ai.heuristic_ai import HeuristicAI
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.neural_net import NeuralNetAI
from app.ai.ebmo_network import ActionFeatureExtractor
from app.models.core import AIType, AIConfig, BoardType
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_search_labeled")


def extract_mcts_quality_labels(
    mcts_ai: GumbelMCTSAI,
    valid_moves: list,
) -> list[tuple[int, float, float]]:
    """Extract move quality labels from MCTS search results.

    Returns:
        List of (move_index, visit_fraction, q_value) tuples, sorted by visits descending
    """
    if mcts_ai._last_search_actions is None:
        return []

    # Build move to index mapping
    move_to_idx = {}
    for i, move in enumerate(valid_moves):
        # Create a unique key for the move
        key = f"{move.type.value}"
        if hasattr(move, 'from_pos') and move.from_pos:
            key += f"_{move.from_pos.x},{move.from_pos.y}"
        if hasattr(move, 'to') and move.to:
            key += f"_{move.to.x},{move.to.y}"
        move_to_idx[key] = i

    # Get search results
    total_visits = sum(a.visit_count for a in mcts_ai._last_search_actions)
    if total_visits == 0:
        return []

    max_visits = max(a.visit_count for a in mcts_ai._last_search_actions)

    results = []
    for action in mcts_ai._last_search_actions:
        # Find matching move index
        key = f"{action.move.type.value}"
        if hasattr(action.move, 'from_pos') and action.move.from_pos:
            key += f"_{action.move.from_pos.x},{action.move.from_pos.y}"
        if hasattr(action.move, 'to') and action.move.to:
            key += f"_{action.move.to.x},{action.move.to.y}"

        if key in move_to_idx:
            idx = move_to_idx[key]
            visit_fraction = action.visit_count / total_visits
            q_value = action.completed_q(max_visits)
            results.append((idx, visit_fraction, q_value))

    # Sort by visit fraction descending
    results.sort(key=lambda x: -x[1])
    return results


def generate_labeled_samples_mcts(
    num_games: int = 100,
    board_type: BoardType = BoardType.SQUARE8,
    mcts_simulations: int = 100,
    min_moves_per_game: int = 10,
    max_moves_per_game: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training samples using Gumbel MCTS for quality labels.

    Returns:
        (boards, globals, actions, is_best, relative_scores) arrays
    """
    engine = GameEngine()
    nn = NeuralNetAI(1, AIConfig(difficulty=5))  # For feature extraction
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

        # Create Gumbel MCTS players
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

        game_samples = 0

        for _move_num in range(max_moves_per_game):
            if state.winner is not None:
                break
            if hasattr(state.game_status, 'value') and state.game_status.value != 'active':
                break

            current_player = state.current_player
            mcts_ai = ais[current_player]

            # Get valid moves
            valid_moves = mcts_ai.get_valid_moves(state)
            if not valid_moves or len(valid_moves) < 2:
                move = mcts_ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            # Extract state features before the move
            try:
                board_feat, global_feat = nn._extract_features(state)
            except Exception:
                move = mcts_ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            # Run MCTS search (this also selects the move)
            selected_move = mcts_ai.select_move(state)
            if selected_move is None:
                continue

            # Extract quality labels from search results
            quality_results = extract_mcts_quality_labels(mcts_ai, valid_moves)

            if len(quality_results) < 2:
                # Not enough data, skip
                state = engine.apply_move(state, selected_move)
                continue

            # Normalize visit fractions to [0, 1]
            visit_fractions = [vf for _, vf, _ in quality_results]
            max_visits = max(visit_fractions)
            min_visits = min(visit_fractions)
            visit_range = max_visits - min_visits

            if visit_range < 1e-6:
                state = engine.apply_move(state, selected_move)
                continue

            # Add samples for all evaluated moves
            for rank, (move_idx, visit_fraction, _q_value) in enumerate(quality_results):
                if move_idx >= len(valid_moves):
                    continue

                move = valid_moves[move_idx]
                action_feat = action_extractor.extract_features(move)

                # Best move is the one with highest visits
                is_best = 1.0 if rank == 0 else 0.0

                # Relative score based on visit fraction
                relative_score = (visit_fraction - min_visits) / visit_range

                all_boards.append(board_feat)
                all_globals.append(global_feat)
                all_actions.append(action_feat)
                all_is_best.append(is_best)
                all_relative_scores.append(relative_score)

                game_samples += 1
                total_samples += 1

            state = engine.apply_move(state, selected_move)

        if game_samples >= min_moves_per_game:
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


def generate_labeled_samples_heuristic(
    num_games: int = 100,
    board_type: BoardType = BoardType.SQUARE8,
    evaluator_difficulty: int = 5,
    min_moves_per_game: int = 10,
    max_moves_per_game: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training samples with heuristic-based quality labels (fallback)."""
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

        evaluator = HeuristicAI(1, AIConfig(difficulty=evaluator_difficulty))
        player1 = AIFactory.create(AIType.HEURISTIC, 1, AIConfig(difficulty=5))
        player2 = AIFactory.create(AIType.HEURISTIC, 2, AIConfig(difficulty=5))
        ais = {1: player1, 2: player2}

        game_samples = 0

        for _move_num in range(max_moves_per_game):
            if state.winner is not None:
                break

            current_player = state.current_player
            ai = ais[current_player]
            valid_moves = ai.get_valid_moves(state)

            if not valid_moves or len(valid_moves) < 2:
                move = ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            try:
                board_feat, global_feat = nn._extract_features(state)
            except Exception:
                move = ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            # Evaluate all moves with heuristic
            move_scores = []
            for i, move in enumerate(valid_moves):
                next_state = engine.apply_move(state, move)
                score = evaluator.evaluate_position(next_state)
                if evaluator.player_number != current_player:
                    score = -score
                move_scores.append((i, score))

            move_scores.sort(key=lambda x: -x[1])

            scores = [s for _, s in move_scores]
            score_range = max(scores) - min(scores)

            if score_range < 1e-6:
                move = ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            for rank, (move_idx, score) in enumerate(move_scores):
                move = valid_moves[move_idx]
                action_feat = action_extractor.extract_features(move)
                is_best = 1.0 if rank == 0 else 0.0
                relative_score = (score - min(scores)) / score_range

                all_boards.append(board_feat)
                all_globals.append(global_feat)
                all_actions.append(action_feat)
                all_is_best.append(is_best)
                all_relative_scores.append(relative_score)

                game_samples += 1
                total_samples += 1

            best_move = valid_moves[move_scores[0][0]]
            state = engine.apply_move(state, best_move)

        if game_samples >= min_moves_per_game:
            games_completed += 1
            if games_completed % 20 == 0:
                logger.info(f"Completed {games_completed}/{num_games} games, {total_samples} samples")

    return (
        np.array(all_boards, dtype=np.float32),
        np.array(all_globals, dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
        np.array(all_is_best, dtype=np.float32),
        np.array(all_relative_scores, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate EBMO training data with search-based quality labels"
    )
    parser.add_argument(
        "--output", type=str, default="data/training/ebmo_mcts_500.npz",
        help="Output NPZ file path"
    )
    parser.add_argument(
        "--num-games", type=int, default=500,
        help="Number of games to generate"
    )
    parser.add_argument(
        "--mode", type=str, choices=["mcts", "heuristic"], default="mcts",
        help="Evaluation mode: 'mcts' uses Gumbel MCTS (better), 'heuristic' uses static eval"
    )
    parser.add_argument(
        "--mcts-simulations", type=int, default=100,
        help="Number of MCTS simulations per move (for mcts mode)"
    )

    args = parser.parse_args()

    logger.info(f"Generating {args.num_games} games with {args.mode} labels...")

    if args.mode == "mcts":
        boards, globals_arr, actions, is_best, relative_scores = generate_labeled_samples_mcts(
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations,
        )
    else:
        boards, globals_arr, actions, is_best, relative_scores = generate_labeled_samples_heuristic(
            num_games=args.num_games,
        )

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to NPZ
    np.savez(
        output_path,
        boards=boards,
        globals=globals_arr,
        actions=actions,
        is_best=is_best,
        relative_scores=relative_scores,
    )

    logger.info(f"Saved {len(boards)} samples to {output_path}")
    logger.info(f"  Best moves: {int(is_best.sum())} ({100*is_best.mean():.1f}%)")
    logger.info(f"  Score range: {relative_scores.min():.3f} - {relative_scores.max():.3f}")


if __name__ == "__main__":
    main()
