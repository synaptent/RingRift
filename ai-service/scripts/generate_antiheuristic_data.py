#!/usr/bin/env python3
"""Generate training data from positions where model fails against heuristic.

This script plays games between a neural network model and heuristic AI,
then extracts positions where:
1. The model chose a losing/suboptimal move
2. The heuristic would have chosen correctly/better

The resulting dataset uses the heuristic's move as the training target,
allowing targeted fine-tuning to improve anti-heuristic performance.

Usage:
    # Basic usage
    python scripts/generate_antiheuristic_data.py \
        --model models/sq8_3p_ba/checkpoint_epoch_20.pth \
        --board-type square8 --num-players 3 \
        --num-games 100 \
        --output data/training/sq8_3p_antiheuristic.npz

    # With custom heuristic difficulty
    python scripts/generate_antiheuristic_data.py \
        --model models/sq8_4p_ba/checkpoint_epoch_15.pth \
        --board-type square8 --num-players 4 \
        --num-games 200 \
        --heuristic-difficulty 8 \
        --output data/training/sq8_4p_antiheuristic.npz

The output NPZ can be used for value-head-only fine-tuning:
    python -m app.training.train \
        --data-path data/training/sq8_3p_antiheuristic.npz \
        --init-weights models/sq8_3p_ba/checkpoint_epoch_20.pth \
        --freeze-policy \
        --learning-rate 0.00001 \
        --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure app imports resolve
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class AntiHeuristicConfig:
    """Configuration for anti-heuristic data generation."""
    model_path: str
    board_type: str = "square8"
    num_players: int = 3
    num_games: int = 100
    output_path: str = ""
    # Gameplay settings
    heuristic_difficulty: int = 8
    model_difficulty: int = 8
    max_moves_per_game: int = 500
    # Sample extraction settings
    min_value_difference: float = 0.1  # Min value difference to consider "mistake"
    sample_all_model_moves: bool = False  # Sample all moves or just mistakes
    include_heuristic_moves: bool = True  # Also sample heuristic's good moves
    # Random seed
    seed: int = 0


@dataclass
class ExtractedSample:
    """A training sample extracted from anti-heuristic play."""
    game_id: str
    move_number: int
    player: int
    features: np.ndarray  # Board features
    global_features: np.ndarray  # Global game features
    heuristic_move_idx: int  # Index of heuristic's preferred move
    model_move_idx: int  # Index of model's chosen move
    heuristic_value: float  # Heuristic's value estimate
    model_value: float  # Model's value estimate
    game_outcome: float  # Final outcome from this player's perspective
    is_mistake: bool  # True if model chose worse than heuristic


def load_model_and_ai(
    model_path: str,
    board_type: str,
    num_players: int,
    difficulty: int = 8,
):
    """Load neural network model and create AI player."""
    from app.ai.factory import AIFactory
    from app.models import AIConfig, AIType, BoardType

    board_type_enum = BoardType(board_type)

    ai_config = AIConfig(
        board_type=board_type_enum,
        num_players=num_players,
        difficulty=difficulty,
        use_neural_net=True,
        nn_model_id=model_path,
    )

    ai = AIFactory.create(
        AIType.GUMBEL_MCTS,
        player_number=1,
        config=ai_config,
    )

    return ai


def create_heuristic_ai(
    board_type: str,
    num_players: int,
    player_number: int,
    difficulty: int = 8,
):
    """Create a heuristic AI player."""
    from app.ai.factory import AIFactory
    from app.models import AIConfig, AIType, BoardType

    board_type_enum = BoardType(board_type)

    ai_config = AIConfig(
        board_type=board_type_enum,
        num_players=num_players,
        difficulty=difficulty,
        use_neural_net=False,
    )

    return AIFactory.create(
        AIType.HEURISTIC,
        player_number=player_number,
        config=ai_config,
    )


def extract_features(state, current_player: int, board_type: str):
    """Extract neural network features from game state."""
    from app.ai.neural_net.hex_neural_net_v2 import HexNeuralNet_v2
    from app.models import BoardType

    board_type_enum = BoardType(board_type)

    # Get feature extraction from model
    try:
        if hasattr(state, "encode_features"):
            features, global_features = state.encode_features(current_player)
        else:
            # Manual feature extraction
            from app.ai.neural_net.feature_encoding_v2 import encode_state_v2
            features, global_features = encode_state_v2(state, current_player)
        return features, global_features
    except Exception as e:
        logger.debug(f"Feature extraction failed: {e}")
        return None, None


def get_move_index(move, valid_moves: list) -> int:
    """Get the index of a move in the valid moves list."""
    for i, vm in enumerate(valid_moves):
        if move == vm or (hasattr(move, 'to') and hasattr(vm, 'to') and
                          move.to == vm.to and move.type == vm.type):
            return i
    return -1


def play_antiheuristic_game(
    config: AntiHeuristicConfig,
    model_ai,
    heuristic_ais: dict,
    game_idx: int,
) -> list[ExtractedSample]:
    """Play a game between model and heuristic AIs, extract learning samples.

    Returns list of ExtractedSample objects.
    """
    from app.game_engine import GameEngine
    from app.models import BoardType, GameStatus
    from app.training.initial_state import create_initial_state

    game_id = str(uuid.uuid4())
    board_type = BoardType(config.board_type)
    state = create_initial_state(board_type, config.num_players)

    # Randomly assign model to one player, heuristic to others
    model_player = random.randint(1, config.num_players)
    model_ai.player_number = model_player

    samples = []
    move_states = []  # Store (move_number, player, state, move, features)

    move_count = 0
    while state.game_status != GameStatus.COMPLETED and move_count < config.max_moves_per_game:
        current_player = state.current_player
        valid_moves = GameEngine.get_valid_moves(state, current_player)

        if not valid_moves:
            break

        # Extract features for this state
        features, global_features = extract_features(state, current_player, config.board_type)

        if current_player == model_player:
            # Model's turn
            model_ai.player_number = current_player
            model_move = model_ai.select_move(state)

            if model_move is None:
                model_move = random.choice(valid_moves)

            # Also get heuristic's preferred move for comparison
            heuristic_ai = create_heuristic_ai(
                config.board_type, config.num_players, current_player, config.heuristic_difficulty
            )
            heuristic_move = heuristic_ai.select_move(state)

            # Get value estimates if available
            model_value = getattr(model_ai, '_last_root_value', 0.5)
            heuristic_value = getattr(heuristic_ai, '_last_value', 0.5)

            move_states.append({
                'move_number': move_count,
                'player': current_player,
                'state': state,
                'model_move': model_move,
                'heuristic_move': heuristic_move,
                'features': features,
                'global_features': global_features,
                'model_value': model_value,
                'heuristic_value': heuristic_value,
                'valid_moves': valid_moves,
                'is_model_turn': True,
            })

            chosen_move = model_move

        else:
            # Heuristic's turn
            if current_player not in heuristic_ais:
                heuristic_ais[current_player] = create_heuristic_ai(
                    config.board_type, config.num_players, current_player, config.heuristic_difficulty
                )

            heuristic_ai = heuristic_ais[current_player]
            heuristic_move = heuristic_ai.select_move(state)

            if heuristic_move is None:
                heuristic_move = random.choice(valid_moves)

            if config.include_heuristic_moves and features is not None:
                move_states.append({
                    'move_number': move_count,
                    'player': current_player,
                    'state': state,
                    'model_move': None,
                    'heuristic_move': heuristic_move,
                    'features': features,
                    'global_features': global_features,
                    'model_value': None,
                    'heuristic_value': 0.5,
                    'valid_moves': valid_moves,
                    'is_model_turn': False,
                })

            chosen_move = heuristic_move

        state = GameEngine.apply_move(state, chosen_move)
        move_count += 1

    # Determine game outcome
    winner = state.winner if state.game_status == GameStatus.COMPLETED else None

    # Convert move states to training samples
    for ms in move_states:
        if ms['features'] is None:
            continue

        # Calculate outcome from this player's perspective
        if winner is None:
            outcome = 0.5  # Draw or incomplete
        elif winner == ms['player']:
            outcome = 1.0  # Win
        else:
            outcome = 0.0  # Loss

        # Determine if this was a "mistake" by the model
        is_mistake = False
        if ms['is_model_turn'] and ms['model_move'] is not None and ms['heuristic_move'] is not None:
            model_idx = get_move_index(ms['model_move'], ms['valid_moves'])
            heuristic_idx = get_move_index(ms['heuristic_move'], ms['valid_moves'])

            # Consider it a mistake if:
            # 1. Model lost the game
            # 2. Model chose different move than heuristic
            if winner is not None and winner != ms['player'] and model_idx != heuristic_idx:
                is_mistake = True

            # Or if value estimates differ significantly
            if (ms['model_value'] is not None and ms['heuristic_value'] is not None and
                    ms['model_value'] - ms['heuristic_value'] < -config.min_value_difference):
                is_mistake = True

        # Only keep samples where we should learn from heuristic
        if config.sample_all_model_moves or is_mistake or not ms['is_model_turn']:
            heuristic_idx = get_move_index(ms['heuristic_move'], ms['valid_moves'])
            model_idx = get_move_index(ms['model_move'], ms['valid_moves']) if ms['model_move'] else -1

            if heuristic_idx >= 0:
                samples.append(ExtractedSample(
                    game_id=game_id,
                    move_number=ms['move_number'],
                    player=ms['player'],
                    features=ms['features'],
                    global_features=ms['global_features'],
                    heuristic_move_idx=heuristic_idx,
                    model_move_idx=model_idx,
                    heuristic_value=ms['heuristic_value'] if ms['heuristic_value'] else 0.5,
                    model_value=ms['model_value'] if ms['model_value'] else 0.5,
                    game_outcome=outcome,
                    is_mistake=is_mistake,
                ))

    return samples


def samples_to_npz(samples: list[ExtractedSample], output_path: str, config: AntiHeuristicConfig):
    """Convert extracted samples to NPZ training format."""
    if not samples:
        logger.warning("No samples to save!")
        return

    logger.info(f"Converting {len(samples)} samples to NPZ format...")

    # Aggregate arrays
    features_list = []
    globals_list = []
    values_list = []
    policy_indices_list = []
    policy_values_list = []
    move_numbers_list = []

    for sample in samples:
        if sample.features is None:
            continue

        features_list.append(sample.features)
        globals_list.append(sample.global_features if sample.global_features is not None else np.zeros(20))

        # Value target: use game outcome (more reliable than estimates)
        values_list.append(sample.game_outcome)

        # Policy target: heuristic's preferred move
        policy_indices_list.append(sample.heuristic_move_idx)
        policy_values_list.append(1.0)  # Full confidence in heuristic's move

        move_numbers_list.append(sample.move_number)

    # Stack into arrays
    features = np.stack(features_list, axis=0)
    globals_arr = np.stack(globals_list, axis=0)
    values = np.array(values_list, dtype=np.float32)
    policy_indices = np.array(policy_indices_list, dtype=np.int32)
    policy_values = np.array(policy_values_list, dtype=np.float32)
    move_numbers = np.array(move_numbers_list, dtype=np.int32)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    np.savez_compressed(
        output_path,
        features=features,
        globals=globals_arr,
        values=values,
        policy_indices=policy_indices,
        policy_values=policy_values,
        move_numbers=move_numbers,
        # Metadata
        board_type=config.board_type,
        num_players=config.num_players,
        source="antiheuristic",
    )

    logger.info(f"Saved {len(features)} samples to {output_path}")
    logger.info(f"  Features shape: {features.shape}")
    logger.info(f"  Values range: [{values.min():.2f}, {values.max():.2f}]")


def run_antiheuristic_generation(config: AntiHeuristicConfig) -> dict[str, Any]:
    """Run anti-heuristic data generation.

    Returns statistics about the run.
    """
    start_time = time.time()

    # Set random seed
    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)

    logger.info("=" * 60)
    logger.info("Anti-Heuristic Data Generation")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"Games: {config.num_games}")
    logger.info(f"Heuristic difficulty: {config.heuristic_difficulty}")
    logger.info("=" * 60)

    # Load model AI
    try:
        model_ai = load_model_and_ai(
            config.model_path,
            config.board_type,
            config.num_players,
            config.model_difficulty,
        )
        logger.info("Model AI loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fall back to heuristic for testing
        logger.info("Falling back to heuristic AI for model (testing mode)")
        model_ai = create_heuristic_ai(
            config.board_type,
            config.num_players,
            1,
            config.model_difficulty,
        )

    # Create heuristic AIs for other players
    heuristic_ais = {}

    all_samples = []
    games_completed = 0
    games_failed = 0
    model_wins = 0
    heuristic_wins = 0
    mistakes_found = 0

    for game_idx in range(config.num_games):
        try:
            samples = play_antiheuristic_game(config, model_ai, heuristic_ais, game_idx)
            all_samples.extend(samples)
            games_completed += 1

            # Count outcomes
            if samples:
                # Check if model won (first sample's outcome for model player)
                model_samples = [s for s in samples if s.is_mistake is not None]
                if model_samples:
                    if model_samples[0].game_outcome > 0.5:
                        model_wins += 1
                    elif model_samples[0].game_outcome < 0.5:
                        heuristic_wins += 1

                mistakes_found += sum(1 for s in samples if s.is_mistake)

            if (game_idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {game_idx + 1}/{config.num_games} games, "
                    f"{len(all_samples)} samples, {mistakes_found} mistakes"
                )

        except Exception as e:
            logger.debug(f"Game {game_idx} failed: {e}")
            games_failed += 1
            continue

    # Save to NPZ
    if config.output_path and all_samples:
        samples_to_npz(all_samples, config.output_path, config)

    duration = time.time() - start_time

    stats = {
        "games_completed": games_completed,
        "games_failed": games_failed,
        "total_samples": len(all_samples),
        "mistakes_found": mistakes_found,
        "model_wins": model_wins,
        "heuristic_wins": heuristic_wins,
        "model_win_rate": model_wins / max(1, games_completed),
        "duration_seconds": duration,
    }

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("ANTI-HEURISTIC GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games: {games_completed} completed, {games_failed} failed")
    logger.info(f"Model wins: {model_wins} ({100*stats['model_win_rate']:.1f}%)")
    logger.info(f"Heuristic wins: {heuristic_wins}")
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Mistakes extracted: {mistakes_found}")
    logger.info(f"Duration: {duration:.1f}s")
    if config.output_path:
        logger.info(f"Output: {config.output_path}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from model mistakes against heuristic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to neural network model checkpoint",
    )
    parser.add_argument(
        "--board-type", "-b",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Number of players (default: 3)",
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output NPZ file path",
    )
    parser.add_argument(
        "--heuristic-difficulty",
        type=int,
        default=8,
        help="Heuristic AI difficulty level (default: 8)",
    )
    parser.add_argument(
        "--sample-all-moves",
        action="store_true",
        help="Sample all model moves, not just mistakes",
    )
    parser.add_argument(
        "--min-value-diff",
        type=float,
        default=0.1,
        help="Minimum value difference to consider a mistake (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (0 = random)",
    )

    args = parser.parse_args()

    # Set default output path
    output_path = args.output
    if not output_path:
        output_path = f"data/training/{args.board_type}_{args.num_players}p_antiheuristic.npz"

    config = AntiHeuristicConfig(
        model_path=args.model,
        board_type=args.board_type,
        num_players=args.num_players,
        num_games=args.num_games,
        output_path=output_path,
        heuristic_difficulty=args.heuristic_difficulty,
        sample_all_model_moves=args.sample_all_moves,
        min_value_difference=args.min_value_diff,
        seed=args.seed,
    )

    run_antiheuristic_generation(config)


if __name__ == "__main__":
    main()
