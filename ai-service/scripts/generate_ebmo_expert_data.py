#!/usr/bin/env python
"""Generate EBMO training data from expert AI games.

This script generates training data for EBMO by recording games between
strong AI opponents (HeuristicAI, MinimaxAI). EBMO learns from expert
moves rather than just self-play.

Key insight: EBMO needs to see what good moves look like, not just
learn from its own (potentially poor) decisions.

Usage:
    # Generate 100 games of HeuristicAI vs HeuristicAI
    python scripts/generate_ebmo_expert_data.py --num-games 100

    # Generate using MinimaxAI
    python scripts/generate_ebmo_expert_data.py --engine minimax --depth 3

    # Mix of engines
    python scripts/generate_ebmo_expert_data.py --engine mixed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Set up path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import (
    GameState, BoardType, GamePhase, GameStatus,
    Player, TimeControl, BoardState, AIConfig, Move
)
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.ai.random_ai import RandomAI
from app.rules.default_engine import DefaultRulesEngine
from app.training.ebmo_dataset import ActionFeatureGenerator
from app.ai.neural_net import NeuralNetAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_ebmo_expert_data")


def create_game_state(board_type=BoardType.SQUARE8) -> GameState:
    """Create a fresh game state."""
    size = 8
    board = BoardState(
        type=board_type, size=size, stacks={}, markers={},
        collapsedSpaces={}, eliminatedRings={}
    )

    players = [
        Player(
            id='player1', username='P1', type='ai', playerNumber=1, isReady=True,
            timeRemaining=600000, aiDifficulty=5, ringsInHand=20,
            eliminatedRings=0, territorySpaces=0
        ),
        Player(
            id='player2', username='P2', type='ai', playerNumber=2, isReady=True,
            timeRemaining=600000, aiDifficulty=5, ringsInHand=20,
            eliminatedRings=0, territorySpaces=0
        ),
    ]

    return GameState(
        id=str(uuid.uuid4()), boardType=board_type, board=board, players=players,
        currentPhase=GamePhase.RING_PLACEMENT, currentPlayer=1, moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type='standard'),
        gameStatus=GameStatus.ACTIVE, createdAt=datetime.now(), lastMoveAt=datetime.now(),
        isRated=False, maxPlayers=2, totalRingsInPlay=0, totalRingsEliminated=0,
        victoryThreshold=3, territoryVictoryThreshold=10, chainCaptureState=None,
        mustMoveFromStackKey=None, zobristHash=None
    )


def create_ai(engine: str, player_num: int, seed: int, depth: int = 3):
    """Create an AI instance based on engine type."""
    config = AIConfig(difficulty=depth, rng_seed=seed)

    if engine == "heuristic":
        return HeuristicAI(player_num, config)
    elif engine == "minimax":
        return MinimaxAI(player_num, config)
    elif engine == "random":
        return RandomAI(player_num, config)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def extract_board_features(game_state: GameState, player_num: int, nn: NeuralNetAI) -> np.ndarray:
    """Extract board features for EBMO training.

    Uses NeuralNetAI feature extraction.
    """
    # Use the NeuralNetAI's feature extraction
    features, _ = nn._extract_features(game_state)
    return features  # Shape: (14, 8, 8)


def extract_global_features(game_state: GameState, player_num: int) -> np.ndarray:
    """Extract global features for EBMO training."""
    players = game_state.players
    me = next((p for p in players if p.player_number == player_num), None)
    opp = next((p for p in players if p.player_number != player_num), None)

    if me is None or opp is None:
        return np.zeros(20, dtype=np.float32)

    features = np.zeros(20, dtype=np.float32)

    # Player stats (normalized)
    features[0] = me.rings_in_hand / 20.0
    features[1] = me.eliminated_rings / 10.0
    features[2] = me.territory_spaces / 64.0
    features[3] = opp.rings_in_hand / 20.0
    features[4] = opp.eliminated_rings / 10.0
    features[5] = opp.territory_spaces / 64.0

    # Game phase (one-hot)
    phase_idx = {"RING_PLACEMENT": 0, "MOVEMENT": 1, "CHAIN_CAPTURE": 2}.get(
        game_state.current_phase.value if hasattr(game_state.current_phase, 'value')
        else str(game_state.current_phase), 1
    )
    features[6 + phase_idx] = 1.0

    # Turn info
    features[9] = game_state.turn_number / 100.0 if hasattr(game_state, 'turn_number') else 0.0
    features[10] = 1.0 if game_state.current_player == player_num else 0.0

    # Board occupancy
    num_stacks = len(game_state.board.stacks)
    num_markers = len(game_state.board.markers)
    features[11] = num_stacks / 64.0
    features[12] = num_markers / 64.0

    return features


def extract_action_features(move: Move, board_size: int = 8) -> np.ndarray:
    """Extract action features from a move."""
    generator = ActionFeatureGenerator(board_size=board_size)

    # Get positions
    from_x = move.from_pos.x if move.from_pos else 0
    from_y = move.from_pos.y if move.from_pos else 0
    to_x = move.to.x if move.to else from_x
    to_y = move.to.y if move.to else from_y

    # Get move type index
    move_type_map = {
        "PLACE_RING": 0, "MOVE_STACK": 1, "PLACE_MARKER": 2,
        "OVERTAKE": 3, "CHAIN_CAPTURE": 4, "SWAP": 5, "PASS": 6, "RESIGN": 7
    }
    move_type_str = move.type.value if hasattr(move.type, 'value') else str(move.type)
    move_type = move_type_map.get(move_type_str.upper(), 0)

    return generator.generate_from_positions(from_x, from_y, to_x, to_y, move_type)


def play_game(
    p1_engine: str,
    p2_engine: str,
    game_idx: int,
    depth: int = 3,
    max_moves: int = 500,
    nn: Optional[NeuralNetAI] = None,
) -> List[Dict[str, Any]]:
    """Play a game and collect training samples.

    Returns list of samples: {board, globals, action, player, outcome}
    """
    rules_engine = DefaultRulesEngine()

    p1_ai = create_ai(p1_engine, 1, game_idx, depth)
    p2_ai = create_ai(p2_engine, 2, game_idx + 10000, depth)

    # Create neural net for feature extraction if not provided
    if nn is None:
        nn = NeuralNetAI(1, AIConfig(difficulty=5))

    game_state = create_game_state()
    samples = []
    move_count = 0

    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = game_state.current_player
        current_ai = p1_ai if current_player == 1 else p2_ai
        current_ai.player_number = current_player

        try:
            # Extract features BEFORE the move
            board_feat = extract_board_features(game_state, current_player, nn)
            global_feat = extract_global_features(game_state, current_player)

            # Get the move
            move = current_ai.select_move(game_state)
            if not move:
                break

            # Extract action features
            action_feat = extract_action_features(move)

            # Store sample (outcome will be filled after game ends)
            samples.append({
                'board': board_feat,
                'globals': global_feat,
                'action': action_feat,
                'player': current_player,
                'move_num': move_count,
            })

            # Apply move
            game_state = rules_engine.apply_move(game_state, move)
            move_count += 1

        except Exception as e:
            logger.warning(f"Error in game {game_idx}: {e}")
            break

    # Determine outcome and assign to samples
    winner = game_state.winner
    for sample in samples:
        if winner is None:
            sample['outcome'] = 0.0  # Draw
        elif sample['player'] == winner:
            sample['outcome'] = 1.0  # Win
        else:
            sample['outcome'] = -1.0  # Loss

    return samples


def generate_expert_data(
    num_games: int,
    engine: str,
    depth: int,
    output_file: str,
    board_size: int = 8,
) -> None:
    """Generate expert training data."""
    logger.info(f"Generating {num_games} games using {engine} engine (depth={depth})")

    # Create shared neural net for feature extraction
    nn = NeuralNetAI(1, AIConfig(difficulty=5))

    all_boards = []
    all_globals = []
    all_actions = []
    all_outcomes = []

    for game_idx in range(num_games):
        # Determine engines for this game
        if engine == "mixed":
            engines = ["heuristic", "minimax"]
            p1_engine = engines[game_idx % 2]
            p2_engine = engines[(game_idx + 1) % 2]
        else:
            p1_engine = engine
            p2_engine = engine

        samples = play_game(p1_engine, p2_engine, game_idx, depth, nn=nn)

        for sample in samples:
            all_boards.append(sample['board'])
            all_globals.append(sample['globals'])
            all_actions.append(sample['action'])
            all_outcomes.append(sample['outcome'])

        if (game_idx + 1) % 10 == 0:
            logger.info(f"  Completed {game_idx + 1}/{num_games} games, {len(all_boards)} samples")

    # Stack all data with history (for 56-channel input)
    boards_array = np.stack(all_boards, axis=0)  # (N, 14, 8, 8)

    # Stack 4 frames for history (EBMO expects 56 channels)
    # For now, just repeat the current frame 4 times
    boards_with_history = np.concatenate([boards_array] * 4, axis=1)  # (N, 56, 8, 8)

    globals_array = np.stack(all_globals, axis=0)  # (N, 20)
    actions_array = np.stack(all_actions, axis=0)  # (N, 14)
    outcomes_array = np.array(all_outcomes, dtype=np.float32)  # (N,)

    # Save as NPZ
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=boards_with_history,
        globals=globals_array,
        actions=actions_array,
        values=outcomes_array,
        # Metadata
        board_size=board_size,
        num_games=num_games,
        engine=engine,
    )

    logger.info(f"Saved {len(all_boards)} samples to {output_path}")
    logger.info(f"  boards: {boards_with_history.shape}")
    logger.info(f"  globals: {globals_array.shape}")
    logger.info(f"  actions: {actions_array.shape}")
    logger.info(f"  outcomes: {outcomes_array.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate EBMO expert training data")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to generate")
    parser.add_argument("--engine", type=str, default="heuristic",
                        choices=["heuristic", "minimax", "random", "mixed"],
                        help="AI engine to use for games")
    parser.add_argument("--depth", type=int, default=5,
                        help="Search depth for minimax/heuristic")
    parser.add_argument("--output", type=str, default="data/training/ebmo_expert.npz",
                        help="Output file path")
    parser.add_argument("--board-size", type=int, default=8,
                        help="Board size")

    args = parser.parse_args()

    generate_expert_data(
        num_games=args.num_games,
        engine=args.engine,
        depth=args.depth,
        output_file=args.output,
        board_size=args.board_size,
    )


if __name__ == "__main__":
    main()
