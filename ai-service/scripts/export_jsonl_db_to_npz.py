#!/usr/bin/env python3
"""Export training data from JSONL-converted database format.

January 2026: Created for databases with games stored in metadata_json format.

Usage:
    python scripts/export_jsonl_db_to_npz.py \
        --db data/games/jsonl_converted_square8_2p.db \
        --board-type square8 --num-players 2 \
        --output data/training/square8_2p.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.game_engine import GameEngine
from app.models import GameState, Move, BoardType
from app.training.encoding import HexStateEncoder, SquareStateEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_state_encoder(board_type: str):
    """Get the appropriate encoder for the board type."""
    if board_type in ("hex8", "hexagonal"):
        board_size = 9 if board_type == "hex8" else 25
        return HexStateEncoder(board_size=board_size)
    else:
        board_size = 8 if board_type == "square8" else 19
        return SquareStateEncoder(board_type=board_type, board_size=board_size)


def encode_state_features(game_state: GameState, board_type: str, num_players: int) -> np.ndarray:
    """Encode game state to feature array for training.

    Args:
        game_state: The game state to encode
        board_type: Board type string (square8, hex8, etc.)
        num_players: Number of players

    Returns:
        Feature array suitable for neural network input
    """
    encoder = get_state_encoder(board_type)
    features, _ = encoder.encode_state(game_state)
    return features


def parse_move(move_dict: dict[str, Any], board_type: str) -> Move | None:
    """Parse a move from JSON dict to Move object."""
    try:
        move_type = move_dict.get("type")
        player = move_dict.get("player", 1)

        to_data = move_dict.get("to", {})
        from_data = move_dict.get("from")

        # Extract coordinates
        to_pos = (to_data.get("x", 0), to_data.get("y", 0)) if to_data else None
        from_pos = (from_data.get("x", 0), from_data.get("y", 0)) if from_data else None

        # Create move based on type
        if move_type == "place_ring":
            return Move(type="place_ring", to=to_pos, player=player)
        elif move_type == "move_stack":
            return Move(type="move_stack", from_pos=from_pos, to=to_pos, player=player)
        elif move_type == "no_line_action":
            return Move(type="no_line_action", to=to_pos, player=player)
        elif move_type == "choose_line":
            # Handle line choices
            line = move_dict.get("line", [])
            return Move(type="choose_line", line=line, player=player)
        elif move_type == "collect_ring":
            return Move(type="collect_ring", to=to_pos, player=player)
        else:
            logger.debug(f"Unknown move type: {move_type}")
            return None
    except Exception as e:
        logger.debug(f"Failed to parse move: {e}")
        return None


def replay_game(
    initial_state_dict: dict[str, Any],
    moves: list[dict[str, Any]],
    board_type: str,
    num_players: int,
) -> list[tuple[GameState, Move]]:
    """Replay a game and collect (state, move) pairs.

    Returns:
        List of (state_before, move) tuples for training samples
    """
    samples = []

    try:
        # Create initial state
        bt = BoardType(board_type)
        state = GameEngine.create_initial_state(bt, num_players)

        # Replay moves
        for move_dict in moves:
            # Parse the move
            move = parse_move(move_dict, board_type)
            if move is None:
                continue

            # Collect sample (state before move)
            samples.append((state, move))

            # Apply move
            try:
                state = GameEngine.apply_move(state, move)
            except Exception as e:
                logger.debug(f"Failed to apply move: {e}")
                break

    except Exception as e:
        logger.debug(f"Failed to replay game: {e}")

    return samples


def create_policy_target(move: Move, state: GameState, num_cells: int) -> np.ndarray:
    """Create a policy target (one-hot) for the move."""
    # Create policy vector
    policy = np.zeros(num_cells, dtype=np.float32)

    if move.to is not None:
        # Simple encoding: use linear index
        x, y = move.to
        if hasattr(state.board, 'width'):
            width = state.board.width
            idx = y * width + x
            if 0 <= idx < num_cells:
                policy[idx] = 1.0

    return policy


def export_database(
    db_path: str,
    board_type: str,
    num_players: int,
    output_path: str,
    max_games: int | None = None,
    sample_every: int = 1,
) -> int:
    """Export training data from JSONL-converted database.

    Args:
        db_path: Path to database
        board_type: Board type (square8, hex8, etc.)
        num_players: Number of players
        output_path: Output NPZ path
        max_games: Maximum games to process
        sample_every: Sample every N moves (1 = all moves)

    Returns:
        Number of samples exported
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get game count
    cursor.execute("SELECT COUNT(*) FROM games WHERE board_type = ? AND num_players = ?",
                   (board_type, num_players))
    total_games = cursor.fetchone()[0]
    logger.info(f"Found {total_games} games for {board_type}_{num_players}p")

    if max_games:
        total_games = min(total_games, max_games)

    # Determine board size for policy encoding
    bt = BoardType(board_type)
    state = GameEngine.create_initial_state(bt, num_players)
    num_cells = len(state.board.cells) if hasattr(state.board, 'cells') else 64

    # For square8, it's 64 cells
    if board_type == "square8":
        num_cells = 64
        board_width = 8
    elif board_type == "square19":
        num_cells = 361
        board_width = 19
    elif board_type == "hex8":
        num_cells = 61
        board_width = 9  # hex with radius 4
    else:
        num_cells = 469  # hexagonal
        board_width = 25

    # Collect samples
    all_boards = []
    all_policies = []
    all_values = []

    # Process in batches
    batch_size = 1000
    processed = 0
    samples_collected = 0

    cursor.execute("""
        SELECT metadata_json, winner FROM games
        WHERE board_type = ? AND num_players = ? AND game_status = 'completed'
        LIMIT ?
    """, (board_type, num_players, total_games))

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        for metadata_json, winner in rows:
            try:
                meta = json.loads(metadata_json)
                moves = meta.get("moves", [])
                initial_state = meta.get("initial_state", {})

                # Replay game
                samples = replay_game(initial_state, moves, board_type, num_players)

                # Collect training samples
                for i, (game_state, move) in enumerate(samples):
                    if i % sample_every != 0:
                        continue

                    # Encode state (simple encoding)
                    board_features = encode_state_features(game_state, board_type, num_players)

                    # Create policy target
                    policy = create_policy_target(move, game_state, num_cells)

                    # Create value target (based on winner)
                    current_player = game_state.current_player
                    if winner == current_player:
                        value = 1.0
                    elif winner == 0:  # draw
                        value = 0.0
                    else:
                        value = -1.0

                    # For multiplayer, create value vector
                    if num_players > 2:
                        value_vec = np.zeros(num_players, dtype=np.float32)
                        if winner > 0:
                            value_vec[winner - 1] = 1.0
                    else:
                        value_vec = np.array([value], dtype=np.float32)

                    all_boards.append(board_features)
                    all_policies.append(policy)
                    all_values.append(value_vec)
                    samples_collected += 1

            except Exception as e:
                logger.debug(f"Error processing game: {e}")
                continue

        processed += len(rows)
        if processed % 10000 == 0:
            logger.info(f"Processed {processed}/{total_games} games, {samples_collected} samples collected")

    conn.close()

    if not all_boards:
        logger.error("No samples collected!")
        return 0

    # Convert to numpy arrays
    boards = np.array(all_boards, dtype=np.float32)
    policies = np.array(all_policies, dtype=np.float32)
    values = np.array(all_values, dtype=np.float32)

    logger.info(f"Saving {len(boards)} samples to {output_path}")
    logger.info(f"  Boards shape: {boards.shape}")
    logger.info(f"  Policies shape: {policies.shape}")
    logger.info(f"  Values shape: {values.shape}")

    # Save to NPZ
    np.savez_compressed(
        output_path,
        boards=boards,
        policies=policies,
        values=values,
        board_type=board_type,
        num_players=num_players,
    )

    return len(boards)


def main():
    parser = argparse.ArgumentParser(description="Export training data from JSONL-converted database")
    parser.add_argument("--db", required=True, help="Path to database")
    parser.add_argument("--board-type", required=True, choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--num-players", required=True, type=int, choices=[2, 3, 4])
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--max-games", type=int, help="Maximum games to process")
    parser.add_argument("--sample-every", type=int, default=1, help="Sample every N moves")

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    count = export_database(
        db_path=args.db,
        board_type=args.board_type,
        num_players=args.num_players,
        output_path=args.output,
        max_games=args.max_games,
        sample_every=args.sample_every,
    )

    if count > 0:
        logger.info(f"Successfully exported {count} samples to {args.output}")
    else:
        logger.error("Export failed - no samples collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
