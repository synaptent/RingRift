#!/usr/bin/env python3
"""Export training data from converted DBs that store games in metadata_json.

This handles the case where JSONL was converted to DB but without game_moves table.
The full game data (initial_state, moves) is stored in metadata_json field.

Usage:
    python scripts/export_from_metadata_json.py \
        --db data/games/jsonl_converted_hexagonal_2p.db \
        --output data/training/hex_2p_from_json.npz \
        --board-type hexagonal \
        --num-players 2
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.game_engine import GameEngine
from app.models.core import BoardType, GameState, Move
from app.training.encoding import P_HEX, POLICY_SIZE_HEX8, HexStateEncoderV3
from app.training.generate_data import create_initial_state


class HexEncoderWrapper:
    """Wrapper for hex encoders with consistent interface."""

    def __init__(self, encoder, board_size: int = 25):
        self._encoder = encoder
        self._board_size = board_size
        self.policy_size = encoder.policy_size

    def encode_state(self, state):
        """Returns (board_features, global_features)."""
        return self._encoder.encode_state(state)

    def encode_move(self, move, state):
        """Encode a move to policy index."""
        return self._encoder.encode_move(move, state.board)


def parse_move(move_dict: dict) -> Move:
    """Parse a move from JSON dict.

    Handles two formats:
    1. GPU selfplay: {"move_type": "PLACEMENT", "player": 1, "from_pos": [x,y], "to_pos": [x,y]}
    2. Standard: {"type": "place_ring", "player": 1, "to": {"x": x, "y": y}}
    """
    from app.models.core import Position

    to_pos = None
    from_pos = None

    # Handle GPU selfplay format: positions as [x, y] arrays
    if "to_pos" in move_dict:
        to_data = move_dict["to_pos"]
        if isinstance(to_data, (list, tuple)):
            to_pos = Position(x=to_data[0], y=to_data[1])
        elif isinstance(to_data, dict):
            to_pos = Position(x=to_data["x"], y=to_data["y"], z=to_data.get("z"))

    if "from_pos" in move_dict:
        from_data = move_dict["from_pos"]
        if isinstance(from_data, (list, tuple)):
            from_pos = Position(x=from_data[0], y=from_data[1])
        elif isinstance(from_data, dict):
            from_pos = Position(x=from_data["x"], y=from_data["y"], z=from_data.get("z"))

    # Handle standard format
    if "to" in move_dict and to_pos is None:
        to_data = move_dict["to"]
        if isinstance(to_data, dict):
            to_pos = Position(x=to_data["x"], y=to_data["y"], z=to_data.get("z"))

    if "from" in move_dict and from_pos is None:
        from_data = move_dict["from"]
        if isinstance(from_data, dict):
            from_pos = Position(x=from_data["x"], y=from_data["y"], z=from_data.get("z"))

    # Parse move type - handle both formats
    move_type_raw = move_dict.get("move_type") or move_dict.get("type", "place_ring")

    # Map GPU selfplay format to standard types
    move_type_map = {
        "PLACEMENT": "place_ring",
        "MOVEMENT": "move_ring",
        "CAPTURE": "capture",
        "RECOVERY_SLIDE": "recovery_slide",
    }
    move_type = move_type_map.get(move_type_raw, move_type_raw.lower() if isinstance(move_type_raw, str) else "place_ring")

    return Move(
        id=move_dict.get("id", ""),
        type=move_type,
        player=move_dict.get("player", 1),
        to=to_pos,
        from_pos=from_pos,
    )


def get_encoder(board_type: BoardType):
    """Get the appropriate encoder for the board type."""
    if board_type == BoardType.HEX8:
        hex_encoder = HexStateEncoderV3(board_size=9, policy_size=POLICY_SIZE_HEX8)
        return HexEncoderWrapper(hex_encoder, board_size=9)
    elif board_type == BoardType.HEXAGONAL:
        hex_encoder = HexStateEncoderV3(board_size=25, policy_size=P_HEX)
        return HexEncoderWrapper(hex_encoder, board_size=25)
    else:
        raise ValueError(f"Unsupported board type: {board_type}. Use db_to_training_npz.py for square boards.")


def export_from_metadata_json(
    db_path: str,
    output_path: str,
    board_type: BoardType,
    num_players: int = 2,
    limit: int | None = None,
):
    """Export training samples from metadata_json field."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Query games
    query = "SELECT game_id, metadata_json FROM games WHERE game_status = 'completed'"
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)

    # Initialize encoder
    encoder = get_encoder(board_type)

    all_board_features = []
    all_global_features = []
    all_policies = []
    all_values = []

    games_processed = 0
    samples_total = 0

    for row in cursor:
        try:
            game_data = json.loads(row["metadata_json"])

            moves_data = game_data.get("moves", [])
            winner = game_data.get("winner")

            if not moves_data or winner is None:
                continue

            # Create initial state
            state = create_initial_state(board_type=board_type, num_players=num_players)

            # Replay game and collect samples
            for _move_idx, move_data in enumerate(moves_data):
                try:
                    current_player = state.current_player

                    # Encode state - returns (board_features, global_features)
                    board_feat, global_feat = encoder.encode_state(state)

                    # Parse and encode move
                    move = parse_move(move_data)

                    # Get valid moves for policy
                    valid_moves = GameEngine.get_valid_moves(state, current_player)
                    if not valid_moves:
                        # Try bookkeeping
                        req = GameEngine.get_phase_requirement(state, current_player)
                        if req:
                            bm = GameEngine.synthesize_bookkeeping_move(req, state)
                            if bm:
                                state = GameEngine.apply_move(state, bm)
                                continue
                        break

                    # Create policy (one-hot for the played move)
                    policy = np.zeros(encoder.policy_size, dtype=np.float32)
                    try:
                        move_idx_encoded = encoder.encode_move(move, state)
                        if 0 <= move_idx_encoded < encoder.policy_size:
                            policy[move_idx_encoded] = 1.0
                    except Exception:
                        pass

                    # Value target
                    if winner == current_player:
                        value = 1.0
                    elif winner == 0:
                        value = 0.0
                    else:
                        value = -1.0

                    all_board_features.append(board_feat)
                    all_global_features.append(global_feat)
                    all_policies.append(policy)
                    all_values.append(value)
                    samples_total += 1

                    # Apply move
                    state = GameEngine.apply_move(state, move)

                except Exception:
                    # Skip problematic moves
                    continue

            games_processed += 1
            if games_processed % 100 == 0:
                print(f"  Processed {games_processed} games, {samples_total} samples...")

        except Exception:
            continue

    conn.close()

    if not all_board_features:
        print("No samples extracted!")
        return

    # Stack arrays
    board_features = np.stack(all_board_features)
    global_features = np.stack(all_global_features)
    policies = np.stack(all_policies)
    values = np.array(all_values, dtype=np.float32)

    # Save
    np.savez(
        output_path,
        states=board_features,  # Board spatial features
        globals=global_features,  # Global features
        policies=policies,
        values=values,
        board_type=str(board_type.value),
        num_players=num_players,
        feature_version="v3",
    )

    print(f"\nExported {samples_total} samples from {games_processed} games")
    print(f"States shape: {board_features.shape}")
    print(f"Globals shape: {global_features.shape}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export from metadata_json DBs")
    parser.add_argument("--db", required=True, help="Input database path")
    parser.add_argument("--output", "-o", required=True, help="Output NPZ path")
    parser.add_argument("--board-type", default="hexagonal",
                       choices=["square8", "square19", "hexagonal", "hex8"])
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Limit number of games")

    args = parser.parse_args()

    board_type = BoardType(args.board_type)

    export_from_metadata_json(
        db_path=args.db,
        output_path=args.output,
        board_type=board_type,
        num_players=args.num_players,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
