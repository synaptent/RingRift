#!/usr/bin/env python3
"""Convert Gumbel selfplay JSONL to NPZ format for EBMO training."""

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.ebmo_network import ActionFeatureExtractor
from app.models import BoardType, Move, Position


def dict_to_move(move_dict: dict) -> Move:
    """Convert a move dict from JSONL to a Move object."""
    # Handle position fields
    to_pos = None
    from_pos = None

    if move_dict.get("to"):
        to_data = move_dict["to"]
        to_pos = Position(x=to_data["x"], y=to_data["y"])

    if move_dict.get("from_pos"):
        from_data = move_dict["from_pos"]
        from_pos = Position(x=from_data["x"], y=from_data["y"])
    elif move_dict.get("from"):
        from_data = move_dict["from"]
        from_pos = Position(x=from_data["x"], y=from_data["y"])

    return Move(
        id=move_dict.get("id", str(uuid.uuid4())),
        type=move_dict.get("type", "place_ring"),
        to=to_pos,
        from_pos=from_pos,
        player=move_dict.get("player", 1),
    )


def load_jsonl(path: Path) -> list[dict]:
    """Load games from JSONL file."""
    games = []
    with open(path) as f:
        for line in f:
            if line.strip():
                games.append(json.loads(line))
    return games


def get_board_size(board_type: BoardType) -> int:
    """Get board size from board type."""
    if board_type == BoardType.SQUARE8:
        return 8
    elif board_type == BoardType.SQUARE19:
        return 19
    elif board_type == BoardType.HEXAGONAL:
        return 11  # Hex board radius
    else:
        return 8


def extract_samples(games: list[dict], board_type: BoardType = BoardType.SQUARE8) -> dict:
    """Extract training samples from games.

    Returns dict with:
        - states: (N, C, H, W) state features
        - actions: (N, action_dim) action features
        - outcomes: (N,) game outcomes (-1, 0, +1)
    """
    board_size = get_board_size(board_type)
    extractor = ActionFeatureExtractor(board_size)

    actions = []
    outcomes = []

    for game in games:
        winner = game.get("winner")
        if winner is None:
            continue  # Skip draws

        moves = game.get("moves", [])
        for i, move_data in enumerate(moves):
            player = move_data.get("player", (i % 2) + 1)

            # Outcome from this player's perspective
            if winner == player:
                outcome = 1.0
            elif winner == 0:
                outcome = 0.0
            else:
                outcome = -1.0

            # Extract action features - convert dict to Move object
            move_dict = move_data.get("move", move_data)  # Handle both nested and flat formats
            move = dict_to_move(move_dict)
            action_feat = extractor.extract_features(move)

            # State features would need full game replay - use placeholder
            # For now just store action features and outcomes
            actions.append(action_feat)
            outcomes.append(outcome)

    return {
        "actions": np.array(actions, dtype=np.float32),
        "outcomes": np.array(outcomes, dtype=np.float32),
        "num_samples": len(actions),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to NPZ")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output NPZ file")
    parser.add_argument("--board", default="square8", help="Board type")
    args = parser.parse_args()

    input_path = Path(args.input)
    board_type = BoardType(args.board)

    all_games = []
    if input_path.is_dir():
        for jsonl_file in input_path.glob("*.jsonl"):
            print(f"Loading {jsonl_file.name}...")
            all_games.extend(load_jsonl(jsonl_file))
    else:
        all_games = load_jsonl(input_path)

    print(f"Loaded {len(all_games)} games")

    samples = extract_samples(all_games, board_type)
    print(f"Extracted {samples['num_samples']} samples")

    np.savez(args.output, **samples)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
