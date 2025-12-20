#!/usr/bin/env python3
"""Distill CNN policy knowledge to NNUE training data.

This script generates soft policy targets from a trained CNN model,
enabling KL divergence training for NNUE without expensive MCTS search.

Supports both JSONL files and SQLite databases as input.

Usage (JSONL):
    python scripts/distill_cnn_to_nnue.py \
        --input data/selfplay/games.jsonl \
        --output data/distilled/sq8_cnn_policy.jsonl \
        --board-type square8

Usage (SQLite DB):
    python scripts/distill_cnn_to_nnue.py \
        --input data/consolidated_training.db \
        --output data/distilled/sq8_cnn_policy.jsonl \
        --board-type square8
"""

import argparse
import gzip
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType, GameState, Move, MoveType, Position, AIConfig
from app.ai.neural_net import NeuralNetAI, get_policy_size_for_board
from app.ai.nnue import get_board_size
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_board_type(board_str: str) -> BoardType:
    """Parse board type string to enum."""
    board_str = board_str.lower()
    if "square8" in board_str or "sq8" in board_str:
        return BoardType.SQUARE8
    elif "square19" in board_str or "sq19" in board_str:
        return BoardType.SQUARE19
    elif "hex" in board_str:
        return BoardType.HEXAGONAL
    return BoardType.SQUARE8


def parse_move(move_dict: dict, move_number: int) -> Move | None:
    """Parse a move dict into a Move object.

    Handles both DB-style (full JSON) and minimal move dicts.
    """
    from datetime import datetime

    move_type_str = move_dict.get("type", "")
    try:
        MoveType(move_type_str)  # Validate type exists
    except ValueError:
        return None

    # If move_dict has timestamp and thinkTime, it's a full Move from DB
    if "timestamp" in move_dict and ("thinkTime" in move_dict or "think_time" in move_dict):
        try:
            return Move(**move_dict)
        except Exception:
            pass  # Fall through to manual construction

    # Manual construction for minimal move dicts
    def parse_pos(pos_dict):
        if not pos_dict or not isinstance(pos_dict, dict):
            return None
        return Position(
            x=pos_dict.get("x", 0),
            y=pos_dict.get("y", 0),
            z=pos_dict.get("z"),
        )

    from_pos = parse_pos(move_dict.get("from") or move_dict.get("from_pos"))
    to_pos = parse_pos(move_dict.get("to"))
    capture_target = parse_pos(move_dict.get("capture_target") or move_dict.get("captureTarget"))

    return Move(
        id=move_dict.get("id", f"distill-{move_number}"),
        type=MoveType(move_type_str),
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        move_number=move_number,
        timestamp=datetime.now(),
        think_time=move_dict.get("thinkTime", 0),
    )


class CNNDistiller:
    """Distills CNN policy knowledge into soft targets."""

    def __init__(
        self,
        board_type: BoardType,
        num_players: int = 2,
        device: str = "cuda",
        temperature: float = 1.0,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.temperature = temperature

        # Load CNN model via NeuralNetAI (handles model loading automatically)
        logger.info(f"Loading CNN model for {board_type.value}...")
        ai_config = AIConfig(
            difficulty=10,
            use_neural_net=True,
        )
        self.nn_ai = NeuralNetAI(
            player_number=1,
            config=ai_config,
            board_type=board_type,
        )

        self.policy_size = get_policy_size_for_board(board_type)
        self.board_size = get_board_size(board_type)
        logger.info(f"CNN loaded: policy_size={self.policy_size}, device={self.device}")

    def get_policy(self, state: GameState, player: int) -> dict[str, float]:
        """Get CNN policy distribution for a state.

        Returns sparse policy dict {move_idx: prob}.
        """
        # Get legal moves
        legal_moves = GameEngine.get_valid_moves(state, player)
        if not legal_moves:
            return {}

        # Use NeuralNetAI to evaluate - it handles encoding internally
        self.nn_ai.player_number = player
        try:
            # Get policy from the model
            values, policy_probs = self.nn_ai.evaluate_batch([state])
            if policy_probs is None or len(policy_probs) == 0:
                return {}

            probs = policy_probs[0]

            # Create sparse policy dict mapping move indices to probabilities
            policy_dict = {}
            for move_idx, move in enumerate(legal_moves[:128]):  # Max 128 moves
                # Encode move to get its policy index
                try:
                    policy_idx = self.nn_ai.encode_move(move, state)
                    if 0 <= policy_idx < len(probs):
                        prob = float(probs[policy_idx])
                        if prob > 1e-6:
                            policy_dict[str(move_idx)] = prob
                except Exception:
                    continue

            # Normalize
            total = sum(policy_dict.values())
            if total > 0:
                policy_dict = {k: v / total for k, v in policy_dict.items()}

            return policy_dict

        except Exception as e:
            logger.debug(f"Failed to get policy: {e}")
            return {}

    def distill_game(
        self,
        game: dict[str, Any],
        sample_every: int = 1,
    ) -> dict[str, Any]:
        """Add CNN policy distributions to a game record."""
        # Parse initial state
        board_type_str = game.get("board_type", "square8")
        num_players = game.get("num_players", 2)

        initial_state_dict = game.get("initial_state")
        if initial_state_dict:
            try:
                state = GameState(**initial_state_dict)
            except Exception:
                state = create_initial_state(self.board_type, num_players)
        else:
            state = create_initial_state(self.board_type, num_players)

        moves = game.get("moves", [])
        distilled_moves = []
        positions_distilled = 0

        for move_idx, move_dict in enumerate(moves):
            new_move = dict(move_dict)

            # Distill every Nth position
            game_status = state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status)
            if move_idx % sample_every == 0 and game_status == "active":
                current_player = state.current_player or 1

                # Get CNN policy for this position
                try:
                    policy_dict = self.get_policy(state, current_player)
                    if policy_dict:
                        # Use mcts_policy key for compatibility with training code
                        new_move["mcts_policy"] = policy_dict
                        positions_distilled += 1
                except Exception as e:
                    logger.debug(f"Failed to distill position {move_idx}: {e}")

            distilled_moves.append(new_move)

            # Apply move to advance state
            try:
                move = parse_move(move_dict, move_idx)
                if move is not None:
                    state = GameEngine.apply_move(state, move)
                else:
                    logger.debug(f"Failed to parse move {move_idx}: {move_dict.get('type')}")
                    break
            except Exception as e:
                logger.debug(f"Failed to apply move {move_idx}: {e}")
                break

        # Return game with distilled policies
        result = dict(game)
        result["moves"] = distilled_moves
        result["distilled"] = True
        result["positions_distilled"] = positions_distilled

        return result


def extract_games_from_db(
    db_path: str,
    board_type: BoardType,
    num_players: int,
    max_games: int | None = None,
    min_game_length: int = 20,
) -> list[dict[str, Any]]:
    """Extract games from SQLite database.

    Returns list of game dicts compatible with distill_game().
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Detect schema - different databases use different column names
    cursor.execute("PRAGMA table_info(games)")
    columns = {row['name'] for row in cursor.fetchall()}
    moves_col = 'total_moves' if 'total_moves' in columns else 'move_count'

    # Get completed games
    board_type_str = board_type.value.lower()
    query = f"""
        SELECT game_id, winner, {moves_col} as move_count
        FROM games
        WHERE game_status = 'completed'
          AND winner IS NOT NULL
          AND board_type = ?
          AND num_players = ?
          AND {moves_col} >= ?
        ORDER BY game_id
    """
    if max_games:
        query += f" LIMIT {max_games}"

    cursor.execute(query, (board_type_str, num_players, min_game_length))
    game_rows = cursor.fetchall()
    logger.info(f"Found {len(game_rows)} games in {db_path}")

    games = []
    for row in game_rows:
        game_id = row['game_id']
        winner = row['winner']
        move_count = row['move_count'] or 0

        # Get initial state
        cursor.execute(
            "SELECT initial_state_json, compressed FROM game_initial_state WHERE game_id = ?",
            (game_id,)
        )
        initial_row = cursor.fetchone()

        initial_state = None
        if initial_row:
            initial_json = initial_row['initial_state_json']
            if initial_row['compressed']:
                try:
                    if isinstance(initial_json, (bytes, bytearray)):
                        initial_json = gzip.decompress(bytes(initial_json)).decode("utf-8")
                    else:
                        initial_json = gzip.decompress(str(initial_json).encode("utf-8")).decode("utf-8")
                except Exception:
                    initial_json = None

            if initial_json:
                try:
                    initial_state = json.loads(initial_json)
                except Exception:
                    pass

        # Get moves - schema stores moves as JSON in move_json column
        cursor.execute(
            """SELECT move_number, move_type, player, move_json
               FROM game_moves
               WHERE game_id = ?
               ORDER BY move_number""",
            (game_id,)
        )
        move_rows = cursor.fetchall()

        moves = []
        for mv in move_rows:
            # Parse move_json if available, otherwise construct from columns
            move_json_str = mv['move_json']
            if move_json_str:
                try:
                    move_dict = json.loads(move_json_str)
                    # Normalize keys for compatibility
                    if 'captureTarget' in move_dict and move_dict['captureTarget']:
                        move_dict['capture_target'] = move_dict.pop('captureTarget')
                    moves.append(move_dict)
                except json.JSONDecodeError:
                    # Fallback to basic structure
                    move_dict = {
                        "type": mv['move_type'],
                        "player": mv['player'],
                    }
                    moves.append(move_dict)
            else:
                move_dict = {
                    "type": mv['move_type'],
                    "player": mv['player'],
                }
                moves.append(move_dict)

        game = {
            "game_id": game_id,
            "board_type": board_type_str,
            "num_players": num_players,
            "winner": winner,
            "move_count": move_count,
            "game_status": "completed",  # We only extract completed games
            "moves": moves,
            "initial_state": initial_state,
        }
        games.append(game)

    conn.close()
    return games


def main():
    parser = argparse.ArgumentParser(description="Distill CNN policy to NNUE training data")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file or .db file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--board-type", type=str, default="square8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--sample-every", type=int, default=1, help="Distill every Nth position")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process")
    parser.add_argument("--min-game-length", type=int, default=20, help="Minimum game length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for CNN inference")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    board_type = parse_board_type(args.board_type)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    # Detect input type
    is_db_input = input_path.suffix.lower() == '.db'

    logger.info(f"CNN Policy Distillation")
    logger.info(f"  Input: {args.input} ({'SQLite DB' if is_db_input else 'JSONL'})")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Board: {board_type.value}, Players: {args.num_players}")
    logger.info(f"  Temperature: {args.temperature}")

    # Initialize distiller
    distiller = CNNDistiller(
        board_type=board_type,
        num_players=args.num_players,
        device=args.device,
        temperature=args.temperature,
    )

    # Process games
    start_time = time.time()
    games_processed = 0
    total_positions = 0

    if is_db_input:
        # Extract games from database
        games = extract_games_from_db(
            str(input_path),
            board_type,
            args.num_players,
            max_games=args.max_games,
            min_game_length=args.min_game_length,
        )
        logger.info(f"Extracted {len(games)} games from database")

        with open(output_path, 'w') as fout:
            for game_idx, game in enumerate(games):
                try:
                    distilled = distiller.distill_game(game, sample_every=args.sample_every)
                    fout.write(json.dumps(distilled) + "\n")

                    games_processed += 1
                    total_positions += distilled.get("positions_distilled", 0)

                    if games_processed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = games_processed / elapsed
                        eta = (len(games) - games_processed) / rate if rate > 0 else 0
                        logger.info(f"Processed {games_processed}/{len(games)} games ({rate:.1f}/s, ETA: {eta:.0f}s), {total_positions} positions")

                except Exception as e:
                    logger.warning(f"Failed to process game {game_idx}: {e}")
    else:
        # Process JSONL file
        with open(args.input, 'r') as fin, open(output_path, 'w') as fout:
            for line_num, line in enumerate(fin):
                if args.max_games and games_processed >= args.max_games:
                    break

                if not line.strip():
                    continue

                try:
                    game = json.loads(line)

                    # Filter by board type and num_players
                    game_board = game.get("board_type", "").lower()
                    if board_type.value.lower() not in game_board:
                        continue
                    if game.get("num_players") != args.num_players:
                        continue

                    # Distill
                    distilled = distiller.distill_game(game, sample_every=args.sample_every)
                    fout.write(json.dumps(distilled) + "\n")

                    games_processed += 1
                    total_positions += distilled.get("positions_distilled", 0)

                    if games_processed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = games_processed / elapsed
                        logger.info(f"Processed {games_processed} games ({rate:.1f}/s), {total_positions} positions")

                except Exception as e:
                    logger.warning(f"Failed to process game {line_num}: {e}")

    elapsed = time.time() - start_time
    rate = games_processed / elapsed if elapsed > 0 else 0
    logger.info("=" * 60)
    logger.info("DISTILLATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games processed: {games_processed}")
    logger.info(f"Positions distilled: {total_positions}")
    logger.info(f"Time: {elapsed:.1f}s ({rate:.1f} games/s)")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
