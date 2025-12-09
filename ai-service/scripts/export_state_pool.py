#!/usr/bin/env python
"""Export mid-game states from recorded games to JSONL evaluation pools.

This utility extracts states from games recorded in a GameReplayDB SQLite
database and exports them to JSONL format compatible with the eval_pools
loader used by CMA-ES and other training scripts.

Usage examples
--------------

From the ``ai-service`` root::

    # Export states from a CMA-ES run's game database
    python scripts/export_state_pool.py \\
        --db logs/cmaes/square8_v2/runs/cmaes_20251202_120000/games.db \\
        --output data/eval_pools/square8_generated/pool_v2.jsonl \\
        --board-type square8 \\
        --num-players 2 \\
        --sample-moves 20,40,60 \\
        --min-game-length 50 \\
        --max-states 500

    # Export from multiple databases
    python scripts/export_state_pool.py \\
        --db logs/cmaes/run1/games.db \\
        --db logs/cmaes/run2/games.db \\
        --output data/eval_pools/combined_pool.jsonl \\
        --sample-moves 30,60,90

    # Filter to only export from games with a winner
    python scripts/export_state_pool.py \\
        --db games.db \\
        --output pool.jsonl \\
        --winners-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import GameReplayDB  # noqa: E402
from app.models import BoardType, GameState  # noqa: E402


def export_states_from_db(
    db_path: str,
    sample_moves: List[int],
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    min_game_length: int = 0,
    max_states: Optional[int] = None,
    winners_only: bool = False,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract game states from a recorded game database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    sample_moves:
        List of move numbers at which to sample states (e.g., [20, 40, 60]).
    board_type:
        Optional filter for board type (e.g., "square8").
    num_players:
        Optional filter for number of players.
    min_game_length:
        Minimum number of moves a game must have to be included.
    max_states:
        Maximum total number of states to export.
    winners_only:
        If True, only include games that have a winner.
    source_filter:
        Optional filter for game source (e.g., "cmaes").

    Returns
    -------
    List[Dict[str, Any]]
        List of state dictionaries ready for JSONL serialization.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    db = GameReplayDB(db_path)
    states: List[Dict[str, Any]] = []

    # Build query to list games
    query = "SELECT game_id, board_type, num_players, total_moves, winner, source FROM games WHERE 1=1"
    params: List[Any] = []

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)

    if num_players is not None:
        query += " AND num_players = ?"
        params.append(num_players)

    if min_game_length > 0:
        query += " AND total_moves >= ?"
        params.append(min_game_length)

    if winners_only:
        query += " AND winner IS NOT NULL"

    if source_filter:
        query += " AND source = ?"
        params.append(source_filter)

    # Execute query using the proper connection context
    with db._get_conn() as conn:
        cursor = conn.execute(query, params)
        games = cursor.fetchall()

    print(f"Found {len(games)} matching games in {db_path}")

    for game_row in games:
        game_id = game_row["game_id"]
        game_board_type = game_row["board_type"]
        game_num_players = game_row["num_players"]
        game_total_moves = game_row["total_moves"]

        # Sample states at specified move numbers
        for sample_move in sample_moves:
            if sample_move > game_total_moves:
                continue

            # Try to get the state at this move using the DB's get_state_at_move method
            try:
                state = db.get_state_at_move(game_id, sample_move)
                if state is None:
                    continue

                # Convert to dict for JSON serialization
                state_dict = state.model_dump()
                state_dict["_export_meta"] = {
                    "source_db": os.path.basename(db_path),
                    "game_id": game_id,
                    "sample_move": sample_move,
                    "board_type": game_board_type,
                    "num_players": game_num_players,
                }
                states.append(state_dict)

                if max_states is not None and len(states) >= max_states:
                    return states
            except Exception as e:
                print(f"WARNING: Failed to get state for game {game_id} move {sample_move}: {e}")
                continue

    return states


def main():
    parser = argparse.ArgumentParser(description="Export mid-game states from recorded games to JSONL pools")
    parser.add_argument(
        "--db",
        type=str,
        action="append",
        required=True,
        help="Path to SQLite database file(s). Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the JSONL file",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default=None,
        choices=["square8", "square19", "hex"],
        help="Filter by board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=None,
        help="Filter by number of players",
    )
    parser.add_argument(
        "--sample-moves",
        type=str,
        default="20,40,60",
        help="Comma-separated list of move numbers at which to sample states (default: 20,40,60)",
    )
    parser.add_argument(
        "--min-game-length",
        type=int,
        default=0,
        help="Minimum game length in moves (default: 0)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Maximum number of states to export (default: unlimited)",
    )
    parser.add_argument(
        "--winners-only",
        action="store_true",
        help="Only include games that have a winner",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by game source (e.g., 'cmaes')",
    )

    args = parser.parse_args()

    # Parse sample moves
    try:
        sample_moves = [int(m.strip()) for m in args.sample_moves.split(",")]
    except ValueError:
        print(f"ERROR: Invalid sample-moves format: {args.sample_moves}")
        sys.exit(1)

    print(f"Exporting states from {len(args.db)} database(s)")
    print(f"Sample moves: {sample_moves}")
    if args.board_type:
        print(f"Board type filter: {args.board_type}")
    if args.num_players:
        print(f"Num players filter: {args.num_players}")
    if args.min_game_length > 0:
        print(f"Min game length: {args.min_game_length}")
    if args.max_states:
        print(f"Max states: {args.max_states}")
    if args.winners_only:
        print("Winners only: True")
    if args.source:
        print(f"Source filter: {args.source}")
    print()

    all_states: List[Dict[str, Any]] = []

    for db_path in args.db:
        try:
            states = export_states_from_db(
                db_path=db_path,
                sample_moves=sample_moves,
                board_type=args.board_type,
                num_players=args.num_players,
                min_game_length=args.min_game_length,
                max_states=(args.max_states - len(all_states) if args.max_states else None),
                winners_only=args.winners_only,
                source_filter=args.source,
            )
            all_states.extend(states)
            print(f"  Extracted {len(states)} states from {db_path}")

            if args.max_states and len(all_states) >= args.max_states:
                print(f"Reached max states limit ({args.max_states})")
                break
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue
        except Exception as e:
            print(f"ERROR processing {db_path}: {e}")
            continue

    if not all_states:
        print("\nNo states extracted. Check your filters and database contents.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write JSONL output
    with open(args.output, "w", encoding="utf-8") as f:
        for state in all_states:
            json.dump(state, f, separators=(",", ":"))
            f.write("\n")

    print(f"\nExported {len(all_states)} states to {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
