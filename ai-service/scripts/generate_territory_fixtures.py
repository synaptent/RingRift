#!/usr/bin/env python3
"""Generate parity fixtures for territory processing moves.

This script extracts territory processing move states from selfplay databases
and creates parity fixture JSON files for TS↔Python comparison testing.

Usage (from ai-service/):
    python scripts/generate_territory_fixtures.py --max-fixtures 15
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add ai-service to path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.game_replay import GameReplayDB, _compute_state_hash


@dataclass
class StateSummary:
    move_index: int
    current_player: int
    current_phase: str
    game_status: str
    state_hash: str


def _canonicalize_status(status: str | None) -> str:
    """Normalize status strings."""
    if status is None:
        return "active"
    s = str(status)
    if s == "finished":
        return "completed"
    return s


def repo_root() -> Path:
    """Return the monorepo root."""
    return Path(__file__).resolve().parents[2]


def find_selfplay_dbs() -> List[Path]:
    """Find selfplay database files."""
    root = repo_root()
    data_dir = root / "ai-service" / "data" / "games"

    dbs = []
    for db_file in data_dir.glob("*.db"):
        # Skip empty or very small DBs
        if db_file.stat().st_size > 10000:
            dbs.append(db_file)
    return dbs


def get_territory_moves(db_path: Path, limit: int = 50) -> List[Tuple[str, int, str, str]]:
    """Get territory processing moves from a database.

    Returns list of (game_id, move_number, move_type, move_json) tuples.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT game_id, move_number, move_type, move_json
            FROM game_moves
            WHERE move_type IN (
                'process_territory_region',
                'eliminate_rings_from_stack',
                'skip_territory_processing'
            )
            ORDER BY game_id, move_number
            LIMIT ?
        """,
            (limit,),
        )

        results = cur.fetchall()
        return results
    except sqlite3.OperationalError:
        # Table doesn't exist or other error
        return []
    finally:
        conn.close()


def get_game_board_type(db_path: Path, game_id: str) -> str:
    """Get board type for a game."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    try:
        cur.execute("SELECT board_type FROM games WHERE game_id = ?", (game_id,))
        row = cur.fetchone()
        return row[0] if row else "square8"
    finally:
        conn.close()


def summarize_python_state(db: GameReplayDB, game_id: str, move_index: int) -> Optional[StateSummary]:
    """Summarize state AFTER move_index is applied."""
    try:
        state = db.get_state_at_move(game_id, move_index)
        if state is None:
            return None
        return StateSummary(
            move_index=move_index,
            current_player=state.current_player,
            current_phase=(
                state.current_phase.value if hasattr(state.current_phase, "value") else str(state.current_phase)
            ),
            game_status=_canonicalize_status(
                state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status)
            ),
            state_hash=_compute_state_hash(state),
        )
    except Exception as e:
        print(f"  Warning: Failed to get state for {game_id} @ {move_index}: {e}")
        return None


def generate_fixture(
    db_path: Path,
    game_id: str,
    move_number: int,
    move_type: str,
    move_json: str,
    output_dir: Path,
) -> Optional[Path]:
    """Generate a single parity fixture file."""

    # Open database
    db = GameReplayDB(str(db_path))

    # Get Python state summary
    python_summary = summarize_python_state(db, game_id, move_number)
    if python_summary is None:
        return None

    # Parse the move JSON
    try:
        canonical_move = json.loads(move_json)
    except json.JSONDecodeError:
        return None

    # Get board type for filename
    board_type = get_game_board_type(db_path, game_id)

    # Create fixture data
    fixture = {
        "canonical_move": canonical_move,
        "canonical_move_index": move_number,
        "db_path": str(db_path),
        "game_id": game_id,
        "move_type": move_type,
        "python_summary": {
            "current_phase": python_summary.current_phase,
            "current_player": python_summary.current_player,
            "game_status": python_summary.game_status,
            "move_index": python_summary.move_index,
            "state_hash": python_summary.state_hash,
        },
        "generated_at": datetime.now().isoformat(),
        "purpose": "territory_processing_coverage",
    }

    # Generate filename
    db_name = db_path.stem.replace(".", "_")
    filename = f"{db_name}__{game_id}__k{move_number}.json"
    output_path = output_dir / filename

    # Don't overwrite existing fixtures
    if output_path.exists():
        return None

    # Write fixture
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2, default=str)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate territory processing parity fixtures")
    parser.add_argument("--max-fixtures", type=int, default=15, help="Maximum fixtures to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for fixtures")
    args = parser.parse_args()

    root = repo_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "ai-service" / "parity_fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for selfplay databases...")
    dbs = find_selfplay_dbs()
    print(f"Found {len(dbs)} databases")

    generated = 0
    seen_positions = set()  # (game_id, move_number) to avoid duplicates

    for db_path in dbs:
        if generated >= args.max_fixtures:
            break

        print(f"\nProcessing {db_path.name}...")
        moves = get_territory_moves(db_path, limit=30)

        if not moves:
            print(f"  No territory moves found")
            continue

        print(f"  Found {len(moves)} territory moves")

        for game_id, move_number, move_type, move_json in moves:
            if generated >= args.max_fixtures:
                break

            # Skip if we've already generated for this position
            pos_key = (game_id, move_number)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            output_path = generate_fixture(db_path, game_id, move_number, move_type, move_json, output_dir)

            if output_path:
                generated += 1
                print(f"  [{generated}/{args.max_fixtures}] Generated: {output_path.name}")

    print(f"\n{'='*60}")
    print(f"Generated {generated} territory processing fixtures")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
