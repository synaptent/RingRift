#!/usr/bin/env python3
"""
Validate game databases against the phase recording invariant (RR-CANON-R074/R075).

This script checks that every game in a database has:
- Every player recording an action for every mandatory turn phase for every turn
- Valid move types for each phase
- Proper no_*_action markers when players have no legal moves

Usage:
    python ai-service/scripts/validate_phase_recording.py [--delete-invalid]
"""
from __future__ import annotations


import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib

from app.models.core import GamePhase

# Canonical phases per RR-CANON-R070
CANONICAL_PHASES = [
    GamePhase.RING_PLACEMENT,
    GamePhase.MOVEMENT,
    GamePhase.CAPTURE,
    GamePhase.CHAIN_CAPTURE,
    GamePhase.LINE_PROCESSING,
    GamePhase.TERRITORY_PROCESSING,
    GamePhase.FORCED_ELIMINATION,
]

# Phases that must always have a recorded action when entered
MANDATORY_RECORD_PHASES = [
    GamePhase.RING_PLACEMENT,
    GamePhase.MOVEMENT,
    GamePhase.LINE_PROCESSING,
    GamePhase.TERRITORY_PROCESSING,
]

# Map of phases to valid move types (aligned with game_engine.py _assert_phase_move_invariant)
PHASE_TO_VALID_MOVE_TYPES: dict[GamePhase, set[str]] = {
    GamePhase.RING_PLACEMENT: {"place_ring", "skip_placement", "no_placement_action"},
    GamePhase.MOVEMENT: {
        "move_stack", "move_ring", "build_stack",  # build_stack is legacy alias
        "overtaking_capture", "continue_capture_segment",  # captures can start from movement
        "no_movement_action",
        "recovery_slide", "skip_recovery",  # RR-CANON-R110-R115
    },
    GamePhase.CAPTURE: {"overtaking_capture", "continue_capture_segment", "skip_capture"},
    GamePhase.CHAIN_CAPTURE: {"overtaking_capture", "continue_capture_segment"},
    GamePhase.LINE_PROCESSING: {
        "process_line",
        "choose_line_option",  # canonical
        "choose_line_reward",  # legacy alias
        "no_line_action",
        "line_formation",  # legacy
    },
    GamePhase.TERRITORY_PROCESSING: {
        "choose_territory_option",  # canonical
        "process_territory_region",  # legacy alias
        "eliminate_rings_from_stack",
        "skip_territory_processing",
        "no_territory_action",
        "territory_claim",  # legacy
    },
    GamePhase.FORCED_ELIMINATION: {"forced_elimination"},
}

# Legacy move types accepted in any phase for backward compatibility
LEGACY_MOVE_TYPES = {"line_formation", "territory_claim", "chain_capture", "swap_sides"}


@dataclass
class PhaseViolation:
    game_id: str
    turn_number: int
    player_number: int
    phase: str
    reason: str


def infer_phase_from_move_type(move_type: str) -> str | None:
    """Infer the phase from move type when phase field is not set."""
    move_type_lower = move_type.lower()

    if move_type_lower in ("place_ring", "skip_placement", "no_placement_action"):
        return "ring_placement"
    elif move_type_lower in ("move_stack", "move_ring", "build_stack", "no_movement_action",
                              "recovery_slide", "skip_recovery"):
        return "movement"
    elif move_type_lower in ("overtaking_capture", "skip_capture"):
        return "capture"
    elif move_type_lower == "continue_capture_segment":
        return "chain_capture"
    elif move_type_lower in ("process_line", "choose_line_option", "choose_line_reward",
                              "no_line_action", "line_formation"):
        return "line_processing"
    elif move_type_lower in ("choose_territory_option", "process_territory_region",
                              "skip_territory_processing", "no_territory_action", "territory_claim"):
        return "territory_processing"
    elif move_type_lower == "eliminate_rings_from_stack":
        return "territory_processing"  # Default, could also be forced_elimination
    elif move_type_lower == "forced_elimination":
        return "forced_elimination"
    elif move_type_lower in ("swap_sides", "chain_capture"):
        return None  # Meta-moves, valid in any phase
    return None


def validate_game_phase_recording(game_id: str, moves: list[dict], num_players: int) -> list[PhaseViolation]:
    """Validate that a game adheres to phase recording invariants."""
    violations = []

    # Track phase records per player per turn
    turn_records: dict[tuple[int, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    current_turn = 1
    current_player = 1

    for move in moves:
        move_player = move.get("player", move.get("playerNumber", 1))
        move_type = move.get("type", move.get("move_type", ""))
        move_phase = move.get("phase") or infer_phase_from_move_type(move_type)

        # Detect turn boundary
        if move_player != current_player:
            if move_player < current_player:
                current_turn += 1
            current_player = move_player

        # Record the phase
        if move_phase:
            key = (current_turn, move_player)
            turn_records[key][move_phase].append(move)

    # Check each turn/player combination
    for (turn_number, player_number), phase_records in turn_records.items():
        for phase in MANDATORY_RECORD_PHASES:
            phase_name = phase.value if hasattr(phase, "value") else str(phase)

            # forced_elimination is conditional, skip checking it
            if phase == GamePhase.FORCED_ELIMINATION:
                continue

            records = phase_records.get(phase_name, [])
            if not records:
                violations.append(
                    PhaseViolation(
                        game_id=game_id,
                        turn_number=turn_number,
                        player_number=player_number,
                        phase=phase_name,
                        reason=f"No recorded action for {phase_name} phase",
                    )
                )
            else:
                # Validate move types
                valid_types = PHASE_TO_VALID_MOVE_TYPES.get(phase, set())
                for record in records:
                    move_type = record.get("type", record.get("move_type", ""))
                    move_type_lower = move_type.lower()
                    # Legacy move types are accepted in any phase
                    if move_type_lower in LEGACY_MOVE_TYPES:
                        continue
                    if move_type_lower not in {t.lower() for t in valid_types}:
                        violations.append(
                            PhaseViolation(
                                game_id=game_id,
                                turn_number=turn_number,
                                player_number=player_number,
                                phase=phase_name,
                                reason=f"Invalid move type '{move_type}' for {phase_name} phase",
                            )
                        )

    return violations


def validate_database(db_path: Path) -> tuple[int, int, list[PhaseViolation]]:
    """Validate all games in a database. Returns (total_games, valid_games, all_violations)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total_games = 0
    valid_games = 0
    all_violations = []

    try:
        # Get all games
        cursor = conn.execute("SELECT game_id, num_players FROM games")
        games = cursor.fetchall()

        for game in games:
            game_id = game["game_id"]
            num_players = game["num_players"]
            total_games += 1

            # Try game_moves table first (new schema), then moves table (old schema)
            moves = []
            try:
                move_cursor = conn.execute(
                    "SELECT * FROM game_moves WHERE game_id = ? ORDER BY move_number", (game_id,)
                )
                for row in move_cursor:
                    move_data = dict(row)
                    # Parse JSON fields if present
                    if move_data.get("move_json"):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            move_data.update(json.loads(move_data["move_json"]))
                    moves.append(move_data)
            except sqlite3.OperationalError:
                # Try old schema
                move_cursor = conn.execute("SELECT * FROM moves WHERE game_id = ? ORDER BY move_number", (game_id,))
                for row in move_cursor:
                    move_data = dict(row)
                    if move_data.get("move_data"):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            move_data.update(json.loads(move_data["move_data"]))
                    moves.append(move_data)

            # Validate
            violations = validate_game_phase_recording(game_id, moves, num_players)
            if violations:
                all_violations.extend(violations)
            else:
                valid_games += 1

    finally:
        conn.close()

    return total_games, valid_games, all_violations


def main():
    parser = argparse.ArgumentParser(description="Validate game databases against phase recording invariant")
    parser.add_argument(
        "--keep-invalid", action="store_true", help="Keep invalid databases (by default, invalid databases are deleted)"
    )
    parser.add_argument("--db", type=str, help="Specific database to validate")
    args = parser.parse_args()

    # Find all databases
    data_dir = Path(__file__).parent.parent / "data" / "games"

    if args.db:
        db_paths = [Path(args.db)]
    else:
        db_paths = list(data_dir.glob("*.db"))

    print(f"Found {len(db_paths)} databases to validate")
    print("=" * 80)

    invalid_dbs = []

    for db_path in sorted(db_paths):
        print(f"\nValidating: {db_path.name}")

        try:
            total, valid, violations = validate_database(db_path)
            invalid = total - valid

            status = "✅ VALID" if invalid == 0 else "❌ INVALID"
            print(f"  {status}: {valid}/{total} games valid")

            if violations:
                # Show first few violations
                print("  First violations:")
                for v in violations[:5]:
                    print(f"    - Game {v.game_id}, Turn {v.turn_number}, P{v.player_number}: {v.reason}")
                if len(violations) > 5:
                    print(f"    ... and {len(violations) - 5} more violations")

                invalid_dbs.append(db_path)

        except Exception as e:
            print(f"  ⚠️  ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"Summary: {len(invalid_dbs)}/{len(db_paths)} databases contain invalid games")

    if invalid_dbs:
        print("\nDatabases with phase recording violations:")
        for db in invalid_dbs:
            print(f"  - {db}")

        if args.keep_invalid:
            print("\nKeeping invalid databases (--keep-invalid specified).")
        else:
            print("\n⚠️  Deleting invalid databases...")
            for db in invalid_dbs:
                print(f"  Deleting: {db}")
                db.unlink()
            print("Done. Invalid databases deleted.")
            print("Use --keep-invalid to preserve invalid databases.")

    return 1 if invalid_dbs else 0


if __name__ == "__main__":
    sys.exit(main())
