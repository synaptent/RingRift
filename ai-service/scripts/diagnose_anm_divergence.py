#!/usr/bin/env python3
"""Diagnostic tool for ANM state divergence between Python and TypeScript.

This script analyzes the hexagonal ANM parity bug where:
- 7/11 hexagonal games fail with ANM state divergence
- Divergence occurs in `line_processing` phase
- Python reports `is_anm: false` (finds lines)
- TypeScript reports `is_anm: true` (no lines found)
- State hashes match (board state is identical)

Usage:
    python scripts/diagnose_anm_divergence.py \
        --parity-gate data/canonical_hexagonal.parity_gate.json \
        --verbose

    python scripts/diagnose_anm_divergence.py \
        --db data/games/canonical_hexagonal.db \
        --game-id <uuid> \
        --move-index <k> \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.board_manager import BoardManager
from app.db.game_replay import GameReplayDB
from app.models import BoardType, GameState, Position
from app.rules.global_actions import (
    global_legal_actions_summary,
    has_phase_local_interactive_move,
    is_anm_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_line_detection(state: GameState, player: int) -> dict:
    """Analyze line detection results for a game state.

    Returns detailed information about what lines are detected.
    """
    board = state.board
    num_players = len(state.players)

    # Get all marker information
    marker_keys = list(board.markers.keys())
    marker_details = []
    for key, marker in board.markers.items():
        marker_details.append({
            "key": key,
            "player": marker.player,
            "position": {"x": marker.position.x, "y": marker.position.y, "z": marker.position.z},
        })

    # Detect lines using Python's BoardManager
    all_lines = BoardManager.find_all_lines(board, num_players)
    player_lines = [line for line in all_lines if line.player == player]

    line_details = []
    for line in player_lines:
        line_details.append({
            "player": line.player,
            "length": line.length,
            "positions": [
                {"x": p.x, "y": p.y, "z": p.z}
                for p in line.positions
            ],
            "direction": {"x": line.direction.x, "y": line.direction.y, "z": line.direction.z},
        })

    # Get ANM-related summary
    summary = global_legal_actions_summary(state, player)

    return {
        "board_type": str(board.type),
        "board_size": board.size,
        "current_phase": str(state.current_phase),
        "current_player": state.current_player,
        "analysis_player": player,
        "marker_count": len(marker_keys),
        "marker_keys_sample": marker_keys[:10] if len(marker_keys) > 10 else marker_keys,
        "marker_details": marker_details[:20] if len(marker_details) > 20 else marker_details,
        "total_lines_detected": len(all_lines),
        "player_lines_detected": len(player_lines),
        "line_details": line_details,
        "has_phase_local_interactive_move": has_phase_local_interactive_move(state, player),
        "is_anm_state": is_anm_state(state),
        "summary": {
            "has_turn_material": summary.has_turn_material,
            "has_global_placement_action": summary.has_global_placement_action,
            "has_phase_local_interactive_move": summary.has_phase_local_interactive_move,
            "has_forced_elimination_action": summary.has_forced_elimination_action,
        },
    }


def run_typescript_analysis(db_path: str, game_id: str, move_index: int) -> dict | None:
    """Run TypeScript replay to get TS-side analysis.

    Returns the TypeScript summary or None if failed.
    """
    # Run the TypeScript replay harness
    cmd = [
        "npx", "ts-node",
        "--transpile-only",
        "scripts/selfplay-db-ts-replay.ts",
        "--db", db_path,
        "--game-id", game_id,
        "--start-at", str(move_index),
        "--end-at", str(move_index + 1),
        "--emit-summary",
        "--json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).resolve().parents[2]),  # RingRift root
        )

        if result.returncode != 0:
            logger.warning(f"TypeScript replay failed: {result.stderr[:500]}")
            return None

        # Parse JSON output
        for line in result.stdout.split("\n"):
            if line.strip().startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        return None
    except subprocess.TimeoutExpired:
        logger.warning("TypeScript replay timed out")
        return None
    except FileNotFoundError:
        logger.warning("npx or ts-node not found")
        return None


def analyze_divergence_from_gate(gate_file: Path, verbose: bool = False) -> None:
    """Analyze ANM divergences from a parity gate file."""
    with open(gate_file) as f:
        gate = json.load(f)

    # Look in parity_summary.semantic_divergences for actual divergences
    parity_summary = gate.get("parity_summary", {})
    failures = parity_summary.get("semantic_divergences", [])
    anm_failures = [f for f in failures if "anm_state" in f.get("mismatch_kinds", [])]

    print(f"\n{'='*60}")
    print(f"ANM Divergence Analysis")
    print(f"{'='*60}")
    print(f"Gate file: {gate_file}")
    print(f"Total failures: {len(failures)}")
    print(f"ANM failures: {len(anm_failures)}")
    print()

    if not anm_failures:
        print("No ANM failures found.")
        return

    # Get the database path
    db_path = gate.get("db_path", "")
    if not db_path or not Path(db_path).exists():
        # Try from the first divergence
        if anm_failures:
            db_path = anm_failures[0].get("db_path", "")
    if not db_path or not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        return

    db = GameReplayDB(db_path)

    for i, failure in enumerate(anm_failures[:3]):  # Analyze first 3
        game_id = failure.get("game_id", "")
        move_index = failure.get("diverged_at", 0)

        print(f"\n{'-'*60}")
        print(f"Failure {i+1}: {game_id} @ k={move_index}")
        print(f"{'-'*60}")

        # Get Python state
        state = db.get_state_at_move(game_id, move_index)
        if not state:
            print(f"  ERROR: Could not load state at k={move_index}")
            continue

        player = state.current_player
        analysis = analyze_line_detection(state, player)

        print(f"\nPython Analysis:")
        print(f"  Board type: {analysis['board_type']}")
        print(f"  Phase: {analysis['current_phase']}")
        print(f"  Player: {analysis['analysis_player']}")
        print(f"  Marker count: {analysis['marker_count']}")
        print(f"  Total lines detected: {analysis['total_lines_detected']}")
        print(f"  Player lines detected: {analysis['player_lines_detected']}")
        print(f"  has_phase_local_interactive_move: {analysis['has_phase_local_interactive_move']}")
        print(f"  is_anm_state: {analysis['is_anm_state']}")

        if verbose and analysis['line_details']:
            print(f"\n  Lines found:")
            for j, line in enumerate(analysis['line_details'][:5]):
                positions = [f"({p['x']},{p['y']},{p['z']})" for p in line['positions']]
                print(f"    {j+1}. Player {line['player']}, length {line['length']}")
                print(f"       Positions: {' -> '.join(positions)}")

        if verbose:
            print(f"\n  Sample marker keys: {analysis['marker_keys_sample']}")

        # TypeScript summary from failure
        ts_summary = failure.get("ts_summary", {})
        print(f"\nTypeScript Summary (from gate):")
        print(f"  Phase: {ts_summary.get('current_phase')}")
        print(f"  is_anm: {ts_summary.get('is_anm')}")
        print(f"  state_hash: {ts_summary.get('state_hash')}")

        # Python summary from failure
        py_summary = failure.get("python_summary", {})
        print(f"\nPython Summary (from gate):")
        print(f"  Phase: {py_summary.get('current_phase')}")
        print(f"  is_anm: {py_summary.get('is_anm')}")
        print(f"  state_hash: {py_summary.get('state_hash')}")

        # Check marker key format
        if analysis['marker_details']:
            sample = analysis['marker_details'][0]
            print(f"\n  Key format check:")
            print(f"    Sample key: {sample['key']}")
            print(f"    Position: x={sample['position']['x']}, y={sample['position']['y']}, z={sample['position']['z']}")

            # Check if key format matches expected
            pos = sample['position']
            if pos['z'] is not None:
                expected_key = f"{pos['x']},{pos['y']},{pos['z']}"
            else:
                expected_key = f"{pos['x']},{pos['y']}"

            if sample['key'] != expected_key:
                print(f"    WARNING: Key mismatch! Expected: {expected_key}")
            else:
                print(f"    Key format OK")

    print(f"\n{'='*60}")


def analyze_single_game(
    db_path: str,
    game_id: str,
    move_index: int,
    verbose: bool = False,
) -> None:
    """Analyze a specific game state for ANM divergence."""
    db = GameReplayDB(db_path)

    state = db.get_state_at_move(game_id, move_index)
    if not state:
        logger.error(f"Could not load state for {game_id} @ k={move_index}")
        return

    player = state.current_player
    analysis = analyze_line_detection(state, player)

    print(f"\n{'='*60}")
    print(f"ANM Analysis: {game_id} @ k={move_index}")
    print(f"{'='*60}")

    print(json.dumps(analysis, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Diagnose ANM state divergence")
    parser.add_argument(
        "--parity-gate",
        type=Path,
        help="Path to parity gate JSON file",
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to game database",
    )
    parser.add_argument(
        "--game-id",
        type=str,
        help="Game ID to analyze",
    )
    parser.add_argument(
        "--move-index",
        type=int,
        help="Move index to analyze",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.parity_gate:
        analyze_divergence_from_gate(args.parity_gate, args.verbose)
    elif args.db and args.game_id and args.move_index is not None:
        analyze_single_game(args.db, args.game_id, args.move_index, args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
