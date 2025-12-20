#!/usr/bin/env python3
"""
Upgrade old JSONL schema to current standardized format.

Old schema fields:
  - board_size: int (8, 19, 25, etc.)
  - victory_type: "ring_elimination", "timeout", "territory", "lps", "stalemate"
  - stalemate_tiebreaker: optional string

New schema fields:
  - board_type: "square8", "square19", "hexagonal", etc.
  - termination_reason: "status:completed:elimination", "status:completed:lps", etc.

Usage:
  python scripts/upgrade_jsonl_schema.py --input old.jsonl --output new.jsonl
  python scripts/upgrade_jsonl_schema.py --input-dir data/old/ --output-dir data/upgraded/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
from collections import Counter


# Mapping from board_size to board_type
BOARD_SIZE_TO_TYPE = {
    8: "square8",
    19: "square19",
    # Legacy hex embedding size used by early GPU/selfplay outputs.
    25: "hexagonal",
    11: "hexagonal",  # Common hex size
    13: "hexagonal",
    15: "hexagonal",
}

# Mapping from old victory_type to new termination_reason
VICTORY_TYPE_TO_TERMINATION = {
    "ring_elimination": "status:completed:elimination",
    "elimination": "status:completed:elimination",
    "territory": "status:completed:territory",
    "lps": "status:completed:lps",
    "last_player_standing": "status:completed:lps",
    "timeout": "status:completed:timeout",
    "stalemate": "status:completed:stalemate",
    "draw": "status:completed:draw",
}

# Stalemate tiebreaker mapping
STALEMATE_TIEBREAKER_TO_TERMINATION = {
    "territory": "status:completed:territory",
    "ring_count": "status:completed:elimination",
    "ring_elimination": "status:completed:elimination",
}


def infer_board_type(game: dict[str, Any]) -> str:
    """Infer board_type from available fields."""
    # Check if already present
    if "board_type" in game:
        return "hexagonal" if game["board_type"] == "square25" else game["board_type"]

    # Check nested config
    if "config" in game and "board_type" in game["config"]:
        value = game["config"]["board_type"]
        return "hexagonal" if value == "square25" else value

    # Infer from board_size
    board_size = game.get("board_size")
    if board_size in BOARD_SIZE_TO_TYPE:
        return BOARD_SIZE_TO_TYPE[board_size]

    # Try to infer from moves (max coordinate)
    if "moves" in game and game["moves"]:
        max_coord = 0
        for move in game["moves"]:
            if "to" in move:
                max_coord = max(max_coord, move["to"].get("x", 0), move["to"].get("y", 0))
            if "from" in move:
                max_coord = max(max_coord, move["from"].get("x", 0), move["from"].get("y", 0))

        # Infer board size from max coordinate
        if max_coord <= 7:
            return "square8"
        elif max_coord <= 18:
            return "square19"
        elif max_coord <= 24:
            return "hexagonal"

    # Check initial_state for board info
    if "initial_state" in game:
        state = game["initial_state"]
        if isinstance(state, dict):
            if "board_type" in state:
                return state["board_type"]
            if "boardType" in state:
                return state["boardType"].lower()

    return "unknown"


def infer_termination_reason(game: dict[str, Any]) -> str:
    """Infer termination_reason from available fields."""
    # Check if already present in new format
    if "termination_reason" in game:
        tr = game["termination_reason"]
        if tr.startswith("status:completed:"):
            return tr

    # Check nested config
    if "config" in game and "termination_reason" in game["config"]:
        return game["config"]["termination_reason"]

    # Map from old victory_type
    victory_type = game.get("victory_type", "").lower()

    # Handle stalemate with tiebreaker
    if victory_type == "stalemate" and game.get("stalemate_tiebreaker"):
        tiebreaker = game["stalemate_tiebreaker"].lower()
        if tiebreaker in STALEMATE_TIEBREAKER_TO_TERMINATION:
            return STALEMATE_TIEBREAKER_TO_TERMINATION[tiebreaker]

    # Direct mapping
    if victory_type in VICTORY_TYPE_TO_TERMINATION:
        return VICTORY_TYPE_TO_TERMINATION[victory_type]

    # Check winner_reason field (some schemas use this)
    winner_reason = game.get("winner_reason", "").lower()
    if winner_reason in VICTORY_TYPE_TO_TERMINATION:
        return VICTORY_TYPE_TO_TERMINATION[winner_reason]

    # If we have a winner but unknown termination, mark as unknown
    if game.get("winner") is not None:
        return "status:completed:unknown"

    return "status:unknown"


def upgrade_game(game: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a single game record to new schema."""
    upgraded = game.copy()

    # Add/update board_type
    if "board_type" not in upgraded or upgraded.get("board_type") == "unknown":
        upgraded["board_type"] = infer_board_type(game)

    # Add/update termination_reason
    if "termination_reason" not in upgraded or not upgraded["termination_reason"].startswith("status:"):
        upgraded["termination_reason"] = infer_termination_reason(game)

    # Ensure num_players is present
    if "num_players" not in upgraded:
        if "config" in game and "num_players" in game["config"]:
            upgraded["num_players"] = game["config"]["num_players"]
        elif "moves" in game and game["moves"]:
            # Infer from max player number in moves
            max_player = max(m.get("player", 1) for m in game["moves"] if "player" in m)
            upgraded["num_players"] = max_player

    # Add schema version marker
    upgraded["_schema_version"] = "v2"
    upgraded["_upgraded_from"] = "v1"

    return upgraded


def process_file(input_path: Path, output_path: Path, stats: Counter) -> int:
    """Process a single JSONL file."""
    upgraded_count = 0

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                game = json.loads(line)

                # Check if upgrade needed
                needs_upgrade = (
                    "board_type" not in game or
                    game.get("board_type") == "unknown" or
                    "termination_reason" not in game or
                    not game.get("termination_reason", "").startswith("status:")
                )

                if needs_upgrade:
                    game = upgrade_game(game)
                    upgraded_count += 1
                    stats["upgraded"] += 1
                else:
                    stats["already_current"] += 1

                # Track results
                stats[f"board:{game.get('board_type', 'unknown')}"] += 1
                stats[f"term:{game.get('termination_reason', 'unknown')}"] += 1

                f_out.write(json.dumps(game) + '\n')
                stats["total"] += 1

            except json.JSONDecodeError as e:
                stats["parse_errors"] += 1
                print(f"  Warning: JSON parse error at {input_path}:{line_num}: {e}", file=sys.stderr)
            except Exception as e:
                stats["other_errors"] += 1
                print(f"  Warning: Error at {input_path}:{line_num}: {e}", file=sys.stderr)

    return upgraded_count


def main():
    parser = argparse.ArgumentParser(description="Upgrade JSONL schema to current format")
    parser.add_argument("--input", "-i", type=Path, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file")
    parser.add_argument("--input-dir", type=Path, help="Input directory (process all .jsonl files)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for upgraded files")
    parser.add_argument("--in-place", action="store_true", help="Modify files in place (backup created)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate args
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input-dir")
    if args.input and not args.output and not args.in_place and not args.dry_run:
        parser.error("Must specify --output, --in-place, or --dry-run with --input")
    if args.input_dir and not args.output_dir and not args.in_place and not args.dry_run:
        parser.error("Must specify --output-dir, --in-place, or --dry-run with --input-dir")

    stats = Counter()
    files_to_process = []

    # Collect files to process
    if args.input:
        files_to_process.append((args.input, args.output or args.input))
    else:
        for jsonl_file in args.input_dir.rglob("*.jsonl"):
            if args.output_dir:
                rel_path = jsonl_file.relative_to(args.input_dir)
                output_file = args.output_dir / rel_path
            else:
                output_file = jsonl_file
            files_to_process.append((jsonl_file, output_file))

    print(f"Processing {len(files_to_process)} file(s)...")

    for input_path, output_path in files_to_process:
        if args.dry_run:
            print(f"  Would process: {input_path} -> {output_path}")
            continue

        # Create output directory if needed
        if args.output_dir:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle in-place with backup
        if args.in_place:
            backup_path = input_path.with_suffix(".jsonl.bak")
            if not backup_path.exists():
                import shutil
                shutil.copy2(input_path, backup_path)
            output_path = input_path

        if args.verbose:
            print(f"  Processing: {input_path}")

        upgraded = process_file(input_path, output_path, stats)

        if args.verbose and upgraded > 0:
            print(f"    Upgraded {upgraded} games")

    # Print summary
    print("\n=== Upgrade Summary ===")
    print(f"Total games processed: {stats['total']}")
    print(f"Games upgraded: {stats['upgraded']}")
    print(f"Already current schema: {stats['already_current']}")
    print(f"Parse errors: {stats['parse_errors']}")
    print(f"Other errors: {stats['other_errors']}")

    print("\n=== Board Type Distribution ===")
    for key, count in sorted(stats.items()):
        if key.startswith("board:"):
            print(f"  {count:>8}  {key[6:]}")

    print("\n=== Termination Reason Distribution ===")
    for key, count in sorted(stats.items()):
        if key.startswith("term:"):
            print(f"  {count:>8}  {key[5:]}")


if __name__ == "__main__":
    main()
