#!/usr/bin/env python3
"""
Replay games through the GameEngine to infer termination reason.

This script handles JSONL files where games have board_size but are missing
victory_type or termination_reason. It replays each game through the Python
GameEngine to determine the actual termination condition.

Usage:
    python scripts/replay_infer_termination.py --input old.jsonl --output upgraded.jsonl
    python scripts/replay_infer_termination.py --input-dir data/old/ --output-dir data/upgraded/
    python scripts/replay_infer_termination.py --input-dir data/ --in-place

Requirements:
    - Python GameEngine must be available in the path
    - Games must have 'moves' field with replay data
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.game_engine import GameEngine
from app.models import BoardType, GamePhase

# Board size to BoardType mapping
BOARD_SIZE_TO_TYPE = {
    8: BoardType.SQUARE8,
    19: BoardType.SQUARE19,
    25: BoardType.SQUARE8,  # Fallback, may need adjustment
    11: BoardType.HEXAGONAL,
    13: BoardType.HEXAGONAL,
    15: BoardType.HEXAGONAL,
}

BOARD_TYPE_STR_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}


@dataclass
class ReplayResult:
    """Result of replaying a game."""
    success: bool
    termination_reason: str | None = None
    winner: int | None = None
    final_phase: str | None = None
    error: str | None = None
    moves_replayed: int = 0


def get_board_type(game: dict[str, Any]) -> BoardType | None:
    """Get BoardType enum from game data."""
    # Try board_type string first
    if "board_type" in game:
        bt = game["board_type"].lower()
        if bt in BOARD_TYPE_STR_MAP:
            return BOARD_TYPE_STR_MAP[bt]

    # Try config
    if "config" in game and "board_type" in game["config"]:
        bt = game["config"]["board_type"].lower()
        if bt in BOARD_TYPE_STR_MAP:
            return BOARD_TYPE_STR_MAP[bt]

    # Fall back to board_size
    if "board_size" in game:
        size = game["board_size"]
        if size in BOARD_SIZE_TO_TYPE:
            return BOARD_SIZE_TO_TYPE[size]
        # Default mapping for unknown sizes
        if size <= 10:
            return BoardType.SQUARE8
        elif size <= 20:
            return BoardType.SQUARE19

    return None


def get_num_players(game: dict[str, Any]) -> int:
    """Get number of players from game data."""
    if "num_players" in game:
        return game["num_players"]
    if "config" in game and "num_players" in game["config"]:
        return game["config"]["num_players"]
    # Infer from moves
    if game.get("moves"):
        max_player = max(
            (m.get("player", 1) for m in game["moves"] if isinstance(m, dict) and "player" in m),
            default=2
        )
        return max_player
    return 2


def convert_move_format(move: dict[str, Any], engine: GameEngine) -> dict[str, Any] | None:
    """Convert JSONL move format to GameEngine format."""
    move_type = move.get("type", "")

    # Map move types
    if move_type == "place_ring":
        to = move.get("to", {})
        return {
            "type": "place_ring",
            "position": {"x": to.get("x", 0), "y": to.get("y", 0)}
        }

    elif move_type == "move_stack":
        from_pos = move.get("from", {})
        to_pos = move.get("to", {})
        return {
            "type": "move_stack",
            "from": {"x": from_pos.get("x", 0), "y": from_pos.get("y", 0)},
            "to": {"x": to_pos.get("x", 0), "y": to_pos.get("y", 0)}
        }

    elif move_type == "overtaking_capture":
        from_pos = move.get("from", {})
        to_pos = move.get("to", {})
        return {
            "type": "overtaking_capture",
            "from": {"x": from_pos.get("x", 0), "y": from_pos.get("y", 0)},
            "to": {"x": to_pos.get("x", 0), "y": to_pos.get("y", 0)}
        }

    elif move_type == "continue_capture_segment":
        to_pos = move.get("to", {})
        return {
            "type": "continue_capture_segment",
            "to": {"x": to_pos.get("x", 0), "y": to_pos.get("y", 0)}
        }

    elif move_type == "skip_capture":
        return {"type": "skip_capture"}

    elif move_type in ("no_line_action", "no_territory_action", "no_placement_action"):
        return {"type": move_type}

    elif move_type == "process_line":
        return {
            "type": "process_line",
            "line_id": move.get("line_id", move.get("lineId", 0))
        }

    elif move_type == "choose_territory_option":
        return {
            "type": "choose_territory_option",
            "option_index": move.get("option_index", move.get("optionIndex", 0))
        }

    elif move_type == "swap_sides":
        return {"type": "swap_sides"}

    elif move_type == "recovery_slide":
        from_pos = move.get("from", {})
        to_pos = move.get("to", {})
        return {
            "type": "recovery_slide",
            "from": {"x": from_pos.get("x", 0), "y": from_pos.get("y", 0)},
            "to": {"x": to_pos.get("x", 0), "y": to_pos.get("y", 0)}
        }

    elif move_type == "forced_elimination":
        # This is an event, not a move - skip it
        return None

    # Unknown move type - try to pass through
    return move


def determine_termination_reason(engine: GameEngine) -> str:
    """Determine termination reason from final game state."""
    state = engine.state

    # Check if game is over
    if state.phase == GamePhase.GAME_OVER:
        winner = state.winner

        # Check victory conditions
        # 1. Check for elimination (only one player has rings)
        players_with_rings = sum(1 for p in range(1, state.num_players + 1)
                                  if engine.get_player_ring_count(p) > 0)

        if players_with_rings <= 1:
            return "status:completed:elimination"

        # 2. Check for LPS (last player standing - others eliminated from game)
        active_players = sum(1 for p in range(1, state.num_players + 1)
                            if not getattr(state, 'eliminated_players', set()) or p not in state.eliminated_players)

        if active_players <= 1:
            return "status:completed:lps"

        # 3. Check for territory victory
        if hasattr(state, 'territory_owner') and state.territory_owner:
            territory_counts = Counter(state.territory_owner.values())
            if territory_counts:
                return "status:completed:territory"

        # 4. Stalemate/timeout
        if state.move_count >= state.max_moves:
            return "status:completed:timeout"

        # Default: if we have a winner but unclear why
        if winner is not None:
            return "status:completed:unknown"

    # Game not over - check if stuck
    if state.move_count >= state.max_moves:
        return "status:completed:timeout"

    return "status:incomplete"


def replay_game(game: dict[str, Any], verbose: bool = False) -> ReplayResult:
    """Replay a game through the GameEngine to determine termination."""
    board_type = get_board_type(game)
    if board_type is None:
        return ReplayResult(success=False, error="Cannot determine board type")

    num_players = get_num_players(game)
    moves = game.get("moves", [])

    if not moves:
        return ReplayResult(success=False, error="No moves in game")

    try:
        # Initialize engine
        engine = GameEngine(board_type=board_type, num_players=num_players)

        moves_replayed = 0
        for i, move in enumerate(moves):
            if not isinstance(move, dict):
                continue

            # Convert move format
            converted = convert_move_format(move, engine)
            if converted is None:
                # Skip non-move events (like forced_elimination)
                continue

            # Try to apply the move
            try:
                result = engine.apply_move(converted)
                moves_replayed += 1

                # Check if game ended
                if engine.state.phase == GamePhase.GAME_OVER:
                    break

            except Exception as e:
                if verbose:
                    print(f"  Move {i} failed: {e}", file=sys.stderr)
                # Try to continue anyway
                continue

        # Determine termination
        termination = determine_termination_reason(engine)
        winner = engine.state.winner

        return ReplayResult(
            success=True,
            termination_reason=termination,
            winner=winner,
            final_phase=str(engine.state.phase),
            moves_replayed=moves_replayed
        )

    except Exception as e:
        return ReplayResult(success=False, error=str(e))


def needs_replay(game: dict[str, Any]) -> bool:
    """Check if a game needs replay to determine termination."""
    # Has valid termination_reason in new format
    if "termination_reason" in game:
        tr = game["termination_reason"]
        if tr and tr.startswith("status:completed:"):
            return False

    # Has valid victory_type in old format
    if "victory_type" in game:
        vt = game["victory_type"]
        if vt and vt not in ("unknown", "", None):
            return False

    return True


def process_file(
    input_path: Path,
    output_path: Path,
    stats: Counter,
    verbose: bool = False,
    dry_run: bool = False
) -> None:
    """Process a single JSONL file."""
    games_to_write = []

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                game = json.loads(line)
                stats["total"] += 1

                if needs_replay(game):
                    if dry_run:
                        stats["needs_replay"] += 1
                        games_to_write.append(game)
                        continue

                    # Replay the game
                    result = replay_game(game, verbose=verbose)

                    if result.success:
                        # Update game with inferred data
                        game["termination_reason"] = result.termination_reason
                        if result.winner is not None:
                            game["winner"] = result.winner
                        game["_inferred_from_replay"] = True
                        stats["replayed_success"] += 1
                        stats[f"inferred:{result.termination_reason}"] += 1
                    else:
                        stats["replay_failed"] += 1
                        if verbose:
                            print(f"  Replay failed at {input_path}:{line_num}: {result.error}", file=sys.stderr)
                else:
                    stats["already_has_termination"] += 1

                games_to_write.append(game)

            except json.JSONDecodeError as e:
                stats["parse_errors"] += 1
                if verbose:
                    print(f"  JSON error at {input_path}:{line_num}: {e}", file=sys.stderr)
            except Exception as e:
                stats["other_errors"] += 1
                if verbose:
                    print(f"  Error at {input_path}:{line_num}: {e}", file=sys.stderr)

    # Write output
    if not dry_run and games_to_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for game in games_to_write:
                f.write(json.dumps(game) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Replay games to infer termination reason",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", type=Path, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file")
    parser.add_argument("--input-dir", type=Path, help="Input directory (process all .jsonl files)")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--in-place", action="store_true", help="Modify files in place (backup created)")
    parser.add_argument("--dry-run", action="store_true", help="Count games needing replay without processing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")

    args = parser.parse_args()

    # Validate args
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input-dir")
    if args.input and not args.output and not args.in_place and not args.dry_run:
        parser.error("Must specify --output, --in-place, or --dry-run")
    if args.input_dir and not args.output_dir and not args.in_place and not args.dry_run:
        parser.error("Must specify --output-dir, --in-place, or --dry-run")

    stats = Counter()
    files_to_process = []

    # Collect files
    if args.input:
        files_to_process.append((args.input, args.output or args.input))
    else:
        for jsonl_file in sorted(args.input_dir.rglob("*.jsonl")):
            if args.output_dir:
                rel_path = jsonl_file.relative_to(args.input_dir)
                output_file = args.output_dir / rel_path
            else:
                output_file = jsonl_file
            files_to_process.append((jsonl_file, output_file))

    if args.limit:
        files_to_process = files_to_process[:args.limit]

    print(f"Processing {len(files_to_process)} file(s)...")
    start_time = time.time()

    for i, (input_path, output_path) in enumerate(files_to_process):
        if args.verbose or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files_to_process)}] {input_path.name}")

        if args.in_place and not args.dry_run:
            backup_path = input_path.with_suffix(".jsonl.bak")
            if not backup_path.exists():
                import shutil
                shutil.copy2(input_path, backup_path)
            output_path = input_path

        process_file(input_path, output_path, stats, args.verbose, args.dry_run)

    elapsed = time.time() - start_time

    # Print summary
    print("\n=== Replay Inference Summary ===")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Total games: {stats['total']}")
    print(f"Already had termination: {stats['already_has_termination']}")

    if args.dry_run:
        print(f"Games needing replay: {stats['needs_replay']}")
    else:
        print(f"Successfully replayed: {stats['replayed_success']}")
        print(f"Replay failed: {stats['replay_failed']}")
        print(f"Parse errors: {stats['parse_errors']}")
        print(f"Other errors: {stats['other_errors']}")

        print("\n=== Inferred Termination Reasons ===")
        for key, count in sorted(stats.items()):
            if key.startswith("inferred:"):
                print(f"  {count:>8}  {key[9:]}")


if __name__ == "__main__":
    main()
