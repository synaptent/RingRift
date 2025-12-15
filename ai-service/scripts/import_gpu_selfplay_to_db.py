#!/usr/bin/env python
"""Import GPU selfplay JSONL games into full GameReplayDB format.

This script converts GPU selfplay JSONL records (which contain full move data)
into the canonical GameReplayDB format with state snapshots, enabling parity
validation and training pipeline usage.

Usage:
    python scripts/import_gpu_selfplay_to_db.py \
        --input data/games/gpu_generated/square8_512.jsonl \
        --output data/games/gpu_square8_canonical.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine
from app.models import BoardType, GameState, Move, MoveType, Position
from app.training.generate_data import create_initial_state


def parse_position(pos_dict: Dict[str, Any]) -> Position:
    """Parse a position dict into a Position object."""
    return Position(
        x=pos_dict["x"],
        y=pos_dict["y"],
        z=pos_dict.get("z"),
    )


def parse_move(move_dict: Dict[str, Any], move_number: int, timestamp: str) -> Optional[Move]:
    """Parse a move dict into a Move object.

    Returns None for unknown or bookkeeping-only move types (e.g. unknown_6/NO_ACTION).
    """
    move_type_str = str(move_dict.get("type") or "").strip()
    if not move_type_str:
        raise ValueError("GPU move is missing required 'type' field")

    # Skip unknown_* move types (internal GPU bookkeeping, e.g. NO_ACTION=6)
    if move_type_str.startswith("unknown_"):
        return None

    # GPU JSONLs may contain both canonical move types and legacy/internal
    # bookkeeping types (e.g. line_formation / territory_claim). We parse them
    # into the shared MoveType enum and decide later whether to skip them.
    try:
        move_type = MoveType(move_type_str)
    except Exception as exc:
        raise ValueError(f"Unknown MoveType for GPU import: {move_type_str!r}") from exc

    # Parse positions
    from_pos = parse_position(move_dict["from"]) if "from" in move_dict else None
    to_pos = parse_position(move_dict["to"]) if "to" in move_dict else None
    capture_target_dict = move_dict.get("capture_target") or move_dict.get("captureTarget")
    capture_target = (
        parse_position(capture_target_dict)
        if isinstance(capture_target_dict, dict)
        else None
    )

    return Move(
        id=f"move-{move_number}",
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        timestamp=timestamp,
        thinkTime=move_dict.get("think_time_ms", 0),
        moveNumber=move_number,
    )


def get_board_type(board_str: str) -> BoardType:
    """Convert board type string to BoardType enum."""
    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        # Legacy alias used by early GPU/selfplay outputs.
        "square25": BoardType.HEXAGONAL,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
    }
    return board_map.get(board_str, BoardType.SQUARE8)


def _find_matching_candidate_move(
    candidates: List[Move],
    *,
    desired_type: MoveType,
    from_pos: Optional[Position],
    to_pos: Optional[Position],
    must_move_from_key: Optional[str] = None,
) -> Optional[Move]:
    from_key = from_pos.to_key() if from_pos else None
    to_key = to_pos.to_key() if to_pos else None
    if to_key is None and desired_type == MoveType.SKIP_CAPTURE:
        # Canonical skip_capture moves use a placeholder `to` position.
        to_key = Position(x=0, y=0).to_key()

    for candidate in candidates:
        if candidate.type != desired_type:
            continue

        cand_from = candidate.from_pos.to_key() if candidate.from_pos else None
        cand_to = candidate.to.to_key() if candidate.to else None

        # Handle implicit from position (after placement, CPU moves have from=None
        # because from is implicit from must_move_from_stack_key)
        if cand_from is None and from_key is not None:
            # If candidate has no from but GPU has explicit from, check if it matches
            # the must_move_from constraint or just match by destination
            if must_move_from_key is not None and must_move_from_key != from_key:
                continue  # GPU from doesn't match the constrained stack
            # If must_move_from_key matches or there's no constraint, continue to to matching
        elif cand_from != from_key:
            continue

        if cand_to != to_key:
            continue
        return candidate

    return None


def expand_gpu_jsonl_moves_to_canonical(
    gpu_moves: List[Move],
    initial_state: GameState,
) -> tuple[List[Move], GameState]:
    """Expand GPU JSONL move histories into canonical phase-recorded moves.

    GPU JSONLs omit explicit phase-bookkeeping and decision moves that are
    required by RR-CANON-R075. This helper advances the CPU engine through:
    - line/territory decision phases (auto-selecting the first option),
    - required no_*_action bookkeeping moves,
    - implicit skip_capture and skip_placement moves when the GPU stream
      advances past those phases without recording them.
    """
    from app.models.core import GamePhase

    expanded: List[Move] = []
    state = initial_state
    move_num = 0

    def stamp(m: Move, ts) -> Move:
        nonlocal move_num
        move_num += 1
        return m.model_copy(
            update={
                "id": f"move-{move_num}",
                "timestamp": ts,
                "think_time": 0,
                "move_number": move_num,
            }
        )

    def apply(m: Move, ts) -> None:
        nonlocal state
        stamped = stamp(m, ts)
        expanded.append(stamped)
        state = GameEngine.apply_move(state, stamped)

    def auto_line(ts) -> None:
        player = state.current_player
        line_moves = GameEngine._get_line_processing_moves(state, player)
        if line_moves:
            choose_moves = [m for m in line_moves if m.type == MoveType.CHOOSE_LINE_OPTION]
            process_moves = [m for m in line_moves if m.type == MoveType.PROCESS_LINE]
            picked = choose_moves[0] if choose_moves else process_moves[0]
            apply(picked, ts)
            return
        apply(
            Move(
                id="no-line-action",
                type=MoveType.NO_LINE_ACTION,
                player=player,
                timestamp=ts,
                think_time=0,
                move_number=0,
            ),
            ts,
        )

    def auto_territory(ts) -> None:
        player = state.current_player
        terr_moves = GameEngine._get_territory_processing_moves(state, player)
        if terr_moves:
            apply(terr_moves[0], ts)
            return
        apply(
            Move(
                id="no-territory-action",
                type=MoveType.NO_TERRITORY_ACTION,
                player=player,
                timestamp=ts,
                think_time=0,
                move_number=0,
            ),
            ts,
        )

    for gpu_move in gpu_moves:
        ts = gpu_move.timestamp
        # GPU JSONL streams sometimes include legacy/internal bookkeeping moves
        # (line_formation, territory_claim) that do not map 1:1 to the canonical
        # per-phase move surface. Canonicalisation is driven by the CPU engine's
        # phase machine, so we ignore these markers here.
        if gpu_move.type in {MoveType.LINE_FORMATION, MoveType.TERRITORY_CLAIM}:
            continue

        # Advance through any implied phases until the GPU move becomes applicable.
        safety = 0
        while state.game_status.value == "active" and safety < 500:
            safety += 1
            phase = state.current_phase
            player = state.current_player

            if phase == GamePhase.LINE_PROCESSING:
                auto_line(ts)
                continue
            if phase == GamePhase.TERRITORY_PROCESSING:
                auto_territory(ts)
                continue

            if phase == GamePhase.CAPTURE and gpu_move.type not in {
                MoveType.OVERTAKING_CAPTURE,
                MoveType.CONTINUE_CAPTURE_SEGMENT,
                MoveType.SKIP_CAPTURE,
            }:
                apply(
                    Move(
                        id="skip-capture",
                        type=MoveType.SKIP_CAPTURE,
                        player=player,
                        timestamp=ts,
                        think_time=0,
                        move_number=0,
                    ),
                    ts,
                )
                continue

            if phase == GamePhase.RING_PLACEMENT and gpu_move.type != MoveType.PLACE_RING:
                # Prefer skip_placement when available to advance to movement.
                candidates = GameEngine.get_valid_moves(state, player)
                skip_moves = [m for m in candidates if m.type == MoveType.SKIP_PLACEMENT]
                if skip_moves:
                    apply(skip_moves[0], ts)
                    continue

            requirement = GameEngine.get_phase_requirement(state, player)
            if requirement is not None:
                synthesized = GameEngine.synthesize_bookkeeping_move(requirement, state)
                apply(synthesized, ts)
                continue

            break

        if safety >= 500:
            raise RuntimeError("Exceeded safety limit while advancing phases for GPU import")

        # In CHAIN_CAPTURE, GPU records segments as overtaking_capture but the
        # canonical engine expects continue_capture_segment.
        desired_type = gpu_move.type
        if desired_type == MoveType.CHAIN_CAPTURE:
            desired_type = MoveType.CONTINUE_CAPTURE_SEGMENT
        if desired_type == MoveType.OVERTAKING_CAPTURE and state.current_phase == GamePhase.CHAIN_CAPTURE:
            desired_type = MoveType.CONTINUE_CAPTURE_SEGMENT

        candidates = GameEngine.get_valid_moves(state, state.current_player)
        matched = _find_matching_candidate_move(
            candidates,
            desired_type=desired_type,
            from_pos=gpu_move.from_pos,
            to_pos=gpu_move.to,
            must_move_from_key=state.must_move_from_stack_key,
        )
        if matched is None:
            raise RuntimeError(
                "No matching candidate move "
                f"(gpu_move_number={gpu_move.move_number}, gpu_player={gpu_move.player}, "
                f"state_player={state.current_player}) "
                f"for gpu={gpu_move.type.value} desired={desired_type.value} "
                f"from={gpu_move.from_pos} to={gpu_move.to} phase={state.current_phase.value}"
            )

        # If the matched move has no from_pos but the GPU move does, copy it over.
        # This handles the case where CPU generates moves with implicit from (after placement).
        if matched.from_pos is None and gpu_move.from_pos is not None:
            matched = matched.model_copy(update={"from_pos": gpu_move.from_pos})

        apply(matched, ts)

    # Flush any remaining decision/bookkeeping phases after the last GPU move.
    safety = 0
    while state.game_status.value == "active" and safety < 500:
        safety += 1
        phase = state.current_phase
        player = state.current_player
        ts = state.last_move_at

        if phase == GamePhase.CAPTURE:
            apply(
                Move(
                    id="skip-capture",
                    type=MoveType.SKIP_CAPTURE,
                    player=player,
                    timestamp=ts,
                    think_time=0,
                    move_number=0,
                ),
                ts,
            )
            continue
        if phase == GamePhase.LINE_PROCESSING:
            auto_line(ts)
            continue
        if phase == GamePhase.TERRITORY_PROCESSING:
            auto_territory(ts)
            continue

        requirement = GameEngine.get_phase_requirement(state, player)
        if requirement is not None:
            synthesized = GameEngine.synthesize_bookkeeping_move(requirement, state)
            apply(synthesized, ts)
            continue

        break

    return expanded, state


def import_game(
    db: GameReplayDB,
    game_record: Dict[str, Any],
    source_file: str,
) -> bool:
    """Import a single game record into the database.

    Returns True on success, False on failure.
    """
    game_id = game_record.get("game_id", "unknown")
    board_type_str = game_record.get("board_type", "square8")
    board_type = get_board_type(board_type_str)
    num_players = game_record.get("num_players", 2)

    # Get initial state - either from record or create default
    initial_state_dict = game_record.get("initial_state")
    if initial_state_dict:
        # Reconstruct initial state from dict (Pydantic v1 API)
        try:
            initial_state = GameState.parse_obj(initial_state_dict)
        except Exception as e:
            print(f"  Warning: Failed to parse initial_state for {game_id}: {e}")
            return False
    else:
        initial_state = create_initial_state(
            board_type=board_type,
            num_players=num_players,
        )

    # Parse moves
    moves_data = game_record.get("moves", [])
    timestamp = game_record.get("timestamp", datetime.now().isoformat())

    moves: List[Move] = []
    for i, move_dict in enumerate(moves_data):
        try:
            move = parse_move(move_dict, i + 1, timestamp)
            if move is not None:  # Skip unknown/bookkeeping move types
                moves.append(move)
        except Exception as e:
            print(f"  Warning: Failed to parse move {i} in {game_id}: {e}")
            return False

    try:
        canonical_moves, final_state = expand_gpu_jsonl_moves_to_canonical(moves, initial_state)
    except Exception as e:
        print(f"  Warning: Failed to canonicalize moves for {game_id}: {e}")
        return False

    # Prepare metadata
    metadata = {
        "source": f"gpu_import:{source_file}",
        "original_game_id": game_record.get("game_id"),
        "termination_reason": game_record.get("termination_reason", game_record.get("victory_type")),
        "victory_type": game_record.get("victory_type"),
        "engine_mode": game_record.get("engine_mode", "gpu_heuristic"),
        "batch_id": game_record.get("batch_id"),
        "device": game_record.get("device"),
        "gpu_move_count": len(moves),
        "canonical_move_count": len(canonical_moves),
    }

    # Store the game
    try:
        db.store_game(
            game_id=f"gpu_{board_type_str}_{game_id}",
            initial_state=initial_state,
            final_state=final_state,
            moves=canonical_moves,
            metadata=metadata,
            store_history_entries=True,
        )
        return True
    except Exception as e:
        print(f"  Warning: Failed to store game {game_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Import GPU selfplay JSONL games into GameReplayDB format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output SQLite DB path"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=0,
        help="Maximum number of games to import (0 = all)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output database
    db = GameReplayDB(args.output)
    source_file = Path(args.input).stem

    print(f"Importing games from {args.input} to {args.output}")

    games_imported = 0
    games_failed = 0

    with open(args.input, "r") as f:
        for line_num, line in enumerate(f, 1):
            if args.limit > 0 and games_imported >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                games_failed += 1
                continue

            if import_game(db, record, source_file):
                games_imported += 1
                if games_imported % 50 == 0:
                    print(f"  Imported {games_imported} games...")
            else:
                games_failed += 1

    print(f"\nImport complete:")
    print(f"  Successfully imported: {games_imported}")
    print(f"  Failed: {games_failed}")
    print(f"  Output: {args.output}")

    # Verify the database
    count = db.get_game_count()
    print(f"  Total games in DB: {count}")

    sys.exit(0 if games_imported > 0 else 1)


if __name__ == "__main__":
    main()
