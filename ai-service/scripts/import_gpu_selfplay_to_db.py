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
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.game_replay import GameReplayDB
from app.game_engine import GameEngine, PhaseRequirementType
from app.models import BoardType, GameState, Move, MoveType, Position
from app.training.initial_state import create_initial_state

# Import canonical export module for GPU->canonical translation (December 2025)
try:
    from app.ai.gpu_canonical_export import (
        gpu_move_type_to_canonical,
        gpu_phase_to_canonical,
    )
    HAS_CANONICAL_EXPORT = True
except ImportError:
    HAS_CANONICAL_EXPORT = False
    gpu_move_type_to_canonical = None
    gpu_phase_to_canonical = None


def parse_position(pos_dict: dict[str, Any], board_type: BoardType = None) -> Position:
    """Parse a position dict into a Position object.

    For hexagonal boards, converts GPU array indices (0-24) to canonical
    axial coordinates (-12 to 12). GPU selfplay uses array-based indexing
    while the canonical game engine uses axial coordinates.

    Args:
        pos_dict: Position dict with 'x', 'y', and optional 'z' keys
        board_type: Board type (HEXAGONAL triggers coordinate conversion)

    Returns:
        Position with coordinates in canonical space
    """
    x = pos_dict["x"]
    y = pos_dict["y"]
    z = pos_dict.get("z")

    # Convert GPU array indices to canonical axial coordinates for hex boards.
    # GPU selfplay uses 0-24 array indexing, while canonical uses -12 to 12 axial.
    # Detection: if ANY coordinate is negative, it's already canonical.
    # Also check if z is provided and satisfies axial constraint (x+y+z=0).
    if board_type == BoardType.HEXAGONAL:
        HEX_RADIUS = 12

        # Already canonical if:
        # 1. Any coordinate is negative (axial coords can be -12 to 12)
        # 2. z is provided and satisfies axial constraint x+y+z=0
        is_already_canonical = (x < 0 or y < 0) or (z is not None and x + y + z == 0)

        if not is_already_canonical:
            # Assume GPU array indices (0-24) - convert to axial
            x = x - HEX_RADIUS
            y = y - HEX_RADIUS

        # Compute z from axial constraint: x + y + z = 0
        z = -x - y

    return Position(x=x, y=y, z=z)


def parse_move(
    move_dict: dict[str, Any],
    move_number: int,
    timestamp: str,
    board_type: BoardType = None,
    skip_coord_conversion: bool = False,
) -> Move | None:
    """Parse a move dict into a Move object.

    Returns None for unknown or bookkeeping-only move types (e.g. unknown_6/NO_ACTION).

    Args:
        move_dict: Move data from GPU JSONL
        move_number: Sequential move number
        timestamp: Move timestamp
        board_type: Board type for coordinate conversion (HEXAGONAL converts GPU→axial)
        skip_coord_conversion: If True, don't convert coordinates (data already canonical)
    """
    # Handle both canonical format ("type": "place_ring") and GPU format ("move_type": "PLACEMENT")
    move_type_str = str(move_dict.get("type") or move_dict.get("move_type") or "").strip()
    if not move_type_str:
        raise ValueError("GPU move is missing required 'type' field")

    # Skip unknown_* move types (internal GPU bookkeeping, e.g. NO_ACTION=6)
    if move_type_str.startswith("unknown_"):
        return None

    # GPU batch state exports uppercase enum names (e.g., "PLACEMENT", "MOVEMENT")
    # Map these to canonical lowercase values for MoveType parsing
    # December 2025: Use gpu_canonical_export module when available for consistency
    gpu_to_canonical = {
        "PLACEMENT": "place_ring",
        "SKIP_PLACEMENT": "skip_placement",
        "NO_PLACEMENT_ACTION": "no_placement_action",
        "MOVEMENT": "move_stack",
        "MOVE_RING": "move_ring",
        "NO_MOVEMENT_ACTION": "no_movement_action",
        "CAPTURE": "overtaking_capture",  # GPU uses generic CAPTURE for all captures
        "OVERTAKING_CAPTURE": "overtaking_capture",
        "CONTINUE_CAPTURE_SEGMENT": "continue_capture_segment",
        "SKIP_CAPTURE": "skip_capture",
        "NO_ACTION": "no_territory_action",  # GPU uses generic NO_ACTION for bookkeeping
        "CHOOSE_LINE_OPTION": "choose_line_option",
        "PROCESS_LINE": "process_line",
        "NO_LINE_ACTION": "no_line_action",
        "CHOOSE_TERRITORY_OPTION": "choose_territory_option",
        "PROCESS_TERRITORY_REGION": "process_territory_region",
        "NO_TERRITORY_ACTION": "no_territory_action",
        "SKIP_TERRITORY_PROCESSING": "skip_territory_processing",
        "ELIMINATE_RINGS_FROM_STACK": "eliminate_rings_from_stack",
        "FORCED_ELIMINATION": "forced_elimination",
        "RECOVERY_SLIDE": "recovery_slide",
        "SKIP_RECOVERY": "skip_recovery",
        # December 2025 additions from gpu_canonical_export
        "LINE_FORMATION": "process_line",
        "TERRITORY_CLAIM": "process_territory_region",
    }

    # Try using gpu_canonical_export module for GPU int enum values
    if HAS_CANONICAL_EXPORT and move_type_str.isdigit():
        canonical_str = gpu_move_type_to_canonical(int(move_type_str))
        if canonical_str != "unknown":
            move_type_str = canonical_str

    if move_type_str in gpu_to_canonical:
        move_type_str = gpu_to_canonical[move_type_str]

    # GPU JSONLs may contain both canonical move types and legacy/internal
    # bookkeeping types (e.g. line_formation / territory_claim). We parse them
    # into the shared MoveType enum and decide later whether to skip them.
    try:
        move_type = MoveType(move_type_str)
    except Exception as exc:
        raise ValueError(f"Unknown MoveType for GPU import: {move_type_str!r}") from exc

    # Handle both canonical format ("from": {"x": 1, "y": 2}) and GPU format ("from_pos": [y, x])
    from_data = move_dict.get("from") or move_dict.get("from_pos")
    to_data = move_dict.get("to") or move_dict.get("to_pos")

    # For coordinate parsing: skip conversion if data is already canonical
    effective_board_type = None if skip_coord_conversion else board_type

    def parse_pos_flexible(pos_data, btype):
        """Parse position from either dict or array format."""
        if pos_data is None:
            return None
        if isinstance(pos_data, dict):
            return parse_position(pos_data, btype)
        if isinstance(pos_data, (list, tuple)) and len(pos_data) >= 2:
            # GPU format: [y, x] - note the order!
            y, x = pos_data[0], pos_data[1]
            if y < 0 or x < 0:  # Invalid position marker
                return None
            return parse_position({"x": x, "y": y}, btype)
        return None

    from_pos = parse_pos_flexible(from_data, effective_board_type)
    to_pos = parse_pos_flexible(to_data, effective_board_type)
    capture_target_dict = move_dict.get("capture_target") or move_dict.get("captureTarget")
    capture_target = (
        parse_position(capture_target_dict, effective_board_type)
        if isinstance(capture_target_dict, dict)
        else None
    )

    # Parse placement_count for place_ring moves
    placement_count = move_dict.get("placement_count")

    return Move(
        id=f"move-{move_number}",
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        placement_count=placement_count,
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


def _fix_capture_target_from_board(move: Move, state: GameState) -> Move:
    """Fix captureTarget for a capture move by scanning the actual board state.

    GPU canonical export computes captureTarget as one step before landing, but this is
    incorrect when the attacker lands multiple steps past the target. The correct target
    is the first stack along the ray from from_pos toward to_pos.

    Args:
        move: The capture move with potentially incorrect capture_target
        state: Current game state with board information

    Returns:
        Move with corrected capture_target, or original move if no fix needed
    """
    from app.board_manager import BoardManager

    if not move.from_pos or not move.to:
        return move

    from_pos = move.from_pos
    to_pos = move.to
    board = state.board

    # Compute direction
    dy = 0 if to_pos.y == from_pos.y else (1 if to_pos.y > from_pos.y else -1)
    dx = 0 if to_pos.x == from_pos.x else (1 if to_pos.x > from_pos.x else -1)

    # Scan along the ray to find the first stack
    dist = max(abs(to_pos.y - from_pos.y), abs(to_pos.x - from_pos.x))

    for step in range(1, dist + 1):
        check_y = from_pos.y + dy * step
        check_x = from_pos.x + dx * step
        check_pos = Position(x=check_x, y=check_y)

        stack = BoardManager.get_stack(check_pos, board)
        if stack is not None:
            # Found the target - update move with correct capture_target
            return move.model_copy(update={"capture_target": check_pos})

    # No stack found along path - this shouldn't happen for valid captures
    # Return original move and let apply_move handle the error
    return move


def _find_matching_candidate_move(
    candidates: list[Move],
    *,
    desired_type: MoveType,
    from_pos: Position | None,
    to_pos: Position | None,
    must_move_from_key: str | None = None,
) -> Move | None:
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
    gpu_moves: list[Move],
    initial_state: GameState,
    verbose: bool = False,
) -> tuple[list[Move], GameState]:
    """Expand GPU JSONL move histories into canonical phase-recorded moves.

    GPU JSONLs omit explicit phase-bookkeeping and decision moves that are
    required by RR-CANON-R075. This helper advances the CPU engine through:
    - line/territory decision phases (auto-selecting the first option),
    - required no_*_action bookkeeping moves,
    - implicit skip_capture and skip_placement moves when the GPU stream
      advances past those phases without recording them.
    """
    from app.models.core import GamePhase

    expanded: list[Move] = []
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

    for gpu_idx, gpu_move in enumerate(gpu_moves):
        ts = gpu_move.timestamp
        if verbose:
            print(f"  [GPU {gpu_idx}] type={gpu_move.type.value} player={gpu_move.player} from={gpu_move.from_pos} to={gpu_move.to}")
            print(f"         state: phase={state.current_phase.value} player={state.current_player}")
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

            # LINE_PROCESSING: Use GPU move if it's a line move, otherwise auto-generate
            if phase == GamePhase.LINE_PROCESSING:
                if gpu_move.type in {MoveType.PROCESS_LINE, MoveType.CHOOSE_LINE_OPTION, MoveType.NO_LINE_ACTION}:
                    break  # Let the main matching logic handle it (including explicit NO_LINE_ACTION from new GPU format)
                auto_line(ts)
                continue
            # TERRITORY_PROCESSING: Use GPU move if it's a territory move, otherwise auto-generate
            if phase == GamePhase.TERRITORY_PROCESSING:
                if gpu_move.type in {
                    MoveType.CHOOSE_TERRITORY_OPTION,
                    MoveType.ELIMINATE_RINGS_FROM_STACK,
                    MoveType.SKIP_TERRITORY_PROCESSING,
                    MoveType.NO_TERRITORY_ACTION,  # New GPU format includes explicit bookkeeping moves
                }:
                    break  # Let the main matching logic handle it
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
                # Prefer skip_placement or no_placement_action when available to advance to movement.
                # This handles recovery-eligible players who have no rings to place.
                candidates = GameEngine.get_valid_moves(state, player)
                skip_moves = [m for m in candidates if m.type == MoveType.SKIP_PLACEMENT]
                if skip_moves:
                    apply(skip_moves[0], ts)
                    continue
                # Try no_placement_action for recovery-eligible players
                no_placement = [m for m in candidates if m.type == MoveType.NO_PLACEMENT_ACTION]
                if no_placement:
                    apply(no_placement[0], ts)
                    continue
                # Check phase requirement for no_placement_action
                req = GameEngine.get_phase_requirement(state, player)
                if req and req.type == PhaseRequirementType.NO_PLACEMENT_ACTION_REQUIRED:
                    synthesized = GameEngine.synthesize_bookkeeping_move(req, state)
                    apply(synthesized, ts)
                    continue

            if phase == GamePhase.FORCED_ELIMINATION and gpu_move.type != MoveType.FORCED_ELIMINATION:
                # FORCED_ELIMINATION phase - player had no actions this turn but controls stacks
                # Must apply explicit forced_elimination move (RR-CANON-R070/R072)
                fe_moves = GameEngine._get_forced_elimination_moves(state, player)
                if fe_moves:
                    apply(fe_moves[0], ts)
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

        # Handle bookkeeping moves directly via phase requirements (they're not in get_valid_moves)
        if desired_type in {MoveType.NO_LINE_ACTION, MoveType.NO_TERRITORY_ACTION, MoveType.NO_MOVEMENT_ACTION, MoveType.NO_PLACEMENT_ACTION}:
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                matched = GameEngine.synthesize_bookkeeping_move(req, state)
            else:
                matched = None
        else:
            candidates = GameEngine.get_valid_moves(state, state.current_player)
            matched = _find_matching_candidate_move(
                candidates,
                desired_type=desired_type,
                from_pos=gpu_move.from_pos,
                to_pos=gpu_move.to,
                must_move_from_key=state.must_move_from_stack_key,
            )
        if matched is None:
            if verbose:
                print(f"  [ERROR] No match for GPU move {gpu_idx}")
                print(f"         Candidates ({len(candidates)}): {[(c.type.value, c.from_pos, c.to) for c in candidates[:5]]}")
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
        if phase == GamePhase.FORCED_ELIMINATION:
            # FORCED_ELIMINATION phase - player had no actions this turn but controls stacks
            # Must apply explicit forced_elimination move (RR-CANON-R070/R072)
            fe_moves = GameEngine._get_forced_elimination_moves(state, player)
            if fe_moves:
                apply(fe_moves[0], ts)
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
    game_record: dict[str, Any],
    source_file: str,
    verbose: bool = False,
    skip_expansion: bool = False,
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

    # Parse moves with coordinate conversion for hex boards
    moves_data = game_record.get("moves", [])
    timestamp = game_record.get("timestamp", datetime.now().isoformat())

    # Detect if data is from random/CPU selfplay (already has canonical coordinates)
    # Random selfplay outputs canonical move types and coordinates - no conversion needed
    source = game_record.get("source", "")
    is_canonical_source = "random" in source.lower() or "run_random_selfplay" in source
    skip_coord_conversion = is_canonical_source

    moves: list[Move] = []
    for i, move_dict in enumerate(moves_data):
        try:
            move = parse_move(move_dict, i + 1, timestamp, board_type, skip_coord_conversion)
            if move is not None:  # Skip unknown/bookkeeping move types
                moves.append(move)
        except Exception as e:
            print(f"  Warning: Failed to parse move {i} in {game_id}: {e}")
            return False

    if skip_expansion:
        # Random/CPU selfplay already has canonical moves with bookkeeping - replay to get final state
        # GPU canonical export may have incorrect captureTarget (computed as one step back from landing),
        # so we fix it by scanning the actual board state to find the first stack along the ray.
        canonical_moves = []
        state = initial_state
        try:
            for move in moves:
                # Fix captureTarget for capture moves by scanning the board
                if move.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT) and move.from_pos and move.to:
                    fixed_move = _fix_capture_target_from_board(move, state)
                else:
                    fixed_move = move
                canonical_moves.append(fixed_move)
                state = GameEngine.apply_move(state, fixed_move)
            final_state = state
        except Exception as e:
            print(f"  Warning: Failed to replay moves for {game_id}: {e}")
            return False
    else:
        try:
            canonical_moves, final_state = expand_gpu_jsonl_moves_to_canonical(moves, initial_state, verbose=verbose)
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
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose debugging output for move canonicalization"
    )
    parser.add_argument(
        "--skip-expansion", action="store_true",
        help="Skip move expansion (use for random/CPU selfplay data that already has canonical moves)"
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

    with open(args.input) as f:
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

            if import_game(db, record, source_file, verbose=args.verbose, skip_expansion=args.skip_expansion):
                games_imported += 1
                if games_imported % 50 == 0:
                    print(f"  Imported {games_imported} games...")
            else:
                games_failed += 1

    print("\nImport complete:")
    print(f"  Successfully imported: {games_imported}")
    print(f"  Failed: {games_failed}")
    print(f"  Output: {args.output}")

    # Verify the database
    count = db.get_game_count()
    print(f"  Total games in DB: {count}")

    sys.exit(0 if games_imported > 0 else 1)


if __name__ == "__main__":
    main()
