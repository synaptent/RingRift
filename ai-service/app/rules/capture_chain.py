from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Union

from app.models import (
    GameState,
    Move,
    Position,
    BoardType,
    BoardState,
    MoveType,
)
from app.board_manager import BoardManager
from .geometry import BoardGeometry


@dataclass
class PyChainCaptureSegment:
    """
    Python analogue of TS ChainCaptureSegment.

    Records one capture segment within a chain.
    """
    from_pos: Position
    target_pos: Position
    landing_pos: Position
    captured_cap_height: int


@dataclass
class PyChainCaptureStateSnapshot:
    """
    Lightweight snapshot for capture-chain enumeration.

    Mirrors TS ChainCaptureStateSnapshot.
    """
    player: int
    current_position: Position
    captured_this_chain: List[Position]


@dataclass
class PyChainCaptureEnumerationOptions:
    """
    Enumeration options for chain-capture segments.

    disallow_revisited_targets is intended for analysis/search tools only
    and must remain False for rules-level enumeration.
    """
    disallow_revisited_targets: bool = False
    move_number: Optional[int] = None
    kind: Literal["initial", "continuation"] = "continuation"


@dataclass
class PyChainCaptureContinuationInfo:
    """
    Python analogue of TS ChainCaptureContinuationInfo.
    """
    must_continue: bool
    available_continuations: List[Move]


@dataclass
class PyCaptureSegmentParams:
    """
    Parameters for applying a single capture segment.
    """
    from_pos: Position
    target_pos: Position
    landing_pos: Position
    player: int


@dataclass
class PyCaptureApplicationOutcome:
    """
    Result of applying a capture segment.
    """
    next_state: GameState
    rings_transferred: int
    chain_continuation_required: bool


def _sign(value: int) -> int:
    if value == 0:
        return 0
    return 1 if value > 0 else -1


def validate_capture_segment_on_board_py(
    board_type: BoardType,
    from_pos: Position,
    target_pos: Position,
    landing_pos: Position,
    player: int,
    board: BoardState,
) -> bool:
    """
    Canonical, board-local validator for a single capture segment.

    Mirrors the shared TS core.validateCaptureSegmentOnBoard helper and
    the Python GameEngine._validate_capture_segment_on_board_for_reachability
    semantics, with the additional explicit check that landing on an
    opponent marker is illegal.

    This helper is intentionally pure: it depends only on the provided
    board view and does not mutate it.
    """
    # Basic position validity
    if not BoardManager.is_valid_position(from_pos, board_type, board.size):
        return False
    if not BoardManager.is_valid_position(target_pos, board_type, board.size):
        return False
    if not BoardManager.is_valid_position(landing_pos, board_type, board.size):
        return False

    attacker = BoardManager.get_stack(from_pos, board)
    if not attacker or attacker.controlling_player != player:
        return False

    target_stack = BoardManager.get_stack(target_pos, board)
    if not target_stack:
        return False

    # Cap-height constraint: attacker.cap_height >= target.cap_height
    if attacker.cap_height < target_stack.cap_height:
        return False

    # Direction from attacker to target must lie on a legal capture ray.
    dx = target_pos.x - from_pos.x
    dy = target_pos.y - from_pos.y
    dz = (target_pos.z or 0) - (from_pos.z or 0)

    if board_type == BoardType.HEXAGONAL:
        coord_changes = sum(1 for d in (dx, dy, dz) if d != 0)
        if coord_changes != 2:
            return False
    else:
        if dx == 0 and dy == 0:
            return False
        if dx != 0 and dy != 0 and abs(dx) != abs(dy):
            return False

    # Path from attacker to target (exclusive) must be clear of stacks and
    # collapsed spaces; markers are allowed.
    path_to_target = BoardGeometry.get_path_positions(
        from_pos,
        target_pos,
    )[1:-1]
    for pos in path_to_target:
        if not BoardManager.is_valid_position(pos, board_type, board.size):
            return False
        if BoardManager.is_collapsed_space(pos, board):
            return False
        if BoardManager.get_stack(pos, board):
            return False

    # Landing must be beyond target in the same direction from from_pos.
    dx2 = landing_pos.x - from_pos.x
    dy2 = landing_pos.y - from_pos.y
    dz2 = (landing_pos.z or 0) - (from_pos.z or 0)

    if dx != 0 and _sign(dx) != _sign(dx2):
        return False
    if dy != 0 and _sign(dy) != _sign(dy2):
        return False
    if dz != 0 and _sign(dz) != _sign(dz2):
        return False

    dist_to_target = abs(dx) + abs(dy) + abs(dz)
    dist_to_landing = abs(dx2) + abs(dy2) + abs(dz2)
    if dist_to_landing <= dist_to_target:
        return False

    # Total distance must be at least the attacker's stack height.
    segment_distance = BoardGeometry.calculate_distance(
        board_type,
        from_pos,
        landing_pos,
    )
    if segment_distance < attacker.stack_height:
        return False

    # Path from target to landing (exclusive) must also be clear.
    path_from_target = BoardGeometry.get_path_positions(
        target_pos,
        landing_pos,
    )[1:-1]
    for pos in path_from_target:
        if not BoardManager.is_valid_position(pos, board_type, board.size):
            return False
        if BoardManager.is_collapsed_space(pos, board):
            return False
        if BoardManager.get_stack(pos, board):
            return False

    # Landing space must be empty (no stack) and not collapsed.
    if BoardManager.is_collapsed_space(landing_pos, board):
        return False
    if BoardManager.get_stack(landing_pos, board):
        return False

    # Per RR-CANON-R101/R102: landing on any marker (own or opponent) is legal.
    # The marker is removed and the top ring of the attacking stack's cap is eliminated.
    # Validation allows this; mutation handles the elimination cost.

    return True


def enumerate_capture_moves_py(
    state: GameState,
    player: int,
    from_pos: Position,
    *,
    move_number: Optional[int] = None,
    kind: Literal["initial", "continuation"] = "initial",
) -> List[Move]:
    """
    Enumerate all legal overtaking capture segments from a given origin.

    This is the Python analogue of TS CaptureAggregate.enumerateCaptureMoves.

    It walks each capture ray, finds the first capturable target stack with
    attacker.cap_height >= target.cap_height, and then enumerates all legal
    landing positions beyond that target, delegating the detailed segment
    legality to validate_capture_segment_on_board_py.
    """
    board = state.board
    board_type = board.type
    size = board.size

    attacker = BoardManager.get_stack(from_pos, board)
    if not attacker or attacker.controlling_player != player:
        return []

    move_num = (
        move_number
        if move_number is not None
        else len(state.move_history) + 1
    )
    moves: List[Move] = []
    directions = BoardManager._get_all_directions(board_type)

    for direction in directions:
        # Step 1: find the first potential target along this ray.
        step = 1
        target_pos: Optional[Position] = None

        while True:
            pos = BoardManager._add_direction(from_pos, direction, step)
            if not BoardManager.is_valid_position(pos, board_type, size):
                break
            if BoardManager.is_collapsed_space(pos, board):
                break

            stack_at_pos = BoardManager.get_stack(pos, board)
            if stack_at_pos and stack_at_pos.stack_height > 0:
                if attacker.cap_height >= stack_at_pos.cap_height:
                    target_pos = pos
                break

            step += 1

        if target_pos is None:
            continue

        # Step 2: enumerate possible landing positions beyond the target.
        landing_step = 1
        while True:
            landing_pos = BoardManager._add_direction(
                target_pos,
                direction,
                landing_step,
            )
            if not BoardManager.is_valid_position(
                landing_pos,
                board_type,
                size,
            ):
                break
            if BoardManager.is_collapsed_space(landing_pos, board):
                break
            if BoardManager.get_stack(landing_pos, board):
                break

            if validate_capture_segment_on_board_py(
                board_type,
                from_pos,
                target_pos,
                landing_pos,
                player,
                board,
            ):
                move_type = (
                    MoveType.OVERTAKING_CAPTURE
                    if kind == "initial"
                    else MoveType.CONTINUE_CAPTURE_SEGMENT
                )
                moves.append(
                    Move(
                        id=(
                            f"capture-"
                            f"{from_pos.to_key()}-"
                            f"{target_pos.to_key()}-"
                            f"{landing_pos.to_key()}-"
                            f"{move_num}"
                        ),
                        type=move_type,
                        player=player,
                        from_pos=from_pos,  # type: ignore[arg-type]
                        to=landing_pos,
                        capture_target=target_pos,  # type: ignore[arg-type]
                        timestamp=state.last_move_at,
                        thinkTime=0,
                        moveNumber=move_num,
                    )
                )

            landing_step += 1

    return moves


def enumerate_chain_capture_segments_py(
    state: GameState,
    snapshot: PyChainCaptureStateSnapshot,
    options: Optional[PyChainCaptureEnumerationOptions] = None,
) -> List[Move]:
    """
    Enumerate all legal chain-capture continuation segments from a snapshot.

    Mirrors TS CaptureAggregate.enumerateChainCaptureSegments:

    - Uses enumerate_capture_moves_py for geometry.
    - Optionally filters revisited targets when
      disallow_revisited_targets is set.
    - Normalises Move.type based on kind ("initial" vs "continuation").
    """
    opts = options or PyChainCaptureEnumerationOptions()
    move_num = (
        opts.move_number
        if opts.move_number is not None
        else len(state.move_history) + 1
    )

    base_moves = enumerate_capture_moves_py(
        state,
        snapshot.player,
        snapshot.current_position,
        move_number=move_num,
        kind=opts.kind,
    )

    if opts.disallow_revisited_targets and snapshot.captured_this_chain:
        visited_keys = {pos.to_key() for pos in snapshot.captured_this_chain}
        filtered: List[Move] = []
        for move in base_moves:
            if move.capture_target is None:
                filtered.append(move)
                continue
            if move.capture_target.to_key() not in visited_keys:
                filtered.append(move)
        return filtered

    return base_moves


def get_chain_capture_continuation_info_py(
    state: GameState,
    player: int,
    current_position: Position,
) -> PyChainCaptureContinuationInfo:
    """
    Convenience wrapper that answers "must this chain continue?" in a
    single call, returning both the boolean and the concrete segment list.

    Mirrors TS CaptureAggregate.getChainCaptureContinuationInfo.
    """
    snapshot = PyChainCaptureStateSnapshot(
        player=player,
        current_position=current_position,
        captured_this_chain=[],
    )
    segments = enumerate_chain_capture_segments_py(
        state,
        snapshot,
        PyChainCaptureEnumerationOptions(kind="continuation"),
    )
    return PyChainCaptureContinuationInfo(
        must_continue=len(segments) > 0,
        available_continuations=segments,
    )


def apply_capture_segment_py(
    state: GameState,
    params: PyCaptureSegmentParams,
) -> PyCaptureApplicationOutcome:
    """
    Apply a single capture segment and return the resulting state and
    whether further captures are available.

    This function delegates the actual mutation semantics to the canonical
    GameEngine.apply_move implementation to avoid duplicating capture
    mutation logic in multiple places. It is therefore pure with respect
    to the input GameState: the returned GameState is a new instance, and
    the input ``state`` is left unchanged.
    """
    from app.game_engine import GameEngine  # Local import to avoid cycles

    move_number = len(state.move_history) + 1
    move = Move(
        id=(
            f"capture-"
            f"{params.from_pos.to_key()}-"
            f"{params.target_pos.to_key()}-"
            f"{params.landing_pos.to_key()}-"
            f"{move_number}"
        ),
        type=MoveType.OVERTAKING_CAPTURE,
        player=params.player,
        from_pos=params.from_pos,  # type: ignore[arg-type]
        to=params.landing_pos,
        capture_target=params.target_pos,  # type: ignore[arg-type]
        timestamp=state.last_move_at,
        thinkTime=0,
        moveNumber=move_number,
    )

    next_state = GameEngine.apply_move(state, move)

    continuation_info = get_chain_capture_continuation_info_py(
        next_state,
        params.player,
        params.landing_pos,
    )

    return PyCaptureApplicationOutcome(
        next_state=next_state,
        rings_transferred=1,
        chain_continuation_required=continuation_info.must_continue,
    )


def apply_capture_py(
    state: GameState,
    move: Move,
) -> Tuple[bool, Optional[GameState], Union[List[Position], str]]:
    """
    High-level capture application helper, analogous to TS applyCapture.

    Returns a triple ``(success, next_state_or_none, continuation_or_error)``:

    - On success, ``success`` is True, ``next_state_or_none`` is the new
      GameState produced by applying the capture, and ``continuation_or_error``
      is a list of landing positions for any mandatory continuation segments.
    - On failure, ``success`` is False, ``next_state_or_none`` is None,
      and ``continuation_or_error`` is a human-readable error string.
    """
    if move.type not in (
        MoveType.OVERTAKING_CAPTURE,
        MoveType.CONTINUE_CAPTURE_SEGMENT,
        MoveType.CHAIN_CAPTURE,
    ):
        return (
            False,
            None,
            f"Expected overtaking or chain capture move, got {move.type}",
        )

    if move.from_pos is None or move.capture_target is None:
        return (
            False,
            None,
            "Move.from and Move.captureTarget are required for capture moves",
        )

    params = PyCaptureSegmentParams(
        from_pos=move.from_pos,
        target_pos=move.capture_target,
        landing_pos=move.to,
        player=move.player,
    )

    outcome = apply_capture_segment_py(state, params)

    continuation_info = get_chain_capture_continuation_info_py(
        outcome.next_state,
        move.player,
        move.to,
    )
    continuation_landings = [
        seg.to
        for seg in continuation_info.available_continuations
        if seg.to is not None
    ]

    return True, outcome.next_state, continuation_landings
