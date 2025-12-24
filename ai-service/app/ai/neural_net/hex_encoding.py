"""Hex board action encoding/decoding for neural network policy heads.

This module provides the ActionEncoderHex class and related helper functions
for encoding and decoding moves on hexagonal boards into policy indices.

Migrated from _neural_net_legacy.py as part of Phase 2 modularization.
"""

from __future__ import annotations

from datetime import datetime

from ...models import (
    BoardState,
    BoardType,
    GameState,
    Move,
    MoveType,
    Position,
)
from ...rules.geometry import BoardGeometry
from .constants import (
    HEX_BOARD_SIZE,
    HEX_DIRS,
    INVALID_MOVE_INDEX,
    MAX_N,
    NUM_HEX_DIRS,
)


def _infer_board_size(board: BoardState | GameState) -> int:
    """
    Infer the canonical 2D board_size for CNN feature tensors.

    For SQUARE8: 8
    For SQUARE19: 19
    For HEXAGONAL: board.size (bounding box = 2*radius + 1)

    The returned value is the height/width of the (10, board_size, board_size)
    feature planes used by the CNN. Raises if the logical size exceeds MAX_N
    for square boards.
    """
    # Allow a GameState to be passed directly
    if isinstance(board, GameState):
        board = board.board

    board_type = board.type

    if board_type == BoardType.SQUARE8:
        return 8
    if board_type == BoardType.SQUARE19:
        return 19
    if board_type == BoardType.HEXAGONAL:
        # board.size is now the bounding box directly (25)
        return board.size
    if board_type == BoardType.HEX8:
        # board.size is now the bounding box directly (9)
        return board.size

    # Defensive fallback: use board.size but guard against unsupported sizes
    size = getattr(board, "size", 8)
    if size > MAX_N:
        raise ValueError(f"Unsupported board size {size}; MAX_N={MAX_N} is the current canonical maximum.")
    return int(size)


def _pos_from_key(pos_key: str) -> Position:
    """Parse a BoardState dict key like 'x,y' or 'x,y,z' into a Position."""
    parts = pos_key.split(",")
    if len(parts) == 2:
        x, y = int(parts[0]), int(parts[1])
        return Position(x=x, y=y)
    if len(parts) == 3:
        x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
        return Position(x=x, y=y, z=z)
    raise ValueError(f"Invalid position key: {pos_key!r}")


def _to_canonical_xy(board: BoardState, pos: Position) -> tuple[int, int]:
    """
    Map a logical Position on this board into canonical (cx, cy) in
    [0, board_size) × [0, board_size), where board_size depends on
    board.type and board.size.

    For SQUARE8/SQUARE19: return (pos.x, pos.y) directly.

    For HEXAGONAL:
      - Interpret pos.(x,y,z) as cube/axial coords where x,y lie in
        [-radius, radius].
      - Let radius = (board.size - 1) // 2.
      - Map x → cx = x + radius, y → cy = y + radius.
      - Return (cx, cy).
    """
    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return pos.x, pos.y

    if board.type in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Hex boards use cube coordinates where x,y in [-radius, radius]
        # Map to bounding box [0, 2*radius+1) via cx = x + radius
        # board.size is bounding box (2*radius + 1), so radius = (size - 1) // 2
        radius = (board.size - 1) // 2  # HEXAGONAL: 12, HEX8: 4
        cx = pos.x + radius
        cy = pos.y + radius
        return cx, cy

    # Fallback: treat as generic square coordinates.
    return pos.x, pos.y


def _from_canonical_xy(
    board: BoardState,
    cx: int,
    cy: int,
) -> Position | None:
    """
    Inverse of _to_canonical_xy.

    Returns a Position instance whose coordinates lie on this board, or None
    if (cx, cy) is outside [0, board_size) × [0, board_size).

    For HEXAGONAL:
      - radius = (board.size - 1) // 2
      - x = cx - radius, y = cy - radius, z = -x - y
    """
    board_size = _infer_board_size(board)
    if not (0 <= cx < board_size and 0 <= cy < board_size):
        return None

    if board.type in (BoardType.SQUARE8, BoardType.SQUARE19):
        return Position(x=cx, y=cy)

    if board.type in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Inverse of _to_canonical_xy for hex boards
        # Map from bounding box [0, 2*radius+1) back to cube coords [-radius, radius]
        # board.size is bounding box (2*radius + 1), so radius = (size - 1) // 2
        radius = (board.size - 1) // 2  # HEXAGONAL: 12, HEX8: 4
        x = cx - radius
        y = cy - radius
        z = -x - y
        return Position(x=x, y=y, z=z)

    # Fallback generic square position.
    return Position(x=cx, y=cy)


class ActionEncoderHex:
    """Hex-only action encoder for the canonical N=12 board.

    The concrete layout matches the design in AI_ARCHITECTURE.md:

      * Spatial frame: 25×25 canonical hex bounding box.
      * Placements (0 .. HEX_PLACEMENT_SPAN-1):
          index = (cy * 25 + cx) * 3 + (count - 1)
        where count ∈ {1,2,3} is the number of rings placed.

      * Movement / capture (HEX_MOVEMENT_BASE .. HEX_SPECIAL_BASE-1):
          index = HEX_MOVEMENT_BASE
                  + from_idx * (6 * HEX_MAX_DIST)
                  + dir_idx * HEX_MAX_DIST
                  + (dist - 1)
        where from_idx = from_cy * 25 + from_cx, dir_idx ∈ [0,6),
        dist ∈ [1,HEX_MAX_DIST]. This shared layout is used for MOVE_STACK,
        OVERTAKING_CAPTURE, and CONTINUE_CAPTURE_SEGMENT.

      * Special (HEX_SPECIAL_BASE):
          SKIP_PLACEMENT sentinel.

    Any decoded index that maps to a canonical cell outside the true hex
    (469-cell) region is treated as invalid and returns None.

    IMPORTANT (Dec 2025): This encoder validates that all encoded positions
    are within the valid hex region. Positions outside the hex (corners of
    the bounding box) return INVALID_MOVE_INDEX. This ensures compatibility
    with V3/V4 model architectures that mask invalid hex cells.
    """

    def __init__(
        self,
        board_size: int = HEX_BOARD_SIZE,
        policy_size: int = 0,
    ) -> None:
        # Spatial dimension of the hex bounding box (2N+1 for side N).
        self.board_size = board_size

        # Hex radius = (board_size - 1) // 2
        # For HEXAGONAL (25x25): radius = 12
        # For HEX8 (9x9): radius = 4
        self.hex_radius = (board_size - 1) // 2

        # Compute layout constants based on board_size
        self.max_dist = board_size - 1  # max distance on hex board
        self.placement_span = board_size * board_size * 3
        self.movement_base = self.placement_span
        self.movement_span = board_size * board_size * NUM_HEX_DIRS * self.max_dist
        self.special_base = self.movement_base + self.movement_span

        # Hex-specific action space dimension.
        self.policy_size = policy_size or (self.special_base + 1)

    def _is_valid_hex_cell(self, cx: int, cy: int) -> bool:
        """Check if canonical (cx, cy) is within the valid hex region.

        Uses axial distance formula: distance = max(|q|, |r|, |q + r|)
        where q = cx - center, r = cy - center.

        This matches the hex_mask used by V3/V4 neural net architectures,
        ensuring encoded policy indices align with model expectations.
        """
        center = self.board_size // 2
        q = cx - center
        r = cy - center
        hex_dist = max(abs(q), abs(r), abs(q + r))
        return hex_dist <= self.hex_radius

    def _encode_canonical_xy(
        self,
        board: BoardState,
        pos: Position,
    ) -> tuple[int, int] | None:
        """Return (cx, cy) in [0, board_size)×[0, board_size) or None if off-grid."""
        cx, cy = _to_canonical_xy(board, pos)
        if not (0 <= cx < self.board_size and 0 <= cy < self.board_size):
            return None
        return cx, cy

    def encode_move(self, move: Move, board: BoardState) -> int:
        """Map a hex move into a [0, policy_size) index.

        This encoder supports both HEXAGONAL (radius-12) and HEX8 (radius-4)
        board types. The board size must match the encoder's configured
        board_size. For non-hex geometries or size mismatches, the move is
        treated as unrepresentable and INVALID_MOVE_INDEX is returned.
        """
        if board.type not in (BoardType.HEXAGONAL, BoardType.HEX8):
            return INVALID_MOVE_INDEX

        # Verify board size matches encoder configuration
        if _infer_board_size(board) != self.board_size:
            return INVALID_MOVE_INDEX

        # --- Placements ---
        if move.type == MoveType.PLACE_RING:
            canon = self._encode_canonical_xy(board, move.to)
            if canon is None:
                return INVALID_MOVE_INDEX
            cx, cy = canon

            # Validate position is within valid hex region (Dec 2025 fix)
            if not self._is_valid_hex_cell(cx, cy):
                return INVALID_MOVE_INDEX

            pos_idx = cy * self.board_size + cx
            count = move.placement_count or 1
            if count < 1 or count > 3:
                return INVALID_MOVE_INDEX

            return pos_idx * 3 + (count - 1)

        # --- Movement / capture / recovery ---
        if move.type in (
            MoveType.MOVE_STACK,
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.RECOVERY_SLIDE,  # RR-CANON-R110–R115: marker slide
        ):
            if move.from_pos is None:
                return INVALID_MOVE_INDEX

            from_canon = self._encode_canonical_xy(board, move.from_pos)
            to_canon = self._encode_canonical_xy(board, move.to)
            if from_canon is None or to_canon is None:
                return INVALID_MOVE_INDEX

            from_cx, from_cy = from_canon
            to_cx, to_cy = to_canon

            # Validate both positions are within valid hex region (Dec 2025 fix)
            # This ensures encoded indices align with V3/V4 model hex_mask
            if not self._is_valid_hex_cell(from_cx, from_cy):
                return INVALID_MOVE_INDEX
            if not self._is_valid_hex_cell(to_cx, to_cy):
                return INVALID_MOVE_INDEX

            dx = to_cx - from_cx
            dy = to_cy - from_cy
            dist = max(abs(dx), abs(dy))
            if dist == 0 or dist > self.max_dist:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist
            dir_y = dy // dist
            try:
                dir_idx = HEX_DIRS.index((dir_x, dir_y))
            except ValueError:
                # Direction not representable in our 6-direction scheme.
                return INVALID_MOVE_INDEX

            from_idx = from_cy * self.board_size + from_cx
            return self.movement_base + from_idx * (NUM_HEX_DIRS * self.max_dist) + dir_idx * self.max_dist + (dist - 1)

        # --- Special ---
        # All "no-op" and choice moves map to the special sentinel index.
        # For training, the value target is derived from game outcome;
        # the policy target for these moves is a single-action "categorical".
        if move.type in (
            MoveType.SKIP_PLACEMENT,
            MoveType.NO_PLACEMENT_ACTION,
            MoveType.NO_MOVEMENT_ACTION,
            MoveType.SKIP_CAPTURE,
            MoveType.NO_LINE_ACTION,
            MoveType.NO_TERRITORY_ACTION,
            MoveType.SKIP_TERRITORY_PROCESSING,
            MoveType.FORCED_ELIMINATION,
            # Additional choice/action moves that need encoding:
            MoveType.ELIMINATE_RINGS_FROM_STACK,
            MoveType.CHOOSE_TERRITORY_OPTION,
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_OPTION,
            MoveType.SKIP_RECOVERY,
        ):
            return self.special_base

        return INVALID_MOVE_INDEX

    def decode_move(
        self,
        index: int,
        game_state: GameState,
    ) -> Move | None:
        """Inverse of encode_move for hex boards.

        Returns a Move instance for valid indices whose endpoints lie on the
        true hex board, or None if the index is out of range or maps off
        the playable hex.
        """
        board = game_state.board

        if board.type not in (BoardType.HEXAGONAL, BoardType.HEX8):
            return None
        if _infer_board_size(board) != self.board_size:
            return None
        if index < 0 or index >= self.policy_size:
            return None

        # --- Placements ---
        if index < self.placement_span:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // self.board_size
            cx = pos_idx % self.board_size

            pos = _from_canonical_xy(board, cx, cy)
            if pos is None or not BoardGeometry.is_within_bounds(pos, board.type, board.size):
                return None

            to_payload: dict[str, int] = {"x": pos.x, "y": pos.y}
            if pos.z is not None:
                to_payload["z"] = pos.z

            move_data = {
                "id": "decoded",
                "type": MoveType.PLACE_RING,
                "player": game_state.current_player,
                "to": to_payload,
                "placementCount": count_idx + 1,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # --- Movement / capture ---
        if index < self.special_base:
            offset = index - self.movement_base

            dist_idx = offset % self.max_dist
            dist = dist_idx + 1
            offset //= self.max_dist

            dir_idx = offset % NUM_HEX_DIRS
            offset //= NUM_HEX_DIRS

            from_idx = offset
            from_cy = from_idx // self.board_size
            from_cx = from_idx % self.board_size

            from_pos = _from_canonical_xy(board, from_cx, from_cy)
            if from_pos is None or not BoardGeometry.is_within_bounds(from_pos, board.type, board.size):
                return None

            dx, dy = HEX_DIRS[dir_idx]
            to_cx = from_cx + dx * dist
            to_cy = from_cy + dy * dist

            to_pos = _from_canonical_xy(board, to_cx, to_cy)
            if to_pos is None or not BoardGeometry.is_within_bounds(to_pos, board.type, board.size):
                return None

            from_payload: dict[str, int] = {"x": from_pos.x, "y": from_pos.y}
            if from_pos.z is not None:
                from_payload["z"] = from_pos.z

            to_payload: dict[str, int] = {"x": to_pos.x, "y": to_pos.y}
            if to_pos.z is not None:
                to_payload["z"] = to_pos.z

            move_data = {
                "id": "decoded",
                "type": MoveType.MOVE_STACK,
                "player": game_state.current_player,
                "from": from_payload,
                "to": to_payload,
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        # --- Special ---
        if index == self.special_base:
            move_data = {
                "id": "decoded",
                "type": MoveType.SKIP_PLACEMENT,
                "player": game_state.current_player,
                "to": {"x": 0, "y": 0},
                "timestamp": datetime.now(),
                "thinkTime": 0,
                "moveNumber": 0,
            }
            return Move(**move_data)

        return None
