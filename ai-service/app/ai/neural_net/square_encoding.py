"""Square board action encoding/decoding for neural network policy heads.

This module provides action encoders for square boards (8x8 and 19x19) with
a class-based interface consistent with ActionEncoderHex.

Extracted from _neural_net_legacy.py as part of encoding consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.models import BoardState, BoardType, GameState, Move, MoveType, Position
from app.rules.legacy.move_type_aliases import convert_legacy_move_type

from .constants import (
    INVALID_MOVE_INDEX,
    MAX_DIST_SQUARE8,
    MAX_DIST_SQUARE19,
    NUM_LINE_DIRS,
    NUM_SQUARE_DIRS,
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    SQUARE8_FORCED_ELIMINATION_IDX,
    SQUARE8_LINE_CHOICE_BASE,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_NO_LINE_ACTION_IDX,
    SQUARE8_NO_MOVEMENT_ACTION_IDX,
    SQUARE8_NO_PLACEMENT_ACTION_IDX,
    SQUARE8_NO_TERRITORY_ACTION_IDX,
    SQUARE8_PLACEMENT_SPAN,
    SQUARE8_SKIP_CAPTURE_IDX,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SKIP_RECOVERY_IDX,
    SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    SQUARE8_TERRITORY_CHOICE_BASE,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE19_FORCED_ELIMINATION_IDX,
    SQUARE19_LINE_CHOICE_BASE,
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_MOVEMENT_BASE,
    SQUARE19_NO_LINE_ACTION_IDX,
    SQUARE19_NO_MOVEMENT_ACTION_IDX,
    SQUARE19_NO_PLACEMENT_ACTION_IDX,
    SQUARE19_NO_TERRITORY_ACTION_IDX,
    SQUARE19_SKIP_CAPTURE_IDX,
    SQUARE19_SKIP_PLACEMENT_IDX,
    SQUARE19_SKIP_RECOVERY_IDX,
    SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE19_SWAP_SIDES_IDX,
    SQUARE19_TERRITORY_CHOICE_BASE,
    SQUARE19_TERRITORY_CLAIM_BASE,
    TERRITORY_MAX_PLAYERS,
    TERRITORY_SIZE_BUCKETS,
)

# Direction vectors for 8-way movement on square boards
SQUARE_DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


@dataclass
class DecodedPolicyIndex:
    """Decoded policy index components for transformation.

    This structure contains enough information to apply symmetry transformations
    (rotations, reflections) and re-encode the policy index.
    """

    action_type: str  # "placement", "movement", "process_line", etc.
    board_size: int  # 8 for square8, 19 for square19
    x: int = 0  # Primary position x
    y: int = 0  # Primary position y
    count_idx: int = 0  # For placement: (placement_count - 1)
    dir_idx: int = 0  # For movement: direction index 0-7
    dist: int = 0  # For movement: distance 1..MAX_DIST
    option: int = 0  # For line/territory choice
    size_bucket: int = 0  # For territory choice
    player_idx: int = 0  # For territory choice
    is_special: bool = False  # Special actions don't transform


def _line_anchor_position(move: Move) -> Position | None:
    """Extract anchor position for line-processing moves."""
    if move.to is not None:
        return move.to
    if move.formed_lines:
        try:
            line = move.formed_lines[0]
            if hasattr(line, "positions") and line.positions:
                return line.positions[0]
        except Exception:
            return None
    return None


class ActionEncoderSquare:
    """Base action encoder for square boards.

    Provides a unified interface consistent with ActionEncoderHex for encoding
    and decoding moves on square boards.
    """

    def __init__(
        self,
        board_size: int,
        policy_size: int,
        max_dist: int,
        placement_span: int,
        movement_base: int,
        line_form_base: int,
        territory_claim_base: int,
        line_choice_base: int,
        territory_choice_base: int,
        special_indices: dict[str, int],
    ) -> None:
        self.board_size = board_size
        self.policy_size = policy_size
        self.max_dist = max_dist
        self.placement_span = placement_span
        self.movement_base = movement_base
        self.line_form_base = line_form_base
        self.territory_claim_base = territory_claim_base
        self.line_choice_base = line_choice_base
        self.territory_choice_base = territory_choice_base
        self.special_indices = special_indices

    def encode_move(self, move: Move, board: BoardState) -> int:
        """Encode a square board move into a policy index.

        Args:
            move: The move to encode
            board: Board context for coordinate validation

        Returns:
            Policy index in [0, policy_size), or INVALID_MOVE_INDEX
        """
        N = self.board_size
        raw_move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
        # Legacy aliases are normalized to canonical move types for encoding.
        move_type = convert_legacy_move_type(raw_move_type, warn=False)

        # Placement
        if move_type == "place_ring":
            cx, cy = move.to.x, move.to.y
            if not (0 <= cx < N and 0 <= cy < N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * N + cx
            count_idx = (move.placement_count or 1) - 1
            return pos_idx * 3 + count_idx

        # Movement / capture
        if move_type in [
            "move_stack",
            "overtaking_capture",
            "continue_capture_segment",
            "recovery_slide",
        ]:
            if not move.from_pos:
                return INVALID_MOVE_INDEX
            cfx, cfy = move.from_pos.x, move.from_pos.y
            ctx, cty = move.to.x, move.to.y

            if not (0 <= cfx < N and 0 <= cfy < N and 0 <= ctx < N and 0 <= cty < N):
                return INVALID_MOVE_INDEX

            from_idx = cfy * N + cfx
            dx, dy = ctx - cfx, cty - cfy
            dist = max(abs(dx), abs(dy))
            if dist == 0 or dist > self.max_dist:
                return INVALID_MOVE_INDEX

            dir_x = dx // dist if dist > 0 else 0
            dir_y = dy // dist if dist > 0 else 0

            try:
                dir_idx = SQUARE_DIRS.index((dir_x, dir_y))
            except ValueError:
                return INVALID_MOVE_INDEX

            return (
                self.movement_base
                + from_idx * (NUM_SQUARE_DIRS * self.max_dist)
                + dir_idx * self.max_dist
                + (dist - 1)
            )

        # Line formation
        if move_type == "process_line":
            line_pos = _line_anchor_position(move)
            if line_pos is None:
                return INVALID_MOVE_INDEX
            cx, cy = line_pos.x, line_pos.y
            if not (0 <= cx < N and 0 <= cy < N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * N + cx
            return self.line_form_base + pos_idx * NUM_LINE_DIRS

        # Territory claim
        if move_type == "eliminate_rings_from_stack":
            if move.to is None:
                return INVALID_MOVE_INDEX
            cx, cy = move.to.x, move.to.y
            if not (0 <= cx < N and 0 <= cy < N):
                return INVALID_MOVE_INDEX
            pos_idx = cy * N + cx
            return self.territory_claim_base + pos_idx

        # Special actions
        if move_type in self.special_indices:
            return self.special_indices[move_type]

        # Line choice
        if move_type == "choose_line_option":
            option = (move.placement_count or 1) - 1
            option = max(0, min(3, option))
            return self.line_choice_base + option

        # Territory choice
        if move_type == "choose_territory_option":
            canonical_pos = move.to
            region_size = 1
            controlling_player = move.player

            if move.disconnected_regions:
                regions = list(move.disconnected_regions)
                if regions:
                    region = regions[0]
                    if hasattr(region, "spaces") and region.spaces:
                        spaces = list(region.spaces)
                        region_size = len(spaces)
                        canonical_pos = min(spaces, key=lambda p: (p.y, p.x))
                    if hasattr(region, "controlling_player"):
                        controlling_player = region.controlling_player

            if canonical_pos is None:
                return INVALID_MOVE_INDEX
            cx, cy = canonical_pos.x, canonical_pos.y
            if not (0 <= cx < N and 0 <= cy < N):
                return INVALID_MOVE_INDEX

            pos_idx = cy * N + cx
            size_bucket = min(region_size - 1, TERRITORY_SIZE_BUCKETS - 1)
            player_idx = (controlling_player - 1) % TERRITORY_MAX_PLAYERS

            return (
                self.territory_choice_base
                + pos_idx * (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
                + size_bucket * TERRITORY_MAX_PLAYERS
                + player_idx
            )

        return INVALID_MOVE_INDEX

    def decode_move(self, index: int, game_state: GameState) -> Move | None:
        """Decode a policy index back to a Move.

        Args:
            index: Policy index to decode
            game_state: Game context for move construction

        Returns:
            Move instance or None if invalid
        """
        if index < 0 or index >= self.policy_size:
            return None

        N = self.board_size
        board = game_state.board

        # Placement
        if index < self.placement_span:
            count_idx = index % 3
            pos_idx = index // 3
            cy = pos_idx // N
            cx = pos_idx % N

            if not (0 <= cx < N and 0 <= cy < N):
                return None

            return Move(
                id="decoded",
                type=MoveType.PLACE_RING,
                player=game_state.current_player,
                to=Position(x=cx, y=cy),
                placementCount=count_idx + 1,
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=0,
            )

        # Movement
        movement_end = self.movement_base + N * N * NUM_SQUARE_DIRS * self.max_dist
        if self.movement_base <= index < movement_end:
            offset = index - self.movement_base

            dist_idx = offset % self.max_dist
            dist = dist_idx + 1
            offset //= self.max_dist

            dir_idx = offset % NUM_SQUARE_DIRS
            offset //= NUM_SQUARE_DIRS

            from_idx = offset
            from_cy = from_idx // N
            from_cx = from_idx % N

            if not (0 <= from_cx < N and 0 <= from_cy < N):
                return None

            dir_x, dir_y = SQUARE_DIRS[dir_idx]
            to_cx = from_cx + dir_x * dist
            to_cy = from_cy + dir_y * dist

            if not (0 <= to_cx < N and 0 <= to_cy < N):
                return None

            return Move(
                id="decoded",
                type=MoveType.MOVE_STACK,
                player=game_state.current_player,
                to=Position(x=to_cx, y=to_cy),
                **{"from": Position(x=from_cx, y=from_cy)},
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=0,
            )

        # Special actions - check if this is a special index
        for move_type, special_idx in self.special_indices.items():
            if index == special_idx:
                return Move(
                    id="decoded",
                    type=MoveType(move_type),
                    player=game_state.current_player,
                    to=Position(x=0, y=0),
                    timestamp=datetime.now(),
                    thinkTime=0,
                    moveNumber=0,
                )

        return None

    def decode_to_components(self, policy_idx: int) -> DecodedPolicyIndex | None:
        """Decode a policy index to components for transformation.

        This is used for data augmentation where we need to transform
        policy indices under rotations and reflections.

        Args:
            policy_idx: Policy index to decode

        Returns:
            DecodedPolicyIndex with components, or None if invalid
        """
        if policy_idx < 0 or policy_idx >= self.policy_size:
            return None

        N = self.board_size

        # Check special actions first (they don't transform)
        for action, idx in self.special_indices.items():
            if policy_idx == idx:
                return DecodedPolicyIndex(
                    action_type=action,
                    board_size=N,
                    is_special=True,
                )

        # Placement
        if policy_idx < self.placement_span:
            count_idx = policy_idx % 3
            pos_idx = policy_idx // 3
            return DecodedPolicyIndex(
                action_type="placement",
                board_size=N,
                y=pos_idx // N,
                x=pos_idx % N,
                count_idx=count_idx,
            )

        # Movement
        movement_end = self.movement_base + N * N * NUM_SQUARE_DIRS * self.max_dist
        if self.movement_base <= policy_idx < movement_end:
            offset = policy_idx - self.movement_base
            dist = (offset % self.max_dist) + 1
            offset //= self.max_dist
            dir_idx = offset % NUM_SQUARE_DIRS
            offset //= NUM_SQUARE_DIRS
            from_idx = offset

            return DecodedPolicyIndex(
                action_type="movement",
                board_size=N,
                y=from_idx // N,
                x=from_idx % N,
                dir_idx=dir_idx,
                dist=dist,
            )

        # Line processing
        line_form_end = self.line_form_base + N * N * NUM_LINE_DIRS
        if self.line_form_base <= policy_idx < line_form_end:
            offset = policy_idx - self.line_form_base
            pos_idx = offset // NUM_LINE_DIRS
            return DecodedPolicyIndex(
                action_type="process_line",
                board_size=N,
                y=pos_idx // N,
                x=pos_idx % N,
                dir_idx=offset % NUM_LINE_DIRS,
            )

        # Territory elimination
        territory_claim_end = self.territory_claim_base + N * N
        if self.territory_claim_base <= policy_idx < territory_claim_end:
            pos_idx = policy_idx - self.territory_claim_base
            return DecodedPolicyIndex(
                action_type="eliminate_rings_from_stack",
                board_size=N,
                y=pos_idx // N,
                x=pos_idx % N,
            )

        # Line choice
        if self.line_choice_base <= policy_idx < self.line_choice_base + 4:
            return DecodedPolicyIndex(
                action_type="line_choice",
                board_size=N,
                option=policy_idx - self.line_choice_base,
                is_special=True,  # Doesn't depend on position
            )

        # Territory choice
        territory_choice_span = N * N * TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS
        if (
            self.territory_choice_base
            <= policy_idx
            < self.territory_choice_base + territory_choice_span
        ):
            offset = policy_idx - self.territory_choice_base
            player_idx = offset % TERRITORY_MAX_PLAYERS
            offset //= TERRITORY_MAX_PLAYERS
            size_bucket = offset % TERRITORY_SIZE_BUCKETS
            offset //= TERRITORY_SIZE_BUCKETS
            pos_idx = offset

            return DecodedPolicyIndex(
                action_type="territory_choice",
                board_size=N,
                y=pos_idx // N,
                x=pos_idx % N,
                size_bucket=size_bucket,
                player_idx=player_idx,
            )

        return None

    def encode_from_decoded(self, decoded: DecodedPolicyIndex) -> int:
        """Re-encode a DecodedPolicyIndex back to a policy index.

        This is the inverse of decode_to_components(), used for data augmentation
        where we transform coordinates and need to re-encode.

        Args:
            decoded: Decoded policy components (potentially with transformed coords)

        Returns:
            Policy index, or INVALID_MOVE_INDEX if encoding fails
        """
        N = self.board_size

        # Validate coordinates
        if not (0 <= decoded.x < N and 0 <= decoded.y < N):
            return INVALID_MOVE_INDEX

        pos_idx = decoded.y * N + decoded.x

        if decoded.action_type == "placement":
            return pos_idx * 3 + decoded.count_idx

        elif decoded.action_type == "movement":
            # For movement, we also need to transform the direction
            # The direction is stored as dir_idx (0-7), which maps to SQUARE_DIRS
            return (
                self.movement_base
                + pos_idx * (NUM_SQUARE_DIRS * self.max_dist)
                + decoded.dir_idx * self.max_dist
                + (decoded.dist - 1)
            )

        elif decoded.action_type == "process_line":
            return self.line_form_base + pos_idx * NUM_LINE_DIRS + decoded.dir_idx

        elif decoded.action_type == "eliminate_rings_from_stack":
            return self.territory_claim_base + pos_idx

        elif decoded.action_type == "territory_choice":
            return (
                self.territory_choice_base
                + pos_idx * (TERRITORY_SIZE_BUCKETS * TERRITORY_MAX_PLAYERS)
                + decoded.size_bucket * TERRITORY_MAX_PLAYERS
                + decoded.player_idx
            )

        elif decoded.action_type == "line_choice":
            return self.line_choice_base + decoded.option

        # Special actions - look up in special_indices
        elif decoded.is_special and decoded.action_type in self.special_indices:
            return self.special_indices[decoded.action_type]

        return INVALID_MOVE_INDEX


class ActionEncoderSquare8(ActionEncoderSquare):
    """Action encoder for 8x8 square boards.

    Policy layout (max ~7000 indices):
      - Placement: 0..191 (8×8×3)
      - Movement: 192..3775 (8×8×8×7)
      - Line processing (process_line): 3776..4031 (8×8×4)
      - Territory elimination (eliminate_rings_from_stack): 4032..4095 (8×8)
      - Special actions: 4096..4102
      - Line choice: 4099..4102
      - Territory choice: 4103..6150
    """

    def __init__(self) -> None:
        special_indices = {
            "skip_placement": SQUARE8_SKIP_PLACEMENT_IDX,
            "swap_sides": SQUARE8_SWAP_SIDES_IDX,
            "skip_recovery": SQUARE8_SKIP_RECOVERY_IDX,
            "no_placement_action": SQUARE8_NO_PLACEMENT_ACTION_IDX,
            "no_movement_action": SQUARE8_NO_MOVEMENT_ACTION_IDX,
            "skip_capture": SQUARE8_SKIP_CAPTURE_IDX,
            "no_line_action": SQUARE8_NO_LINE_ACTION_IDX,
            "no_territory_action": SQUARE8_NO_TERRITORY_ACTION_IDX,
            "skip_territory_processing": SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
            "forced_elimination": SQUARE8_FORCED_ELIMINATION_IDX,
        }

        super().__init__(
            board_size=8,
            policy_size=POLICY_SIZE_8x8,
            max_dist=MAX_DIST_SQUARE8,
            placement_span=SQUARE8_PLACEMENT_SPAN,
            movement_base=SQUARE8_MOVEMENT_BASE,
            line_form_base=SQUARE8_LINE_FORM_BASE,
            territory_claim_base=SQUARE8_TERRITORY_CLAIM_BASE,
            line_choice_base=SQUARE8_LINE_CHOICE_BASE,
            territory_choice_base=SQUARE8_TERRITORY_CHOICE_BASE,
            special_indices=special_indices,
        )


class ActionEncoderSquare19(ActionEncoderSquare):
    """Action encoder for 19x19 square boards.

    Policy layout (max ~67000 indices):
      - Placement: 0..1082 (19×19×3)
      - Movement: 1083..53066 (19×19×8×18)
      - Line processing (process_line): 53067..54510 (19×19×4)
      - Territory elimination (eliminate_rings_from_stack): 54511..54871 (19×19)
      - Special actions: 54872..54878
      - Line choice: 54875..54878
      - Territory choice: 54879..66430
    """

    def __init__(self) -> None:
        special_indices = {
            "skip_placement": SQUARE19_SKIP_PLACEMENT_IDX,
            "swap_sides": SQUARE19_SWAP_SIDES_IDX,
            "skip_recovery": SQUARE19_SKIP_RECOVERY_IDX,
            "no_placement_action": SQUARE19_NO_PLACEMENT_ACTION_IDX,
            "no_movement_action": SQUARE19_NO_MOVEMENT_ACTION_IDX,
            "skip_capture": SQUARE19_SKIP_CAPTURE_IDX,
            "no_line_action": SQUARE19_NO_LINE_ACTION_IDX,
            "no_territory_action": SQUARE19_NO_TERRITORY_ACTION_IDX,
            "skip_territory_processing": SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
            "forced_elimination": SQUARE19_FORCED_ELIMINATION_IDX,
        }

        super().__init__(
            board_size=19,
            policy_size=POLICY_SIZE_19x19,
            max_dist=MAX_DIST_SQUARE19,
            placement_span=3 * 19 * 19,  # 1083
            movement_base=3 * 19 * 19,  # Movement starts after placement
            line_form_base=SQUARE19_LINE_FORM_BASE,
            territory_claim_base=SQUARE19_TERRITORY_CLAIM_BASE,
            line_choice_base=SQUARE19_LINE_CHOICE_BASE,
            territory_choice_base=SQUARE19_TERRITORY_CHOICE_BASE,
            special_indices=special_indices,
        )


def get_action_encoder(board_type: BoardType) -> ActionEncoderSquare | None:
    """Factory function to get the appropriate action encoder for a board type.

    Args:
        board_type: The board type to get an encoder for

    Returns:
        ActionEncoderSquare instance for square boards, None for hex boards
        (use ActionEncoderHex from hex_encoding.py for hex boards)
    """
    if board_type == BoardType.SQUARE8:
        return ActionEncoderSquare8()
    elif board_type == BoardType.SQUARE19:
        return ActionEncoderSquare19()
    return None


# ============================================================================
# Backwards-compatible wrapper functions (migrated from _neural_net_legacy.py)
# ============================================================================

# Singleton encoder instances for backwards compatibility
_encoder_8 = ActionEncoderSquare8()
_encoder_19 = ActionEncoderSquare19()


def _encode_move_square8(move: Move, board: BoardState) -> int:
    """Encode a move for square8 board (legacy wrapper).

    This function is provided for backwards compatibility with code that
    imports `_encode_move_square8` from the legacy module.

    For new code, use ActionEncoderSquare8().encode_move(move, board) instead.
    """
    return _encoder_8.encode_move(move, board)


def _encode_move_square19(move: Move, board: BoardState) -> int:
    """Encode a move for square19 board (legacy wrapper).

    This function is provided for backwards compatibility with code that
    imports `_encode_move_square19` from the legacy module.

    For new code, use ActionEncoderSquare19().encode_move(move, board) instead.
    """
    return _encoder_19.encode_move(move, board)


def _decode_move_square8(idx: int) -> DecodedPolicyIndex | None:
    """Decode a policy index for square8 board (legacy wrapper).

    This function is provided for backwards compatibility.

    For new code, use ActionEncoderSquare8().decode_to_components(idx) instead.
    """
    return _encoder_8.decode_to_components(idx)


def _decode_move_square19(idx: int) -> DecodedPolicyIndex | None:
    """Decode a policy index for square19 board (legacy wrapper).

    This function is provided for backwards compatibility.

    For new code, use ActionEncoderSquare19().decode_to_components(idx) instead.
    """
    return _encoder_19.decode_to_components(idx)


def transform_policy_index_square(
    idx: int,
    board_size: int,
    transform_fn,
) -> int:
    """Transform a policy index under a coordinate transformation.

    Decodes the policy index, applies the transformation to any spatial
    coordinates, and re-encodes. Used for data augmentation (rotations,
    reflections).

    Args:
        idx: Original policy index
        board_size: Board size (8 or 19)
        transform_fn: Function (x, y) -> (x', y') for coordinate transformation

    Returns:
        Transformed policy index, or original idx if transformation not applicable
    """
    encoder = _encoder_8 if board_size == 8 else _encoder_19

    decoded = encoder.decode_to_components(idx)
    if decoded is None or decoded.is_special:
        return idx  # Special actions don't transform

    # Apply transformation to coordinates
    new_x, new_y = transform_fn(decoded.x, decoded.y)
    decoded.x = new_x
    decoded.y = new_y

    # Re-encode with transformed coordinates
    return encoder.encode_from_decoded(decoded)
