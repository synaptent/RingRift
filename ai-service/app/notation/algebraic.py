"""Algebraic notation conversion for RingRift game records.

Implements the notation system defined in docs/GAME_NOTATION_SPEC.md:
- Square board notation: Chess-style a1-h8 (8x8) or a1-s19 (19x19)
- Hexagonal board notation: Cube coordinate x.y format
- Move action codes: P, M, C, CC, L, LR, T, E, etc.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Union

from app.models import BoardType, GameState, Move, MoveType, Position

# ============================================================================
# MoveType <-> Notation Code Mapping (per GAME_NOTATION_SPEC.md Section 6.1)
# ============================================================================

MOVE_TYPE_TO_CODE: dict[MoveType, str] = {
    MoveType.PLACE_RING: "P",
    MoveType.SKIP_PLACEMENT: "SP",
    MoveType.SWAP_SIDES: "SW",
    MoveType.MOVE_STACK: "M",
    MoveType.MOVE_RING: "MR",
    MoveType.BUILD_STACK: "B",
    MoveType.OVERTAKING_CAPTURE: "C",
    MoveType.CONTINUE_CAPTURE_SEGMENT: "CC",
    MoveType.PROCESS_LINE: "L",
    MoveType.CHOOSE_LINE_REWARD: "LR",
    MoveType.CHOOSE_LINE_OPTION: "LR",  # Alias
    MoveType.PROCESS_TERRITORY_REGION: "T",
    MoveType.CHOOSE_TERRITORY_OPTION: "T",  # Alias
    MoveType.ELIMINATE_RINGS_FROM_STACK: "E",
    # Legacy types
    MoveType.LINE_FORMATION: "L",
    MoveType.TERRITORY_CLAIM: "T",
    MoveType.CHAIN_CAPTURE: "C",
    MoveType.FORCED_ELIMINATION: "E",
}

CODE_TO_MOVE_TYPE: dict[str, MoveType] = {
    "P": MoveType.PLACE_RING,
    "SP": MoveType.SKIP_PLACEMENT,
    "SW": MoveType.SWAP_SIDES,
    "M": MoveType.MOVE_STACK,
    "MR": MoveType.MOVE_RING,
    "B": MoveType.BUILD_STACK,
    "C": MoveType.OVERTAKING_CAPTURE,
    "CC": MoveType.CONTINUE_CAPTURE_SEGMENT,
    "EC": MoveType.CONTINUE_CAPTURE_SEGMENT,  # End chain - context determines
    "L": MoveType.PROCESS_LINE,
    "LR": MoveType.CHOOSE_LINE_REWARD,
    "T": MoveType.PROCESS_TERRITORY_REGION,
    "E": MoveType.ELIMINATE_RINGS_FROM_STACK,
}


# ============================================================================
# Position Conversion Functions
# ============================================================================

def position_to_algebraic(pos: Position, board_type: BoardType) -> str:
    """Convert Position to algebraic notation.

    Square boards use chess-style notation (a1-h8 or a1-s19).
    Hexagonal boards use cube coordinate notation (x.y).

    Args:
        pos: Position object with x, y (and optionally z for hex)
        board_type: The board type for notation context

    Returns:
        Algebraic notation string
    """
    if board_type == BoardType.HEXAGONAL:
        return _position_to_algebraic_hex(pos)
    else:
        return _position_to_algebraic_square(pos)


def _position_to_algebraic_square(pos: Position) -> str:
    """Convert square board position to algebraic notation.

    Example: Position(x=3, y=4) -> "d5"
    """
    col = chr(ord('a') + pos.x)
    row = str(pos.y + 1)
    return f"{col}{row}"


def _position_to_algebraic_hex(pos: Position) -> str:
    """Convert hexagonal board position to algebraic notation.

    Uses cube coordinate x.y format (z is derived from constraint x+y+z=0).
    Example: Position(x=3, y=-2, z=-1) -> "3.-2"
    """
    return f"{pos.x}.{pos.y}"


def algebraic_to_position(notation: str, board_type: BoardType) -> Position:
    """Parse algebraic notation to Position.

    Args:
        notation: Algebraic notation string
        board_type: The board type for parsing context

    Returns:
        Position object

    Raises:
        ValueError: If notation format is invalid
    """
    if board_type == BoardType.HEXAGONAL:
        return _algebraic_to_position_hex(notation)
    else:
        return _algebraic_to_position_square(notation)


def _algebraic_to_position_square(notation: str) -> Position:
    """Parse square board algebraic notation.

    Example: "d5" -> Position(x=3, y=4)
    """
    if not notation or len(notation) < 2:
        raise ValueError(f"Invalid square notation: {notation}")

    col_char = notation[0].lower()
    if not 'a' <= col_char <= 's':
        raise ValueError(f"Invalid column in notation: {notation}")

    try:
        row = int(notation[1:])
    except ValueError:
        raise ValueError(f"Invalid row in notation: {notation}")

    x = ord(col_char) - ord('a')
    y = row - 1

    return Position(x=x, y=y)


def _algebraic_to_position_hex(notation: str) -> Position:
    """Parse hexagonal board algebraic notation.

    Example: "3.-2" -> Position(x=3, y=-2, z=-1)
    """
    if '.' not in notation:
        raise ValueError(f"Invalid hex notation (missing '.'): {notation}")

    parts = notation.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid hex notation format: {notation}")

    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid hex coordinate values: {notation}")

    z = -x - y  # Cube coordinate constraint
    return Position(x=x, y=y, z=z)


# ============================================================================
# Move Notation Conversion
# ============================================================================

def move_to_algebraic(move: Move, board_type: BoardType) -> str:
    """Convert Move to algebraic notation string.

    Args:
        move: Move object to convert
        board_type: Board type for position notation

    Returns:
        Algebraic notation string like "P d4" or "M d4-e5 xc4"
    """
    code = MOVE_TYPE_TO_CODE.get(move.type, "?")
    parts = [code]

    # Handle different move types
    if move.type in (MoveType.PLACE_RING,):
        # Placement: P <to>
        parts.append(position_to_algebraic(move.to, board_type))

    elif move.type in (MoveType.SKIP_PLACEMENT, MoveType.SWAP_SIDES):
        # No position needed
        pass

    elif move.type in (MoveType.MOVE_STACK, MoveType.MOVE_RING, MoveType.BUILD_STACK):
        # Movement: M <from>-<to>
        if move.from_pos:
            from_str = position_to_algebraic(move.from_pos, board_type)
            to_str = position_to_algebraic(move.to, board_type)
            parts.append(f"{from_str}-{to_str}")
        else:
            parts.append(position_to_algebraic(move.to, board_type))

        # Add marker annotation if present
        if move.marker_left:
            marker_str = position_to_algebraic(move.marker_left, board_type)
            parts.append(f"@{marker_str}")

    elif move.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT,
                       MoveType.CHAIN_CAPTURE):
        # Capture: C <from>-<to> x<captured>
        if move.from_pos:
            from_str = position_to_algebraic(move.from_pos, board_type)
            to_str = position_to_algebraic(move.to, board_type)
            parts.append(f"{from_str}-{to_str}")
        else:
            parts.append(position_to_algebraic(move.to, board_type))

        if move.capture_target:
            cap_str = position_to_algebraic(move.capture_target, board_type)
            parts.append(f"x{cap_str}")

    elif move.type in (MoveType.PROCESS_LINE, MoveType.LINE_FORMATION):
        # Line: L <index> (index derived from context, use 1 as default)
        parts.append("1")

    elif move.type in (MoveType.CHOOSE_LINE_REWARD, MoveType.CHOOSE_LINE_OPTION):
        # Line reward: LR <option>
        parts.append("1")  # Default option

    elif move.type in (MoveType.PROCESS_TERRITORY_REGION, MoveType.TERRITORY_CLAIM,
                       MoveType.CHOOSE_TERRITORY_OPTION):
        # Territory: T <index>
        parts.append("1")

    elif move.type in (MoveType.ELIMINATE_RINGS_FROM_STACK, MoveType.FORCED_ELIMINATION):
        # Elimination: E <position>
        parts.append(position_to_algebraic(move.to, board_type))

    return " ".join(parts)


def algebraic_to_move(
    notation: str,
    state: GameState,
    player: int,
    move_number: int,
) -> Move:
    """Parse algebraic notation to Move in context of game state.

    Args:
        notation: Algebraic notation string
        state: Current game state for context
        player: Player number making the move
        move_number: Move number in game sequence

    Returns:
        Move object

    Raises:
        ValueError: If notation cannot be parsed
    """
    parts = notation.strip().split()
    if not parts:
        raise ValueError("Empty notation string")

    code = parts[0]
    move_type = CODE_TO_MOVE_TYPE.get(code)
    if move_type is None:
        raise ValueError(f"Unknown move code: {code}")

    board_type = state.board_type
    from_pos: Position | None = None
    to_pos: Position | None = None
    capture_target: Position | None = None
    marker_left: Position | None = None

    # Parse remaining parts based on move type
    for _i, part in enumerate(parts[1:], 1):
        if part.startswith('x'):
            # Capture target
            capture_target = algebraic_to_position(part[1:], board_type)
        elif part.startswith('@'):
            # Marker left
            marker_left = algebraic_to_position(part[1:], board_type)
        elif '-' in part:
            # From-to notation
            from_str, to_str = part.split('-', 1)
            from_pos = algebraic_to_position(from_str, board_type)
            to_pos = algebraic_to_position(to_str, board_type)
        elif part[0].isalpha() or (part[0] == '-' and '.' in part) or part[0].isdigit():
            # Single position (placement or hex)
            if to_pos is None:
                to_pos = algebraic_to_position(part, board_type)

    # Default position for moves that don't have spatial component
    if to_pos is None:
        to_pos = Position(x=0, y=0)

    from uuid import uuid4

    return Move(
        id=str(uuid4()),
        type=move_type,
        player=player,
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        marker_left=marker_left,
        timestamp=datetime.now(),
        think_time=0,
        move_number=move_number,
    )


# ============================================================================
# Game Record Functions
# ============================================================================

def moves_to_notation_list(
    moves: list[Move],
    board_type: BoardType,
) -> list[str]:
    """Convert a list of moves to algebraic notation strings.

    Args:
        moves: List of Move objects
        board_type: Board type for position notation

    Returns:
        List of algebraic notation strings
    """
    return [move_to_algebraic(m, board_type) for m in moves]


def game_to_pgn(
    moves: list[Move],
    metadata: dict[str, Union[str, int, None]],
    board_type: BoardType,
) -> str:
    """Generate PGN-style game record.

    Args:
        moves: List of moves in game order
        metadata: Game metadata dict with keys like:
            - game_id, board, date, player1, player2, result,
            - termination, rng_seed, total_moves
        board_type: Board type for notation

    Returns:
        PGN-formatted game record string
    """
    lines = []

    # Header section
    lines.append('[Game "RingRift"]')

    board_name = {
        BoardType.SQUARE8: "square8",
        BoardType.SQUARE19: "square19",
        BoardType.HEXAGONAL: "hexagonal",
    }.get(board_type, "unknown")
    lines.append(f'[Board "{board_name}"]')

    if "date" in metadata:
        lines.append(f'[Date "{metadata["date"]}"]')
    else:
        lines.append(f'[Date "{datetime.now().strftime("%Y-%m-%d")}"]')

    if "player1" in metadata:
        lines.append(f'[Player1 "{metadata["player1"]}"]')
    if "player2" in metadata:
        lines.append(f'[Player2 "{metadata["player2"]}"]')

    # Result: "1-0", "0-1", or "1/2-1/2"
    if metadata.get("winner"):
        winner = metadata["winner"]
        if winner == 1:
            result = "1-0"
        elif winner == 2:
            result = "0-1"
        else:
            result = "1/2-1/2"
    else:
        result = "*"  # Incomplete/unknown
    lines.append(f'[Result "{result}"]')

    if "termination" in metadata:
        lines.append(f'[Termination "{metadata["termination"]}"]')

    if "rng_seed" in metadata:
        lines.append(f'[RNGSeed "{metadata["rng_seed"]}"]')

    lines.append(f'[TotalMoves "{len(moves)}"]')

    # Empty line before move list
    lines.append('')

    # Move list in turn-based notation
    # Group moves by turn number (2 moves per turn line for 2-player)
    notation_list = moves_to_notation_list(moves, board_type)

    turn_num = 1
    i = 0
    while i < len(notation_list):
        # Get P1's move
        p1_move = notation_list[i] if i < len(notation_list) else ""
        i += 1

        # Get P2's move
        p2_move = notation_list[i] if i < len(notation_list) else ""
        i += 1

        if p2_move:
            lines.append(f"{turn_num}. {p1_move}      {p2_move}")
        else:
            lines.append(f"{turn_num}. {p1_move}")

        turn_num += 1

    # Final result indicator
    lines.append(f"\n{result}")

    return '\n'.join(lines)


def parse_pgn(pgn_text: str) -> tuple[dict[str, str], list[str]]:
    """Parse a PGN-format game record.

    Args:
        pgn_text: PGN formatted string

    Returns:
        Tuple of (metadata dict, list of notation strings)
    """
    metadata: dict[str, str] = {}
    move_notations: list[str] = []

    lines = pgn_text.strip().split('\n')
    in_moves = False

    for line in lines:
        line = line.strip()
        if not line:
            in_moves = True
            continue

        if not in_moves:
            # Parse header tags
            match = re.match(r'\[(\w+)\s+"(.*)"\]', line)
            if match:
                key = match.group(1).lower()
                value = match.group(2)
                metadata[key] = value
        else:
            # Parse move list
            if line in ('1-0', '0-1', '1/2-1/2', '*'):
                continue

            # Remove turn numbers
            line = re.sub(r'\d+\.', '', line)

            # Split into individual moves
            parts = line.split()
            for part in parts:
                part = part.strip()
                if part and not part.startswith('{') and not part.endswith('}'):
                    move_notations.append(part)

    return metadata, move_notations
