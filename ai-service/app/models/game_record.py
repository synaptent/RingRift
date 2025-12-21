"""
Pydantic Models for RingRift Game Records

Game records are the canonical format for storing completed games, supporting:
- Training data pipelines (JSONL export)
- Replay systems
- Analysis tooling
- Historical game storage

Mirrors TypeScript types from src/shared/types/gameRecord.ts
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .core import (
    BoardType,
    LineInfo,
    Move,
    MoveType,
    Position,
    ProgressSnapshot,
    Territory,
)


class GameOutcome(str, Enum):
    """How the game ended."""
    RING_ELIMINATION = "ring_elimination"
    TERRITORY_CONTROL = "territory_control"
    LAST_PLAYER_STANDING = "last_player_standing"
    TIMEOUT = "timeout"
    RESIGNATION = "resignation"
    DRAW = "draw"
    ABANDONMENT = "abandonment"


class RecordSource(str, Enum):
    """Where the game record originated."""
    ONLINE_GAME = "online_game"
    SELF_PLAY = "self_play"
    CMAES_OPTIMIZATION = "cmaes_optimization"
    TOURNAMENT = "tournament"
    SOAK_TEST = "soak_test"
    MANUAL_IMPORT = "manual_import"


class PlayerRecordInfo(BaseModel):
    """Player information as recorded in a game record."""
    player_number: int = Field(alias="playerNumber")
    username: str
    player_type: Literal["human", "ai"] = Field(alias="playerType")
    rating_before: int | None = Field(None, alias="ratingBefore")
    rating_after: int | None = Field(None, alias="ratingAfter")
    ai_difficulty: int | None = Field(None, alias="aiDifficulty")
    ai_type: str | None = Field(None, alias="aiType")

    model_config = ConfigDict(populate_by_name=True)


class MoveRecord(BaseModel):
    """
    Lightweight move record for storage and training data.

    This is a simplified version of Move that retains only the fields
    needed for replay, training, and analysis. Full diagnostic fields
    (stackMoved, capturedStacks, etc.) are omitted for space efficiency.
    """
    move_number: int = Field(alias="moveNumber")
    player: int
    type: MoveType
    from_pos: Position | None = Field(None, alias="from")
    to: Position | None = None

    # Capture metadata (when applicable)
    capture_target: Position | None = Field(None, alias="captureTarget")

    # Placement metadata
    placement_count: int | None = Field(None, alias="placementCount")
    placed_on_stack: bool | None = Field(None, alias="placedOnStack")

    # Line/territory processing metadata
    formed_lines: tuple[LineInfo, ...] | None = Field(None, alias="formedLines")
    collapsed_markers: tuple[Position, ...] | None = Field(None, alias="collapsedMarkers")
    disconnected_regions: tuple[Territory, ...] | None = Field(None, alias="disconnectedRegions")
    eliminated_rings: tuple[dict[str, int], ...] | None = Field(None, alias="eliminatedRings")

    # Timing
    think_time_ms: int = Field(alias="thinkTimeMs")

    # Optional RingRift Notation representation
    rrn: str | None = None

    # MCTS visit distribution for KL-divergence training
    # Maps move indices to visit probabilities (normalized visit counts)
    # Only populated when MCTS is used during self-play
    mcts_policy: dict[int, float] | None = Field(None, alias="mctsPolicy")

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    @classmethod
    def from_move(cls, move: Move) -> "MoveRecord":
        """Create a MoveRecord from a full Move object."""
        return cls.model_validate({
            "moveNumber": move.move_number,
            "player": move.player,
            "type": move.type,
            "from": move.from_pos,
            "to": move.to,
            "captureTarget": move.capture_target,
            "placementCount": move.placement_count,
            "placedOnStack": move.placed_on_stack,
            "formedLines": move.formed_lines,
            "collapsedMarkers": move.collapsed_markers,
            "disconnectedRegions": move.disconnected_regions,
            "eliminatedRings": move.eliminated_rings,
            "thinkTimeMs": move.think_time,
        })


class GameRecordMetadata(BaseModel):
    """Metadata about the game record itself (not the game)."""
    record_version: str = Field("1.0", alias="recordVersion")
    created_at: datetime = Field(alias="createdAt")
    source: RecordSource
    source_id: str | None = Field(None, alias="sourceId")  # e.g., CMA-ES run ID
    generation: int | None = None  # For evolutionary algorithms
    candidate_id: int | None = Field(None, alias="candidateId")
    tags: list[str] = Field(default_factory=list)
    # FSM validation status (Phase 7: Data Pipeline)
    # - True: All moves validated against FSM orchestrator
    # - False: One or more moves failed FSM validation
    # - None: FSM validation not performed (legacy data)
    fsm_validated: bool | None = Field(None, alias="fsmValidated")

    model_config = ConfigDict(populate_by_name=True)


class FinalScore(BaseModel):
    """Final score breakdown at game end."""
    rings_eliminated: dict[int, int] = Field(alias="ringsEliminated")
    territory_spaces: dict[int, int] = Field(alias="territorySpaces")
    rings_remaining: dict[int, int] = Field(alias="ringsRemaining")

    model_config = ConfigDict(populate_by_name=True)


class GameRecord(BaseModel):
    """
    Complete record of a finished RingRift game.

    This is the canonical format for storing games for:
    - Training data generation (JSONL export)
    - Replay viewing
    - Statistical analysis
    - Historical record keeping

    The record is designed to be:
    - Space-efficient (no redundant board snapshots)
    - Self-contained (all info needed to replay without external state)
    - Forward-compatible (versioned schema with optional fields)
    """
    # Unique identifier
    id: str

    # Game configuration
    board_type: BoardType = Field(alias="boardType")
    num_players: int = Field(alias="numPlayers")
    rng_seed: int | None = Field(None, alias="rngSeed")
    is_rated: bool = Field(alias="isRated")

    # Players
    players: list[PlayerRecordInfo]

    # Game result
    winner: int | None = None
    outcome: GameOutcome
    final_score: FinalScore = Field(alias="finalScore")

    # Timing
    started_at: datetime = Field(alias="startedAt")
    ended_at: datetime = Field(alias="endedAt")
    total_moves: int = Field(alias="totalMoves")
    total_duration_ms: int = Field(alias="totalDurationMs")

    # Move history
    moves: list[MoveRecord]

    # Record metadata
    metadata: GameRecordMetadata

    # Optional extended data
    initial_state_hash: str | None = Field(None, alias="initialStateHash")
    final_state_hash: str | None = Field(None, alias="finalStateHash")
    progress_snapshots: list[ProgressSnapshot] | None = Field(
        None, alias="progressSnapshots"
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line for training data pipelines."""
        import json
        return json.dumps(self.model_dump(by_alias=True, mode="json"))

    @classmethod
    def from_jsonl_line(cls, line: str) -> "GameRecord":
        """Deserialize from a JSONL line."""
        import json
        data = json.loads(line)
        return cls.model_validate(data)


# ────────────────────────────────────────────────────────────────────────────
# RingRift Notation (RRN)
# ────────────────────────────────────────────────────────────────────────────

class RRNCoordinate(BaseModel):
    """
    Coordinate representation for RingRift Notation.

    Square boards use algebraic notation: a1-h8 (8x8) or a1-s19 (19x19)
    Hexagonal boards use axial notation: (x,y,z) or simplified (x,y)
    """
    notation: str
    position: Position

    @classmethod
    def from_position(cls, pos: Position, board_type: BoardType) -> "RRNCoordinate":
        """Convert a Position to RRN coordinate notation."""
        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            if pos.z is not None:
                notation = f"({pos.x},{pos.y},{pos.z})"
            else:
                notation = f"({pos.x},{pos.y})"
        else:
            # Square boards: column letter + row number (1-indexed)
            col_letter = chr(ord('a') + pos.x)
            row_number = pos.y + 1
            notation = f"{col_letter}{row_number}"
        return cls(notation=notation, position=pos)

    @classmethod
    def parse(cls, notation: str, board_type: BoardType) -> Position:
        """Parse RRN coordinate notation to Position."""
        notation = notation.strip()

        if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            # Hex notation: (x,y) or (x,y,z)
            if notation.startswith("(") and notation.endswith(")"):
                parts = notation[1:-1].split(",")
                if len(parts) == 2:
                    return Position(x=int(parts[0]), y=int(parts[1]))
                elif len(parts) == 3:
                    return Position(x=int(parts[0]), y=int(parts[1]), z=int(parts[2]))
            raise ValueError(f"Invalid hex coordinate: {notation}")
        else:
            # Square notation: letter + number (e.g., a1, h8, s19)
            if len(notation) < 2:
                raise ValueError(f"Invalid square coordinate: {notation}")
            col = ord(notation[0].lower()) - ord('a')
            row = int(notation[1:]) - 1
            return Position(x=col, y=row)


class RRNMove(BaseModel):
    """
    RingRift Notation representation of a single move.

    Notation format:
    - Placement: P{coord} or P{coord}x{count} for multi-ring
    - Movement: {from}-{to}
    - Capture: {from}x{target}-{to}
    - Chain capture: {from}x{target}-{to}+
    - Line processing: L{coord} (first marker in line)
    - Territory processing: T{coord} (representative position)
    - Skip: -
    - Swap sides: S

    Examples:
    - "Pa1"       - Place ring at a1
    - "e4-e6"     - Move stack from e4 to e6
    - "d4xd5-d6"  - Capture at d5, land at d6
    - "d4xd5-d6+" - Capture with continuation available
    - "La3"       - Process line starting at a3
    - "Tb2"       - Process territory region at b2
    - "-"         - Skip (placement/capture)
    - "S"         - Swap sides (pie rule)
    """
    notation: str
    move_type: MoveType = Field(alias="moveType")
    player: int

    @classmethod
    def from_move_record(
        cls, record: MoveRecord, board_type: BoardType
    ) -> "RRNMove":
        """Generate RRN notation from a MoveRecord."""
        notation = _generate_rrn(record, board_type)
        return cls(notation=notation, move_type=record.type, player=record.player)


def _generate_rrn(record: MoveRecord, board_type: BoardType) -> str:
    """Generate RingRift Notation string from a MoveRecord."""

    def pos_to_str(pos: Position | None) -> str:
        if pos is None:
            return "?"
        return RRNCoordinate.from_position(pos, board_type).notation

    t = record.type

    if t == MoveType.PLACE_RING:
        base = f"P{pos_to_str(record.to)}"
        if record.placement_count and record.placement_count > 1:
            base += f"x{record.placement_count}"
        return base

    elif t == MoveType.SKIP_PLACEMENT:
        return "-"

    elif t == MoveType.SWAP_SIDES:
        return "S"

    elif t in (MoveType.MOVE_STACK, MoveType.MOVE_RING):
        return f"{pos_to_str(record.from_pos)}-{pos_to_str(record.to)}"

    elif t == MoveType.BUILD_STACK:
        return f"{pos_to_str(record.from_pos)}>{pos_to_str(record.to)}"

    elif t == MoveType.OVERTAKING_CAPTURE:
        return f"{pos_to_str(record.from_pos)}x{pos_to_str(record.capture_target)}-{pos_to_str(record.to)}"

    elif t == MoveType.CONTINUE_CAPTURE_SEGMENT:
        # Chain capture continuation
        return f"{pos_to_str(record.from_pos)}x{pos_to_str(record.capture_target)}-{pos_to_str(record.to)}+"

    elif t == MoveType.PROCESS_LINE:
        # Use first marker position from formed line
        if record.formed_lines and len(record.formed_lines) > 0:
            first_pos = record.formed_lines[0].positions[0]
            return f"L{pos_to_str(first_pos)}"
        return "L?"

    elif t == MoveType.CHOOSE_LINE_REWARD:
        # O1 for option 1 (collapse all), O2 for option 2 (minimum collapse)
        # Determine from collapsed_markers count vs formed_line length
        if record.formed_lines and record.collapsed_markers:
            line_len = len(record.formed_lines[0].positions)
            collapsed_len = len(record.collapsed_markers)
            if collapsed_len == line_len:
                return "O1"  # Option 1: collapse all
            else:
                return "O2"  # Option 2: minimum collapse
        return "O?"

    elif t == MoveType.PROCESS_TERRITORY_REGION:
        # Use representative position from disconnected region
        if record.disconnected_regions and len(record.disconnected_regions) > 0:
            rep_pos = record.disconnected_regions[0].spaces[0]
            return f"T{pos_to_str(rep_pos)}"
        return "T?"

    elif t == MoveType.ELIMINATE_RINGS_FROM_STACK:
        return f"E{pos_to_str(record.to)}"

    elif t == MoveType.RECOVERY_SLIDE:
        # Recovery slide: R{from}-{to} optionally with option indicator
        base = f"R{pos_to_str(record.from_pos)}-{pos_to_str(record.to)}"
        if hasattr(record, 'recovery_option') and record.recovery_option == 2:
            base += "/2"  # Option 2 (free minimum collapse)
        return base

    else:
        # Fallback for legacy/unknown move types
        return f"?{t}"


def parse_rrn_move(notation: str, board_type: BoardType) -> tuple[MoveType, Position | None, Position | None]:
    """
    Parse a RingRift Notation move string.

    Returns (move_type, from_position, to_position).
    For moves without spatial coordinates, positions may be None.
    """
    notation = notation.strip()

    if notation == "-":
        return (MoveType.SKIP_PLACEMENT, None, None)

    if notation == "S":
        return (MoveType.SWAP_SIDES, None, None)

    if notation.startswith("P"):
        # Placement: P{coord} or P{coord}x{count}
        rest = notation[1:]
        if "x" in rest:
            coord_part = rest.split("x")[0]
        else:
            coord_part = rest
        pos = RRNCoordinate.parse(coord_part, board_type)
        return (MoveType.PLACE_RING, None, pos)

    if notation.startswith("L"):
        # Line processing
        pos = RRNCoordinate.parse(notation[1:], board_type)
        return (MoveType.PROCESS_LINE, None, pos)

    if notation.startswith("T"):
        # Territory processing
        pos = RRNCoordinate.parse(notation[1:], board_type)
        return (MoveType.PROCESS_TERRITORY_REGION, None, pos)

    if notation.startswith("E"):
        # Ring elimination
        pos = RRNCoordinate.parse(notation[1:], board_type)
        return (MoveType.ELIMINATE_RINGS_FROM_STACK, None, pos)

    if notation in ("O1", "O2"):
        return (MoveType.CHOOSE_LINE_REWARD, None, None)

    if notation.startswith("R"):
        # Recovery slide: R{from}-{to} or R{from}-{to}/2
        rest = notation[1:].replace("/2", "")  # Strip option indicator
        parts = rest.split("-")
        from_pos = RRNCoordinate.parse(parts[0], board_type)
        to_pos = RRNCoordinate.parse(parts[1], board_type)
        return (MoveType.RECOVERY_SLIDE, from_pos, to_pos)

    # Movement or capture: {from}-{to} or {from}x{target}-{to}
    if "x" in notation:
        # Capture
        parts = notation.replace("+", "").split("x")
        from_str = parts[0]
        rest = parts[1]
        target_to = rest.split("-")
        # target = target_to[0]  # We don't return target separately
        to_str = target_to[1] if len(target_to) > 1 else target_to[0]

        from_pos = RRNCoordinate.parse(from_str, board_type)
        to_pos = RRNCoordinate.parse(to_str, board_type)

        if notation.endswith("+"):
            return (MoveType.CONTINUE_CAPTURE_SEGMENT, from_pos, to_pos)
        return (MoveType.OVERTAKING_CAPTURE, from_pos, to_pos)

    if "-" in notation:
        # Movement
        parts = notation.split("-")
        from_pos = RRNCoordinate.parse(parts[0], board_type)
        to_pos = RRNCoordinate.parse(parts[1], board_type)
        return (MoveType.MOVE_STACK, from_pos, to_pos)

    if ">" in notation:
        # Build stack
        parts = notation.split(">")
        from_pos = RRNCoordinate.parse(parts[0], board_type)
        to_pos = RRNCoordinate.parse(parts[1], board_type)
        return (MoveType.BUILD_STACK, from_pos, to_pos)

    raise ValueError(f"Unable to parse RRN: {notation}")


def game_record_to_rrn(record: GameRecord) -> str:
    """
    Convert a complete GameRecord to RRN notation string.

    Format: {board_type}:{num_players}:{seed?}:{moves...}
    Example: "square8:2:12345:Pa1 Pa8 d4-d6 d8-d4 d6xd5-d4"
    """
    header_parts = [
        record.board_type.value,
        str(record.num_players),
    ]
    if record.rng_seed is not None:
        header_parts.append(str(record.rng_seed))
    else:
        header_parts.append("_")

    move_strs = []
    for move in record.moves:
        rrn_move = RRNMove.from_move_record(move, record.board_type)
        move_strs.append(rrn_move.notation)

    header = ":".join(header_parts)
    moves = " ".join(move_strs)
    return f"{header}:{moves}"


def rrn_to_moves(
    rrn_string: str
) -> tuple[BoardType, int, int | None, list[tuple[MoveType, Position | None, Position | None]]]:
    """
    Parse an RRN string to extract board config and move list.

    Returns (board_type, num_players, rng_seed, moves).
    """
    parts = rrn_string.split(":", 3)
    if len(parts) < 4:
        raise ValueError(f"Invalid RRN format: {rrn_string}")

    board_type = BoardType(parts[0])
    num_players = int(parts[1])
    rng_seed = int(parts[2]) if parts[2] != "_" else None
    moves_str = parts[3]

    moves = []
    for move_notation in moves_str.split():
        parsed = parse_rrn_move(move_notation, board_type)
        moves.append(parsed)

    return (board_type, num_players, rng_seed, moves)
