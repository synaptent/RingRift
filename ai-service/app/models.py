"""
Pydantic Models for RingRift Game State
Mirrors TypeScript types from src/shared/types/game.ts
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime


class BoardType(str, Enum):
    """Board type enumeration"""
    SQUARE8 = "square8"
    SQUARE19 = "square19"
    HEXAGONAL = "hexagonal"


class GamePhase(str, Enum):
    """Game phase enumeration"""
    RING_PLACEMENT = "ring_placement"
    MOVEMENT = "movement"
    CAPTURE = "capture"
    LINE_PROCESSING = "line_processing"
    TERRITORY_PROCESSING = "territory_processing"


class GameStatus(str, Enum):
    """Game status enumeration"""
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    PAUSED = "paused"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


class AIType(str, Enum):
    """AI type enumeration"""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    MINIMAX = "minimax"
    MCTS = "mcts"


class Position(BaseModel):
    """Board position (2D or 3D for hexagonal)"""
    x: int
    y: int
    z: Optional[int] = None
    
    class Config:
        frozen = True

    def to_key(self) -> str:
        """Convert position to string key"""
        if self.z is not None:
            return f"{self.x},{self.y},{self.z}"
        return f"{self.x},{self.y}"


class LineInfo(BaseModel):
    """Information about a formed line"""
    positions: List[Position]
    player: int
    length: int
    direction: Position  # Direction vector

    class Config:
        populate_by_name = True


class Territory(BaseModel):
    """Information about a territory region"""
    spaces: List[Position]
    controlling_player: int = Field(alias="controllingPlayer")
    is_disconnected: bool = Field(alias="isDisconnected")

    class Config:
        populate_by_name = True


class RingStack(BaseModel):
    """Ring stack on the board"""
    position: Position
    rings: List[int]  # Player numbers from bottom to top
    stack_height: int = Field(alias="stackHeight")
    cap_height: int = Field(alias="capHeight")
    controlling_player: int = Field(alias="controllingPlayer")
    
    class Config:
        populate_by_name = True


class MarkerInfo(BaseModel):
    """Marker information"""
    player: int
    position: Position
    type: str  # 'regular' or 'collapsed'


class Player(BaseModel):
    """Player state"""
    id: str
    username: str
    type: str
    player_number: int = Field(alias="playerNumber")
    is_ready: bool = Field(alias="isReady")
    time_remaining: int = Field(alias="timeRemaining")
    ai_difficulty: Optional[int] = Field(None, alias="aiDifficulty")
    rings_in_hand: int = Field(alias="ringsInHand")
    eliminated_rings: int = Field(alias="eliminatedRings")
    territory_spaces: int = Field(alias="territorySpaces")
    
    class Config:
        populate_by_name = True


class TimeControl(BaseModel):
    """Time control settings"""
    initial_time: int = Field(alias="initialTime")
    increment: int
    type: str
    
    class Config:
        populate_by_name = True


class Move(BaseModel):
    """Move representation.

    This mirrors the TypeScript Move type closely enough for the AI
    service to participate in both movement and ring placement:
    - `type` is a string MoveType (e.g. 'place_ring', 'move_ring',
      'move_stack', 'overtaking_capture').
    - For placement moves, `placement_count` and `placed_on_stack`
      carry the multi-ring and stacking metadata introduced on the
      backend.
    """

    id: str
    type: str
    player: int
    from_pos: Optional[Position] = Field(None, alias="from")
    to: Position
    # Capture metadata
    capture_target: Optional[Position] = Field(None, alias="captureTarget")
    # Ring placement specific metadata (optional for non-placement moves)
    placed_on_stack: Optional[bool] = Field(None, alias="placedOnStack")
    placement_count: Optional[int] = Field(None, alias="placementCount")
    timestamp: datetime
    think_time: int = Field(alias="thinkTime")
    move_number: int = Field(alias="moveNumber")
    
    class Config:
        populate_by_name = True
        frozen = True


class BoardState(BaseModel):
    """Current board state"""
    type: BoardType
    size: int
    stacks: Dict[str, RingStack] = {}
    markers: Dict[str, MarkerInfo] = {}
    collapsed_spaces: Dict[str, int] = Field(
        default_factory=dict, alias="collapsedSpaces"
    )
    eliminated_rings: Dict[str, int] = Field(
        default_factory=dict, alias="eliminatedRings"
    )
    formed_lines: List[LineInfo] = Field(
        default_factory=list, alias="formedLines"
    )
    territories: Dict[str, Territory] = Field(
        default_factory=dict, alias="territories"
    )

    class Config:
        populate_by_name = True


class GameState(BaseModel):
    """Complete game state"""
    id: str
    board_type: BoardType = Field(alias="boardType")
    board: BoardState
    players: List[Player]
    current_phase: GamePhase = Field(alias="currentPhase")
    current_player: int = Field(alias="currentPlayer")
    move_history: List[Move] = Field(default_factory=list, alias="moveHistory")
    time_control: TimeControl = Field(alias="timeControl")
    spectators: List[str] = Field(default_factory=list)
    game_status: GameStatus = Field(alias="gameStatus")
    winner: Optional[int] = None
    created_at: datetime = Field(alias="createdAt")
    last_move_at: datetime = Field(alias="lastMoveAt")
    is_rated: bool = Field(alias="isRated")
    max_players: int = Field(alias="maxPlayers")
    total_rings_in_play: int = Field(alias="totalRingsInPlay")
    total_rings_eliminated: int = Field(alias="totalRingsEliminated")
    victory_threshold: int = Field(alias="victoryThreshold")
    territory_victory_threshold: int = Field(alias="territoryVictoryThreshold")
    
    class Config:
        populate_by_name = True


class AIConfig(BaseModel):
    """AI configuration"""
    difficulty: int = Field(ge=1, le=10)
    think_time: Optional[int] = Field(None, alias="thinkTime")
    randomness: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        populate_by_name = True


class LineRewardChoiceOption(str, Enum):
    """Line reward choice options, mirroring TypeScript LineRewardChoice."""
    OPTION_1 = "option_1_collapse_all_and_eliminate"
    OPTION_2 = "option_2_min_collapse_no_elimination"


class LineRewardChoiceRequest(BaseModel):
    """Request model for AI-backed line reward choices.

    This mirrors the core fields of the TypeScript LineRewardChoice plus
    the AI configuration metadata the service needs. GameState is not
    required for the initial implementation but can be added later if we
    want the Python service to make more context-aware decisions.
    """

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: List[LineRewardChoiceOption]

    class Config:
        populate_by_name = True


class LineRewardChoiceResponse(BaseModel):
    """Response model for AI-backed line reward choices."""

    selected_option: LineRewardChoiceOption = Field(alias="selectedOption")
    ai_type: str = Field(alias="aiType")
    difficulty: int

    class Config:
        populate_by_name = True


class RingEliminationChoiceOption(BaseModel):
    """Option for ring elimination choice.

    Mirrors the TypeScript RingEliminationChoice option shape:
    { stackPosition, capHeight, totalHeight }.
    """

    stack_position: Position = Field(alias="stackPosition")
    cap_height: int = Field(alias="capHeight")
    total_height: int = Field(alias="totalHeight")

    class Config:
        populate_by_name = True


class RingEliminationChoiceRequest(BaseModel):
    """Request model for AI-backed ring elimination choices.

    Carries the same metadata as LineRewardChoiceRequest plus the
    ring-elimination specific option list.
    """

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: List[RingEliminationChoiceOption]

    class Config:
        populate_by_name = True


class RingEliminationChoiceResponse(BaseModel):
    """Response model for AI-backed ring elimination choices."""

    selected_option: RingEliminationChoiceOption = Field(
        alias="selectedOption"
    )
    ai_type: str = Field(alias="aiType")
    difficulty: int

    class Config:
        populate_by_name = True


class RegionOrderChoiceOption(BaseModel):
    """Option for region order choice.

    Mirrors the TypeScript RegionOrderChoice option shape:
    { regionId, size, representativePosition }.
    """

    region_id: str = Field(alias="regionId")
    size: int
    representative_position: Position = Field(alias="representativePosition")

    class Config:
        populate_by_name = True


class ProgressSnapshot(BaseModel):
    """
    Canonical, engine-agnostic progress snapshot used for invariant checks
    and history entries. S is defined as markers + collapsed + eliminated.
    """
    markers: int
    collapsed: int
    eliminated: int
    S: int


class ChainCaptureSegment(BaseModel):
    """Segment of a chain capture"""
    from_pos: Position = Field(alias="from")
    target: Position
    landing: Position
    captured_cap_height: int = Field(alias="capturedCapHeight")

    class Config:
        populate_by_name = True


class ChainCaptureState(BaseModel):
    """State of an ongoing chain capture"""
    player_number: int = Field(alias="playerNumber")
    start_position: Position = Field(alias="startPosition")
    current_position: Position = Field(alias="currentPosition")
    segments: List[ChainCaptureSegment]
    available_moves: List[Move] = Field(alias="availableMoves")
    visited_positions: List[str] = Field(alias="visitedPositions")

    class Config:
        populate_by_name = True


class RegionOrderChoiceRequest(BaseModel):
    """Request model for AI-backed region order choices."""

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: List[RegionOrderChoiceOption]

    class Config:
        populate_by_name = True


class RegionOrderChoiceResponse(BaseModel):
    """Response model for AI-backed region order choices."""

    selected_option: RegionOrderChoiceOption = Field(alias="selectedOption")
    ai_type: str = Field(alias="aiType")
    difficulty: int

    class Config:
        populate_by_name = True
