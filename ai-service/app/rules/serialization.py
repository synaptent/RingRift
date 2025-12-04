"""
Serialization utilities for contract-based cross-language parity testing.

This module provides serialization/deserialization functions that match the
TypeScript format from src/shared/engine/contracts/serialization.ts, enabling
Python contract tests to load test vectors and compare results.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from ..models import (
    BoardState,
    BoardType,
    GamePhase,
    GameStatus,
    GameState,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)


# ============================================================================
# Position serialization
# ============================================================================


def serialize_position(pos: Position) -> Dict[str, int]:
    """Serialize a Position to JSON-compatible dict."""
    result: Dict[str, int] = {"x": pos.x, "y": pos.y}
    if pos.z is not None:
        result["z"] = pos.z
    return result


def deserialize_position(data: Dict[str, Any]) -> Position:
    """Deserialize a Position from JSON dict."""
    return Position(
        x=data["x"],
        y=data["y"],
        z=data.get("z"),
    )


# ============================================================================
# RingStack serialization
# ============================================================================


def serialize_stack(stack: RingStack) -> Dict[str, Any]:
    """Serialize a RingStack to JSON-compatible dict."""
    return {
        "position": serialize_position(stack.position),
        "rings": list(stack.rings),
        "stackHeight": stack.stack_height,
        "capHeight": stack.cap_height,
        "controllingPlayer": stack.controlling_player,
    }


def deserialize_stack(data: Dict[str, Any]) -> RingStack:
    """Deserialize a RingStack from JSON dict."""
    return RingStack(
        position=deserialize_position(data["position"]),
        rings=data["rings"],
        stackHeight=data["stackHeight"],
        capHeight=data["capHeight"],
        controllingPlayer=data["controllingPlayer"],
    )


# ============================================================================
# Marker serialization
# ============================================================================


def serialize_marker(marker: MarkerInfo) -> Dict[str, Any]:
    """Serialize a MarkerInfo to JSON-compatible dict."""
    return {
        "position": serialize_position(marker.position),
        "player": marker.player,
    }


def deserialize_marker(data: Dict[str, Any]) -> MarkerInfo:
    """Deserialize a MarkerInfo from JSON dict."""
    return MarkerInfo(
        position=deserialize_position(data["position"]),
        player=data["player"],
        type=data.get("type", "regular"),
    )


# ============================================================================
# BoardState serialization
# ============================================================================


def serialize_board_state(board: BoardState) -> Dict[str, Any]:
    """Serialize a BoardState to JSON-compatible dict."""
    stacks: Dict[str, Any] = {}
    for key, stack in board.stacks.items():
        stacks[key] = serialize_stack(stack)

    markers: Dict[str, Any] = {}
    for key, marker in board.markers.items():
        markers[key] = serialize_marker(marker)

    collapsed: Dict[str, int] = {}
    for key, player in board.collapsed_spaces.items():
        collapsed[key] = player

    eliminated: Dict[str, int] = {}
    for key, count in board.eliminated_rings.items():
        eliminated[key] = count

    return {
        "type": board.type.value,
        "size": board.size,
        "stacks": stacks,
        "markers": markers,
        "collapsedSpaces": collapsed,
        "eliminatedRings": eliminated,
    }


def deserialize_board_state(data: Dict[str, Any]) -> BoardState:
    """Deserialize a BoardState from JSON dict."""
    # Parse board type
    board_type_str = data.get("type", "square8")
    board_type = BoardType(board_type_str)

    # Parse stacks
    stacks: Dict[str, RingStack] = {}
    for key, stack_data in data.get("stacks", {}).items():
        stacks[key] = deserialize_stack(stack_data)

    # Parse markers
    markers: Dict[str, MarkerInfo] = {}
    for key, marker_data in data.get("markers", {}).items():
        markers[key] = deserialize_marker(marker_data)

    # Parse collapsed spaces
    collapsed_spaces: Dict[str, int] = {}
    for key, player in data.get("collapsedSpaces", {}).items():
        collapsed_spaces[key] = player

    # Parse eliminated rings
    eliminated_rings: Dict[str, int] = {}
    for key, count in data.get("eliminatedRings", {}).items():
        eliminated_rings[key] = count

    return BoardState(
        type=board_type,
        size=data.get("size", 8),
        stacks=stacks,
        markers=markers,
        collapsedSpaces=collapsed_spaces,
        eliminatedRings=eliminated_rings,
        formedLines=[],
        territories={},
    )


# ============================================================================
# Player serialization
# ============================================================================


def serialize_player(player: Player) -> Dict[str, Any]:
    """Serialize a Player to JSON-compatible dict."""
    return {
        "playerNumber": player.player_number,
        "ringsInHand": player.rings_in_hand,
        "eliminatedRings": player.eliminated_rings,
        "territorySpaces": player.territory_spaces,
        "isActive": True,  # Contract vectors use isActive flag
    }


def deserialize_player(data: Dict[str, Any], index: int) -> Player:
    """Deserialize a Player from JSON dict."""
    player_number = data.get("playerNumber", index + 1)
    return Player(
        id=str(player_number),
        username=f"Player{player_number}",
        type="human",
        playerNumber=player_number,
        isReady=True,
        timeRemaining=600000,
        aiDifficulty=None,
        ringsInHand=data.get("ringsInHand", 18),
        eliminatedRings=data.get("eliminatedRings", 0),
        territorySpaces=data.get("territorySpaces", 0),
    )


# ============================================================================
# Move serialization
# ============================================================================


def serialize_move(move: Move) -> Dict[str, Any]:
    """Serialize a Move to JSON-compatible dict."""
    result: Dict[str, Any] = {
        "id": move.id,
        "type": move.type.value,
        "player": move.player,
        "to": serialize_position(move.to),
        "timestamp": (
            move.timestamp.isoformat()
            if isinstance(move.timestamp, datetime)
            else str(move.timestamp)
        ),
        "thinkTime": move.think_time,
        "moveNumber": move.move_number,
    }

    if move.from_pos:
        result["from"] = serialize_position(move.from_pos)

    if move.capture_target:
        result["captureTarget"] = serialize_position(move.capture_target)

    if move.placement_count is not None and move.placement_count > 0:
        result["placementCount"] = move.placement_count

    if move.placed_on_stack is not None:
        result["placedOnStack"] = move.placed_on_stack

    return result


def deserialize_move(data: Dict[str, Any]) -> Optional[Move]:
    """Deserialize a Move from JSON dict.

    This mirrors the TypeScript contract move format from
    src/shared/engine/contracts/serialization.ts and preserves
    decision-phase metadata such as formedLines, disconnectedRegions,
    collapsedMarkers, and eliminatedRings so that parity tests can
    reconstruct rich decision moves exactly.

    Returns None for empty/invalid move data.
    """
    # Return None for empty move data (e.g., multi-phase vectors that use initialMove)
    if not data or ("type" not in data and "to" not in data):
        return None

    # Parse move type
    move_type_str = data.get("type", "place_ring")
    move_type = MoveType(move_type_str)

    # Parse timestamp
    timestamp_str = data.get("timestamp", datetime.now().isoformat())
    if isinstance(timestamp_str, str):
        try:
            ts_clean = timestamp_str.replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(ts_clean)
        except ValueError:
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    # Parse positions - 'to' is optional for some move types (e.g., process_line)
    to_pos = deserialize_position(data["to"]) if "to" in data else None
    from_pos = (
        deserialize_position(data["from"])
        if "from" in data
        else None
    )
    capture_target = (
        deserialize_position(data["captureTarget"])
        if "captureTarget" in data
        else None
    )

    # Build move kwargs dict to allow "from" alias and preserve
    # additional decision/metadata fields. Pydantic will coerce the
    # raw dict/list payloads into the appropriate typed models based
    # on field aliases (e.g. formedLines → LineInfo[], disconnectedRegions → Territory[]).
    move_kwargs: Dict[str, Any] = {
        "id": data.get("id", "test-move"),
        "type": move_type,
        "player": data.get("player", 1),
        "timestamp": timestamp,
        "thinkTime": data.get("thinkTime", 0),
        "moveNumber": data.get("moveNumber", 1),
        "captureTarget": capture_target,
        "placementCount": data.get("placementCount"),
        "placedOnStack": data.get("placedOnStack"),
    }

    # Some move types (e.g., forced_elimination, swap_sides) don't have a 'to' position
    if to_pos is not None:
        move_kwargs["to"] = to_pos
    if from_pos is not None:
        move_kwargs["from"] = from_pos

    # Preserve line / territory decision metadata when present.
    if "formedLines" in data:
        move_kwargs["formedLines"] = data["formedLines"]
    if "collapsedMarkers" in data:
        move_kwargs["collapsedMarkers"] = data["collapsedMarkers"]
    if "claimedTerritory" in data:
        move_kwargs["claimedTerritory"] = data["claimedTerritory"]
    if "disconnectedRegions" in data:
        move_kwargs["disconnectedRegions"] = data["disconnectedRegions"]
    if "eliminatedRings" in data:
        move_kwargs["eliminatedRings"] = data["eliminatedRings"]

    return Move(**move_kwargs)  # type: ignore[arg-type]


# ============================================================================
# GameState serialization
# ============================================================================


def serialize_game_state(state: GameState) -> Dict[str, Any]:
    """Serialize a GameState to JSON-compatible dict."""
    return {
        "gameId": state.id,
        "board": serialize_board_state(state.board),
        "players": [serialize_player(p) for p in state.players],
        "currentPlayer": state.current_player,
        "currentPhase": state.current_phase.value,
        "turnNumber": len(state.move_history) + 1,
        "moveHistory": [serialize_move(m) for m in state.move_history],
        "gameStatus": state.game_status.value,
        "victoryThreshold": state.victory_threshold,
        "territoryVictoryThreshold": state.territory_victory_threshold,
        "winner": state.winner,
    }


def deserialize_game_state(data: Dict[str, Any]) -> GameState:
    """Deserialize a GameState from JSON dict.

    This function parses the contract test vector format and produces a
    GameState compatible with the Python GameEngine.apply_move function.
    """
    # Parse board
    board = deserialize_board_state(data.get("board", {}))

    # Parse players
    players_data = data.get("players", [])
    players: List[Player] = []
    for i, pdata in enumerate(players_data):
        players.append(deserialize_player(pdata, i))

    # Ensure we have at least 2 players for the engine
    while len(players) < 2:
        idx = len(players)
        players.append(
            Player(
                id=str(idx + 1),
                username=f"Player{idx + 1}",
                type="human",
                playerNumber=idx + 1,
                isReady=True,
                timeRemaining=600000,
                aiDifficulty=None,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
            )
        )

    # Parse current phase
    phase_str = data.get("currentPhase", "ring_placement")
    current_phase = GamePhase(phase_str)

    # Parse game status
    status_str = data.get("gameStatus", "active")
    game_status = GameStatus(status_str)

    # Parse timestamps
    now = datetime.now()

    # Parse move history (usually empty in test vectors)
    move_history: List[Move] = []
    for mdata in data.get("moveHistory", []):
        move_history.append(deserialize_move(mdata))

    # Build default time control
    time_control = TimeControl(
        initialTime=600000,
        increment=0,
        type="none",
    )

    return GameState(
        id=data.get("gameId", "test-game"),
        boardType=board.type,
        rngSeed=None,
        board=board,
        players=players,
        currentPhase=current_phase,
        currentPlayer=data.get("currentPlayer", 1),
        moveHistory=move_history,
        timeControl=time_control,
        spectators=[],
        gameStatus=game_status,
        winner=data.get("winner"),
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,  # Default for 2-player square8
        totalRingsEliminated=0,
        victoryThreshold=data.get("victoryThreshold", 19),
        territoryVictoryThreshold=data.get("territoryVictoryThreshold", 33),
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
    )


# ============================================================================
# Progress snapshot computation (S-invariant)
# ============================================================================


def compute_s_invariant(state: GameState) -> int:
    """Compute the S-invariant: markers + collapsed + eliminated.

    This matches the TypeScript computeProgressSnapshot semantics.
    """
    board = state.board
    marker_count = len(board.markers)
    collapsed_count = len(board.collapsed_spaces)
    eliminated_count = sum(board.eliminated_rings.values())
    return marker_count + collapsed_count + eliminated_count


def compute_stack_count(state: GameState) -> int:
    """Count total stacks on the board."""
    return len(state.board.stacks)


def compute_marker_count(state: GameState) -> int:
    """Count total markers on the board."""
    return len(state.board.markers)


def compute_collapsed_count(state: GameState) -> int:
    """Count total collapsed spaces on the board."""
    return len(state.board.collapsed_spaces)


# ============================================================================
# Test vector types
# ============================================================================


class ContractVectorAssertions:
    """Assertions for validating test vector output."""

    def __init__(self, data: Dict[str, Any]):
        self.current_player: Optional[int] = data.get("currentPlayer")
        self.current_phase: Optional[str] = data.get("currentPhase")
        self.game_status: Optional[str] = data.get("gameStatus")
        self.stack_count: Optional[int] = data.get("stackCount")
        self.marker_count: Optional[int] = data.get("markerCount")
        self.collapsed_count: Optional[int] = data.get("collapsedCount")
        self.s_invariant_delta: Optional[int] = data.get("sInvariantDelta")


# Alias for backwards compatibility
TestVectorAssertions = ContractVectorAssertions


class ContractVector:
    """Parsed test vector for contract testing."""

    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get("id", "unknown")
        self.version: str = data.get("version", "v2")
        self.category: str = data.get("category", "unknown")
        self.description: str = data.get("description", "")
        self.tags: List[str] = data.get("tags", [])
        self.source: str = data.get("source", "manual")
        self.created_at: str = data.get("createdAt", "")
        # Skip reason if present (for tests requiring unimplemented functionality)
        self.skip: Optional[str] = data.get("skip")

        # Input
        input_data = data.get("input", {})
        self.input_state: GameState = deserialize_game_state(
            input_data.get("state", {})
        )
        # Check for 'move' first, then fall back to 'initialMove' for multi-phase vectors
        move_data = input_data.get("move") or input_data.get("initialMove") or {}
        self.input_move: Optional[Move] = deserialize_move(move_data)
        # Expected chain sequence for multi-segment chain capture tests
        self.expected_chain_sequence: List[Dict[str, Any]] = input_data.get(
            "expectedChainSequence", []
        )

        # Expected output
        expected_data = data.get("expectedOutput", {})
        self.expected_status: str = expected_data.get("status", "complete")
        self.assertions = ContractVectorAssertions(
            expected_data.get("assertions", {})
        )

    def __repr__(self) -> str:
        return f"ContractVector({self.id})"


# Alias for backwards compatibility
TestVector = ContractVector


class ContractVectorBundle:
    """Collection of test vectors from a single file."""

    def __init__(self, data: Dict[str, Any]):
        self.version: str = data.get("version", "v2")
        self.generated: str = data.get("generated", "")
        self.count: int = data.get("count", 0)
        self.categories: List[str] = data.get("categories", [])
        self.description: str = data.get("description", "")
        self.vectors: List[ContractVector] = [
            ContractVector(v) for v in data.get("vectors", [])
        ]

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self):
        return iter(self.vectors)


# Alias for backwards compatibility
TestVectorBundle = ContractVectorBundle
