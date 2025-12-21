"""Core test fixtures and factory functions for RingRift.

This module provides reusable factory functions for creating test objects.
Unlike pytest fixtures, these are plain functions that can be called directly
from any test or fixture.

Usage:
    from app.testing.fixtures import create_game_state, create_player

    def test_something():
        game = create_game_state(num_players=4)
        player = create_player(player_number=1)
        ...

    # Or in conftest.py as fixtures:
    @pytest.fixture
    def game_state():
        return create_game_state()
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import (
        BoardState,
        GameState,
        Move,
        Player,
        Position,
        RingStack,
    )

__all__ = [
    "create_board_state",
    "create_game_state",
    "create_move",
    "create_player",
    "create_position",
    "create_ring_stack",
]


def create_position(x: int = 0, y: int = 0) -> "Position":
    """Create a Position instance.

    Args:
        x: X coordinate (column)
        y: Y coordinate (row)

    Returns:
        Position instance
    """
    from app.models import Position

    return Position(x=x, y=y)


def create_player(
    player_number: int = 1,
    username: str | None = None,
    player_type: str = "human",
    rings_in_hand: int = 10,
    eliminated_rings: int = 0,
    territory_spaces: int = 0,
    time_remaining: int = 600,
    ai_difficulty: int | None = None,
    is_ready: bool = True,
) -> "Player":
    """Create a Player instance with sensible defaults.

    Args:
        player_number: Player number (1-indexed)
        username: Player username (defaults to "Player{number}")
        player_type: "human" or "ai"
        rings_in_hand: Number of rings available to place
        eliminated_rings: Number of rings eliminated
        territory_spaces: Territory control count
        time_remaining: Time remaining in seconds
        ai_difficulty: AI difficulty level (1-5) if AI player
        is_ready: Whether player is ready

    Returns:
        Player instance
    """
    from app.models import Player

    return Player(
        id=f"p{player_number}",
        username=username or f"Player{player_number}",
        type=player_type,
        playerNumber=player_number,
        isReady=is_ready,
        timeRemaining=time_remaining,
        ringsInHand=rings_in_hand,
        eliminatedRings=eliminated_rings,
        territorySpaces=territory_spaces,
        aiDifficulty=ai_difficulty,
    )


def create_ring_stack(
    x: int,
    y: int,
    rings: list[int],
    controlling_player: int | None = None,
) -> "RingStack":
    """Create a RingStack instance.

    Args:
        x: X coordinate
        y: Y coordinate
        rings: List of player numbers representing rings (bottom to top)
        controlling_player: Override controller (defaults to topmost contiguous)

    Returns:
        RingStack instance

    Raises:
        ValueError: If rings list is empty
    """
    from app.models import Position, RingStack

    if not rings:
        raise ValueError("Stack must have at least one ring")

    controller = controlling_player if controlling_player is not None else rings[0]

    # Calculate cap height (contiguous rings from top controlled by same player)
    cap_height = 0
    for ring in rings:
        if ring == controller:
            cap_height += 1
        else:
            break

    return RingStack(
        position=Position(x=x, y=y),
        rings=rings,
        stackHeight=len(rings),
        capHeight=cap_height,
        controllingPlayer=controller,
    )


def create_move(
    player: int = 1,
    x: int = 0,
    y: int = 0,
    move_type: str = "place_ring",
    move_number: int = 1,
    from_pos: tuple[int, int] | None = None,
    count: int = 1,
) -> "Move":
    """Create a Move instance.

    Args:
        player: Player number making the move
        x: Target X coordinate
        y: Target Y coordinate
        move_type: Type of move (place_ring, move_stack, overtaking_capture, etc.)
        move_number: Sequential move number
        from_pos: Source position as (x, y) tuple for moves with origin
        count: Ring count for place_ring moves

    Returns:
        Move instance
    """
    from app.models import Move, MoveType, Position

    # Convert string to MoveType enum if needed
    if isinstance(move_type, str):
        move_type = MoveType(move_type)

    move_kwargs = {
        "id": f"m-{player}-{x}-{y}-{move_number}",
        "type": move_type,
        "player": player,
        "to": Position(x=x, y=y),
        "timestamp": datetime.now(),
        "thinkTime": 0,
        "moveNumber": move_number,
    }

    if from_pos is not None:
        move_kwargs["from"] = Position(x=from_pos[0], y=from_pos[1])

    if move_type == MoveType.PLACE_RING:
        move_kwargs["count"] = count

    return Move(**move_kwargs)


def create_board_state(
    board_type: str = "square8",
    stacks: dict[str, "RingStack"] | None = None,
    markers: dict | None = None,
    collapsed_spaces: dict[str, int] | None = None,
    eliminated_rings: dict[int, int] | None = None,
) -> "BoardState":
    """Create a BoardState instance.

    Args:
        board_type: Board type (square8, square19, hex8, hexagonal)
        stacks: Ring stacks by position key
        markers: Markers by position key
        collapsed_spaces: Collapsed spaces by position key
        eliminated_rings: Eliminated rings per player

    Returns:
        BoardState instance
    """
    from app.models import BoardState, BoardType

    # Convert string to BoardType enum if needed
    if isinstance(board_type, str):
        board_type = BoardType(board_type)

    # Determine size from type
    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    elif board_type == BoardType.HEX8:
        size = 8  # Radius-4 hex (61 cells)
    else:
        size = 25  # Full hexagonal

    return BoardState(
        type=board_type,
        size=size,
        stacks=stacks or {},
        markers=markers or {},
        collapsedSpaces=collapsed_spaces or {},
        eliminatedRings=eliminated_rings or {},
    )


def create_game_state(
    game_id: str = "test-game",
    board_type: str = "square8",
    num_players: int = 2,
    current_player: int = 1,
    current_phase: str = "ring_placement",
    game_status: str = "active",
    players: list["Player"] | None = None,
    board: "BoardState" | None = None,
    move_history: list["Move"] | None = None,
    rings_in_hand: int = 10,
    total_rings_in_play: int = 36,
    rng_seed: int | None = None,
) -> "GameState":
    """Create a GameState instance with full customization.

    Args:
        game_id: Unique game identifier
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2-4)
        current_player: Current player number
        current_phase: Game phase (ring_placement, movement, capture, etc.)
        game_status: Game status (active, completed, waiting, etc.)
        players: List of players (auto-generated if None)
        board: Board state (auto-generated if None)
        move_history: List of moves
        rings_in_hand: Rings per player (for auto-generated players)
        total_rings_in_play: Total rings in play
        rng_seed: Random seed for reproducibility

    Returns:
        GameState instance
    """
    from app.models import BoardType, GamePhase, GameState, GameStatus, TimeControl

    # Convert strings to enums if needed
    if isinstance(board_type, str):
        board_type = BoardType(board_type)
    if isinstance(current_phase, str):
        current_phase = GamePhase(current_phase)
    if isinstance(game_status, str):
        game_status = GameStatus(game_status)

    # Create players if not provided
    if players is None:
        players = [
            create_player(i, rings_in_hand=rings_in_hand)
            for i in range(1, num_players + 1)
        ]

    # Create board if not provided
    if board is None:
        board = create_board_state(board_type)

    return GameState(
        id=game_id,
        boardType=board_type,
        rngSeed=rng_seed,
        board=board,
        players=players,
        currentPhase=current_phase,
        currentPlayer=current_player,
        moveHistory=move_history or [],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=game_status,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=total_rings_in_play,
        totalRingsEliminated=0,
        victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer for square8
        territoryVictoryThreshold=33,
    )
