"""
Shared pytest fixtures for rules module tests.

Provides reusable fixtures for:
- Board type parametrization (square8, hex8, square19, hexagonal)
- Player count parametrization (2, 3, 4 players)
- GameState factory fixtures
- Stack and marker helpers
- Recovery-eligible state factories
- Capture chain state factories

Usage:
    @pytest.mark.parametrize("board_type", SQUARE_BOARDS)
    def test_something(board_type, game_state_factory):
        state = game_state_factory(board_type=board_type)
        ...
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pytest

from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MarkerInfo,
    Player,
    Position,
    RingStack,
    TimeControl,
)


# =============================================================================
# Board Type Collections for Parametrization
# =============================================================================

SQUARE_BOARDS = [BoardType.SQUARE8, BoardType.SQUARE19]
HEX_BOARDS = [BoardType.HEX8, BoardType.HEXAGONAL]
ALL_BOARD_TYPES = [BoardType.SQUARE8, BoardType.HEX8, BoardType.SQUARE19, BoardType.HEXAGONAL]

# Standard player counts
PLAYER_COUNTS = [2, 3, 4]
TWO_PLAYER = [2]
MULTIPLAYER = [3, 4]


# =============================================================================
# Board Size Helpers
# =============================================================================

def get_board_size(board_type: BoardType) -> int:
    """Get the standard size for a board type."""
    sizes = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEX8: 9,  # Radius 4 = 9x9 grid
        BoardType.HEXAGONAL: 25,  # Radius 12 = 25x25 grid
    }
    return sizes.get(board_type, 8)


def get_center_position(board_type: BoardType) -> Position:
    """Get the center position for a board type."""
    if board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
        return Position(x=0, y=0, z=0)
    size = get_board_size(board_type)
    center = size // 2
    return Position(x=center, y=center)


def get_valid_positions(board_type: BoardType, count: int = 5) -> list[Position]:
    """Get a list of valid positions for a board type."""
    if board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
        # Hex positions around center
        return [
            Position(x=0, y=0, z=0),
            Position(x=1, y=0, z=-1),
            Position(x=0, y=1, z=-1),
            Position(x=-1, y=1, z=0),
            Position(x=-1, y=0, z=1),
        ][:count]
    else:
        # Square positions around center
        size = get_board_size(board_type)
        center = size // 2
        return [
            Position(x=center, y=center),
            Position(x=center + 1, y=center),
            Position(x=center, y=center + 1),
            Position(x=center - 1, y=center),
            Position(x=center, y=center - 1),
        ][:count]


# =============================================================================
# State Manipulation Helpers
# =============================================================================

def add_stack(state: GameState, pos: Position, rings: list[int]) -> None:
    """
    Add a stack to the board.

    Args:
        state: GameState to modify
        pos: Position for the stack
        rings: List of player numbers from bottom to top (e.g., [1, 2, 1] means
               P1 at bottom, P2 in middle, P1 on top controlling)
    """
    if not rings:
        return

    controlling_player = rings[-1]
    cap_height = 0
    for r in reversed(rings):
        if r == controlling_player:
            cap_height += 1
        else:
            break

    pos_key = pos.to_key()
    state.board.stacks[pos_key] = RingStack(
        position=pos,
        rings=rings,
        controlling_player=controlling_player,
        stack_height=len(rings),
        cap_height=cap_height,
    )


def add_marker(state: GameState, pos: Position, player: int) -> None:
    """Add a marker to the board."""
    pos_key = pos.to_key()
    state.board.markers[pos_key] = MarkerInfo(
        position=pos,
        player=player,
        type="regular"
    )


def collapse_space(state: GameState, pos: Position) -> None:
    """Mark a space as collapsed."""
    pos_key = pos.to_key()
    state.board.collapsed_spaces[pos_key] = True


def clear_position(state: GameState, pos: Position) -> None:
    """Clear a position of stacks, markers, and collapsed state."""
    pos_key = pos.to_key()
    state.board.stacks.pop(pos_key, None)
    state.board.markers.pop(pos_key, None)
    state.board.collapsed_spaces.pop(pos_key, None)


# =============================================================================
# Player Factory
# =============================================================================

def create_player(
    player_number: int,
    rings_in_hand: int = 0,
    eliminated_rings: int = 0,
    territory_spaces: int = 0,
) -> Player:
    """Create a Player object for testing."""
    return Player(
        id=f"p{player_number}",
        username=f"player{player_number}",
        type="human",
        playerNumber=player_number,
        isReady=True,
        timeRemaining=60,
        aiDifficulty=None,
        ringsInHand=rings_in_hand,
        eliminatedRings=eliminated_rings,
        territorySpaces=territory_spaces,
    )


def create_players(
    num_players: int = 2,
    rings_per_player: int = 0,
) -> list[Player]:
    """Create a list of Player objects."""
    return [
        create_player(i + 1, rings_in_hand=rings_per_player)
        for i in range(num_players)
    ]


# =============================================================================
# GameState Factory
# =============================================================================

def create_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    current_player: int = 1,
    current_phase: GamePhase = GamePhase.MOVEMENT,
    rings_in_hand: int = 0,
) -> GameState:
    """
    Create a GameState for testing.

    Args:
        board_type: Type of board (SQUARE8, HEX8, etc.)
        num_players: Number of players (2, 3, or 4)
        current_player: Player whose turn it is (1-indexed)
        current_phase: Current game phase
        rings_in_hand: Rings in hand per player

    Returns:
        A fresh GameState ready for testing
    """
    size = get_board_size(board_type)
    board = BoardState(type=board_type, size=size)
    players = create_players(num_players, rings_per_player=rings_in_hand)
    now = datetime.now()

    return GameState(
        id="test",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=current_phase,
        currentPlayer=current_player,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def game_state_factory() -> Callable[..., GameState]:
    """
    Factory fixture for creating GameState objects.

    Usage:
        def test_something(game_state_factory):
            state = game_state_factory(board_type=BoardType.HEX8, num_players=4)
    """
    return create_game_state


@pytest.fixture
def square8_state() -> GameState:
    """Pre-configured Square8 2-player state."""
    return create_game_state(BoardType.SQUARE8, num_players=2)


@pytest.fixture
def hex8_state() -> GameState:
    """Pre-configured Hex8 2-player state."""
    return create_game_state(BoardType.HEX8, num_players=2)


@pytest.fixture
def four_player_state() -> GameState:
    """Pre-configured Square8 4-player state."""
    return create_game_state(BoardType.SQUARE8, num_players=4)


# =============================================================================
# Specialized State Factories for Specific Test Scenarios
# =============================================================================

def create_capture_scenario_state(
    board_type: BoardType = BoardType.SQUARE8,
    attacker_pos: Position | None = None,
    attacker_rings: list[int] | None = None,
    target_pos: Position | None = None,
    target_rings: list[int] | None = None,
) -> GameState:
    """
    Create a GameState with an attacker and target set up for capture testing.

    The attacker is controlled by player 1, the target by player 2.
    """
    state = create_game_state(board_type, num_players=2, current_phase=GamePhase.MOVEMENT)

    # Default positions if not provided
    if board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
        attacker_pos = attacker_pos or Position(x=0, y=-2, z=2)
        target_pos = target_pos or Position(x=0, y=0, z=0)
    else:
        size = get_board_size(board_type)
        center = size // 2
        attacker_pos = attacker_pos or Position(x=center - 2, y=center)
        target_pos = target_pos or Position(x=center, y=center)

    attacker_rings = attacker_rings or [1, 1]  # P1 controls with cap_height=2
    target_rings = target_rings or [2]  # P2 controls with cap_height=1

    add_stack(state, attacker_pos, attacker_rings)
    add_stack(state, target_pos, target_rings)

    return state


def create_recovery_eligible_state(
    board_type: BoardType = BoardType.SQUARE8,
    player: int = 1,
    buried_count: int = 3,
) -> GameState:
    """
    Create a GameState where a player is eligible for recovery.

    The player will have the specified number of buried rings.
    """
    state = create_game_state(
        board_type,
        num_players=2,
        current_player=player,
        current_phase=GamePhase.RECOVERY,
    )

    positions = get_valid_positions(board_type, buried_count + 1)
    opponent = 2 if player == 1 else 1

    # Create stacks with buried rings (opponent controls with player's rings buried)
    for i in range(buried_count):
        if i < len(positions):
            # [player, opponent] means player's ring is buried under opponent's
            add_stack(state, positions[i], [player, opponent])

    # Give the player a marker for recovery slide
    if len(positions) > buried_count:
        add_marker(state, positions[buried_count], player)

    return state


def create_chain_capture_state(
    board_type: BoardType = BoardType.SQUARE8,
) -> GameState:
    """
    Create a GameState set up for chain capture testing.

    Sets up multiple targets in a line so chain capture is possible.
    """
    state = create_game_state(board_type, num_players=2, current_phase=GamePhase.MOVEMENT)

    if board_type in (BoardType.HEX8, BoardType.HEXAGONAL):
        # Line of targets along hex axis
        add_stack(state, Position(x=-3, y=0, z=3), [1, 1, 1])  # Strong attacker
        add_stack(state, Position(x=-1, y=0, z=1), [2])  # Target 1
        add_stack(state, Position(x=1, y=0, z=-1), [2])  # Target 2 (chain)
    else:
        size = get_board_size(board_type)
        center = size // 2
        # Line of targets along horizontal axis
        add_stack(state, Position(x=center - 3, y=center), [1, 1, 1])  # Strong attacker
        add_stack(state, Position(x=center - 1, y=center), [2])  # Target 1
        add_stack(state, Position(x=center + 1, y=center), [2])  # Target 2 (chain)

    return state


@pytest.fixture
def capture_state_factory() -> Callable[..., GameState]:
    """Factory fixture for capture scenario states."""
    return create_capture_scenario_state


@pytest.fixture
def recovery_state_factory() -> Callable[..., GameState]:
    """Factory fixture for recovery-eligible states."""
    return create_recovery_eligible_state


@pytest.fixture
def chain_capture_state_factory() -> Callable[..., GameState]:
    """Factory fixture for chain capture states."""
    return create_chain_capture_state
