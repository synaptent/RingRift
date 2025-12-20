"""Pytest fixtures for GPU parity tests.

Provides common fixtures for comparing GPU and CPU implementations.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from app.models import (
    GameState,
    BoardType,
    BoardState,
    GamePhase,
    GameStatus,
    Player,
    Position,
    TimeControl,
)
from app.game_engine import GameEngine
from app.rules.core import (
    BOARD_CONFIGS,
    get_effective_line_length,
    get_victory_threshold,
    get_territory_victory_threshold,
)


# =============================================================================
# Device Configuration
# =============================================================================


@pytest.fixture(scope="session")
def device():
    """Get the appropriate torch device for GPU tests."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


# =============================================================================
# Game Engine Fixtures
# =============================================================================


@pytest.fixture
def cpu_engine():
    """Create a CPU game engine instance."""
    return GameEngine()


# =============================================================================
# Game State Fixtures
# =============================================================================


def create_test_players(num_players: int, rings_per_player: int) -> list[Player]:
    """Create test players with specified ring count."""
    return [
        Player(
            id=f"p{i}",
            username=f"Player{i}",
            type="ai",
            playerNumber=i,
            isReady=True,
            timeRemaining=600,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=10,
        )
        for i in range(1, num_players + 1)
    ]


def create_empty_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> GameState:
    """Create an empty game state for testing."""
    from datetime import datetime

    config = BOARD_CONFIGS[board_type]
    players = create_test_players(num_players, config.rings_per_player)

    now = datetime.now()
    board = BoardState(
        type=board_type,
        size=config.size,
        stacks={},
        markers={},
        collapsedSpaces={},
    )

    return GameState(
        id="test-game",
        boardType=board_type,
        board=board,
        players=players,
        gameStatus=GameStatus.ACTIVE,
        currentPlayer=1,
        currentPhase=GamePhase.RING_PLACEMENT,
        moveHistory=[],
        timeControl=TimeControl(
            type="untimed",
            initialTime=0,
            increment=0,
        ),
        spectators=[],
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=config.rings_per_player * num_players,
        totalRingsEliminated=0,
        victoryThreshold=get_victory_threshold(board_type, num_players),
        territoryVictoryThreshold=get_territory_victory_threshold(board_type),
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsCurrentRoundActorMask={},
        lpsExclusivePlayerForCompletedRound=None,
        lpsCurrentRoundFirstPlayer=None,
        lpsConsecutiveExclusiveRounds=0,
        lpsConsecutiveExclusivePlayer=None,
    )


@pytest.fixture
def empty_square8_2p():
    """Empty 8x8 board with 2 players."""
    return create_empty_game_state(BoardType.SQUARE8, 2)


@pytest.fixture
def empty_square8_3p():
    """Empty 8x8 board with 3 players."""
    return create_empty_game_state(BoardType.SQUARE8, 3)


@pytest.fixture
def empty_square8_4p():
    """Empty 8x8 board with 4 players."""
    return create_empty_game_state(BoardType.SQUARE8, 4)


@pytest.fixture
def empty_square19_2p():
    """Empty 19x19 board with 2 players."""
    return create_empty_game_state(BoardType.SQUARE19, 2)


@pytest.fixture
def empty_hexagonal_2p():
    """Empty hexagonal board with 2 players."""
    return create_empty_game_state(BoardType.HEXAGONAL, 2)


# =============================================================================
# Parameterized Fixtures
# =============================================================================


@pytest.fixture(params=[
    (BoardType.SQUARE8, 2),
    (BoardType.SQUARE8, 3),
    (BoardType.SQUARE8, 4),
    (BoardType.SQUARE19, 2),
])
def board_player_combo(request):
    """Parameterized fixture for board type and player count combinations."""
    board_type, num_players = request.param
    return {
        "board_type": board_type,
        "num_players": num_players,
        "game_state": create_empty_game_state(board_type, num_players),
        "line_length": get_effective_line_length(board_type, num_players),
        "victory_threshold": get_victory_threshold(board_type, num_players),
    }


# =============================================================================
# Position Fixtures for Specific Scenarios
# =============================================================================


def add_stack_to_state(
    state: GameState,
    x: int,
    y: int,
    rings: list[int],
) -> GameState:
    """Add a stack to the game state.

    Args:
        state: Game state to modify
        x, y: Position coordinates
        rings: List of player numbers from bottom to top
    """
    from app.models import RingStack

    key = f"{x},{y}"
    controlling_player = rings[-1] if rings else 0

    # Calculate cap height
    cap_height = 0
    for ring in reversed(rings):
        if ring == controlling_player:
            cap_height += 1
        else:
            break

    state.board.stacks[key] = RingStack(
        position=Position(x=x, y=y),
        rings=rings,
        stackHeight=len(rings),
        controllingPlayer=controlling_player,
        capHeight=cap_height,
    )
    return state


def add_marker_to_state(
    state: GameState,
    x: int,
    y: int,
    player: int,
) -> GameState:
    """Add a marker to the game state."""
    key = f"{x},{y}"
    state.board.markers[key] = player
    return state


def add_collapsed_space(
    state: GameState,
    x: int,
    y: int,
    player: int,
) -> GameState:
    """Add a collapsed (territory) space to the game state."""
    key = f"{x},{y}"
    state.board.collapsedSpaces[key] = player
    return state


@pytest.fixture
def state_with_single_stack(empty_square8_2p):
    """State with a single stack at center."""
    state = empty_square8_2p
    add_stack_to_state(state, 3, 3, [1])
    state.players[0] = state.players[0].model_copy(
        update={'rings_in_hand': state.players[0].rings_in_hand - 1}
    )
    return state


@pytest.fixture
def state_with_capture_opportunity(empty_square8_2p):
    """State where player 1 can capture player 2's stack."""
    state = empty_square8_2p
    # Player 1 stack at (2, 3)
    add_stack_to_state(state, 2, 3, [1])
    state.players[0] = state.players[0].model_copy(
        update={'rings_in_hand': state.players[0].rings_in_hand - 1}
    )
    # Player 2 stack at (3, 3)
    add_stack_to_state(state, 3, 3, [2])
    state.players[1] = state.players[1].model_copy(
        update={'rings_in_hand': state.players[1].rings_in_hand - 1}
    )
    # Empty space at (4, 3) for landing
    # Note: GameState is immutable (Pydantic frozen), phase doesn't affect line detection test
    return state


@pytest.fixture
def state_with_line_opportunity(empty_square8_2p):
    """State where player 1 has markers that could form a line."""
    state = empty_square8_2p
    # 3 markers in a row (need 4 for 2-player 8x8)
    add_marker_to_state(state, 2, 3, 1)
    add_marker_to_state(state, 3, 3, 1)
    add_marker_to_state(state, 4, 3, 1)
    # Stack that could extend the line
    add_stack_to_state(state, 1, 3, [1])
    state.players[0] = state.players[0].model_copy(
        update={'rings_in_hand': state.players[0].rings_in_hand - 1}
    )
    # Note: GameState is immutable (Pydantic frozen), phase doesn't affect line detection test
    return state


@pytest.fixture
def state_3p_with_line_opportunity(empty_square8_3p):
    """State where player 1 has markers that could form a line (3-player, needs 3)."""
    state = empty_square8_3p
    # 2 markers in a row (need 3 for 3-player 8x8)
    add_marker_to_state(state, 2, 3, 1)
    add_marker_to_state(state, 3, 3, 1)
    # Stack that could extend the line
    add_stack_to_state(state, 1, 3, [1])
    state.players[0] = state.players[0].model_copy(
        update={'rings_in_hand': state.players[0].rings_in_hand - 1}
    )
    # Note: GameState is immutable (Pydantic frozen), phase doesn't affect line detection test
    return state


# =============================================================================
# Comparison Utilities
# =============================================================================


def moves_to_comparable_set(moves: list[Any]) -> set:
    """Convert a list of moves to a comparable set of tuples."""
    result = set()
    for move in moves:
        if hasattr(move, "type"):
            # CPU move format
            if hasattr(move, "from_pos") and move.from_pos:
                from_tuple = (move.from_pos.x, move.from_pos.y)
            else:
                from_tuple = None

            if hasattr(move, "to_pos") and move.to_pos:
                to_tuple = (move.to_pos.x, move.to_pos.y)
            else:
                to_tuple = None

            result.add((move.type, from_tuple, to_tuple))
        else:
            # Assume tuple format
            result.add(tuple(move))
    return result


def assert_moves_equal(cpu_moves: list[Any], gpu_moves: list[Any], context: str = ""):
    """Assert that two move lists contain the same moves."""
    cpu_set = moves_to_comparable_set(cpu_moves)
    gpu_set = moves_to_comparable_set(gpu_moves)

    missing_in_gpu = cpu_set - gpu_set
    extra_in_gpu = gpu_set - cpu_set

    if missing_in_gpu or extra_in_gpu:
        msg = f"Move mismatch{' (' + context + ')' if context else ''}\n"
        msg += f"CPU moves ({len(cpu_set)}): {sorted(cpu_set)[:10]}...\n"
        msg += f"GPU moves ({len(gpu_set)}): {sorted(gpu_set)[:10]}...\n"
        if missing_in_gpu:
            msg += f"Missing in GPU ({len(missing_in_gpu)}): {sorted(missing_in_gpu)[:5]}\n"
        if extra_in_gpu:
            msg += f"Extra in GPU ({len(extra_in_gpu)}): {sorted(extra_in_gpu)[:5]}\n"
        pytest.fail(msg)


def assert_scores_close(
    cpu_score: float,
    gpu_score: float,
    tolerance: float = 0.01,
    context: str = "",
):
    """Assert that two scores are within tolerance."""
    diff = abs(cpu_score - gpu_score)
    if diff > tolerance:
        msg = f"Score mismatch{' (' + context + ')' if context else ''}: "
        msg += f"CPU={cpu_score:.6f}, GPU={gpu_score:.6f}, diff={diff:.6f}"
        pytest.fail(msg)
