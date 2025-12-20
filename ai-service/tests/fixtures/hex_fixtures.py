"""Hex Board Test Fixtures.

Per Section 3.3.1 of the action plan, this module provides:
- Reusable hex game state fixtures
- D6 symmetry verification helpers
- Hex coordinate utilities
- Standard test positions

Usage:
    from tests.fixtures.hex_fixtures import (
        create_hex_game_state,
        create_hex_board_with_stacks,
        HexCoord,
        D6_ROTATIONS,
        verify_d6_symmetry,
    )

    # Create a basic hex game state
    state = create_hex_game_state(size=11, num_players=2)

    # Create a state with specific stacks
    state = create_hex_board_with_stacks({
        HexCoord(0, 0): [1, 1],
        HexCoord(1, 0): [2],
    })

    # Verify D6 symmetry of a function
    verify_d6_symmetry(evaluate_fn, state)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.rules.core import BOARD_CONFIGS, get_rings_per_player

# =============================================================================
# Hex Coordinate System
# =============================================================================

@dataclass(frozen=True)
class HexCoord:
    """Axial hex coordinate (q, r).

    Uses the axial coordinate system where:
    - q: column (east-west axis)
    - r: row (northwest-southeast axis)
    - s = -q - r (implicit, northeast-southwest axis)

    This is the standard coordinate system for hex grids.
    """
    q: int
    r: int

    @property
    def s(self) -> int:
        """Get the implicit s coordinate (q + r + s = 0)."""
        return -self.q - self.r

    def to_cube(self) -> tuple[int, int, int]:
        """Convert to cube coordinates (x, y, z)."""
        return (self.q, self.r, self.s)

    @classmethod
    def from_cube(cls, x: int, y: int, z: int) -> HexCoord:
        """Create from cube coordinates."""
        return cls(q=x, r=y)

    def to_offset(self, size: int) -> tuple[int, int]:
        """Convert to offset (x, y) coordinates for board storage.

        Args:
            size: Board size (radius)

        Returns:
            (x, y) tuple for board indexing
        """
        # Offset coordinates for odd-r layout
        x = self.q + (self.r + 1) // 2
        y = self.r + size
        return (x, y)

    @classmethod
    def from_offset(cls, x: int, y: int, size: int) -> HexCoord:
        """Create from offset coordinates.

        Args:
            x, y: Offset coordinates
            size: Board size (radius)
        """
        r = y - size
        q = x - (r + 1) // 2
        return cls(q=q, r=r)

    def distance(self, other: HexCoord) -> int:
        """Manhattan distance in hex grid."""
        return (abs(self.q - other.q) +
                abs(self.q + self.r - other.q - other.r) +
                abs(self.r - other.r)) // 2

    def neighbors(self) -> list[HexCoord]:
        """Get the 6 neighboring hex coordinates."""
        return [
            HexCoord(self.q + 1, self.r),
            HexCoord(self.q + 1, self.r - 1),
            HexCoord(self.q, self.r - 1),
            HexCoord(self.q - 1, self.r),
            HexCoord(self.q - 1, self.r + 1),
            HexCoord(self.q, self.r + 1),
        ]

    def rotate_60(self) -> HexCoord:
        """Rotate 60 degrees clockwise around origin."""
        # In cube coords: (x,y,z) -> (-z,-x,-y)
        x, y, z = self.to_cube()
        return HexCoord.from_cube(-z, -x, -y)

    def rotate_120(self) -> HexCoord:
        """Rotate 120 degrees clockwise around origin."""
        return self.rotate_60().rotate_60()

    def rotate_180(self) -> HexCoord:
        """Rotate 180 degrees around origin."""
        return HexCoord(-self.q, -self.r)

    def rotate_240(self) -> HexCoord:
        """Rotate 240 degrees clockwise around origin."""
        return self.rotate_180().rotate_60()

    def rotate_300(self) -> HexCoord:
        """Rotate 300 degrees clockwise around origin."""
        return self.rotate_180().rotate_120()

    def reflect_q(self) -> HexCoord:
        """Reflect across the q axis."""
        x, y, z = self.to_cube()
        return HexCoord.from_cube(x, z, y)

    def __str__(self) -> str:
        return f"({self.q},{self.r})"


# =============================================================================
# D6 Symmetry Operations
# =============================================================================

# D6 symmetry group: 6 rotations + 6 reflections
D6_ROTATIONS = [
    ("identity", lambda c: c),
    ("rotate_60", lambda c: c.rotate_60()),
    ("rotate_120", lambda c: c.rotate_120()),
    ("rotate_180", lambda c: c.rotate_180()),
    ("rotate_240", lambda c: c.rotate_240()),
    ("rotate_300", lambda c: c.rotate_300()),
]

D6_REFLECTIONS = [
    ("reflect_q", lambda c: c.reflect_q()),
    ("reflect_q_60", lambda c: c.rotate_60().reflect_q()),
    ("reflect_q_120", lambda c: c.rotate_120().reflect_q()),
    ("reflect_q_180", lambda c: c.rotate_180().reflect_q()),
    ("reflect_q_240", lambda c: c.rotate_240().reflect_q()),
    ("reflect_q_300", lambda c: c.rotate_300().reflect_q()),
]

D6_SYMMETRIES = D6_ROTATIONS + D6_REFLECTIONS


def apply_symmetry_to_board(
    stacks: dict[str, RingStack],
    symmetry_fn: Callable[[HexCoord], HexCoord],
    size: int,
) -> dict[str, RingStack]:
    """Apply a D6 symmetry operation to board stacks.

    Args:
        stacks: Dict mapping position keys to RingStack
        symmetry_fn: Symmetry transformation function
        size: Board size (radius)

    Returns:
        New stacks dict with transformed positions
    """
    new_stacks = {}
    for key, stack in stacks.items():
        # Parse position key "x,y"
        x, y = map(int, key.split(','))
        coord = HexCoord.from_offset(x, y, size)

        # Apply symmetry
        new_coord = symmetry_fn(coord)
        new_x, new_y = new_coord.to_offset(size)
        new_key = f"{new_x},{new_y}"

        new_stacks[new_key] = stack

    return new_stacks


def verify_d6_symmetry(
    evaluate_fn: Callable[[GameState], float],
    state: GameState,
    tolerance: float = 1e-6,
) -> dict[str, bool]:
    """Verify that an evaluation function respects D6 symmetry.

    For a function to respect D6 symmetry, it should return the same
    value for all symmetric positions.

    Args:
        evaluate_fn: Function that evaluates a game state
        state: Base game state to test
        tolerance: Tolerance for float comparison

    Returns:
        Dict mapping symmetry name to whether it passed
    """
    base_value = evaluate_fn(state)
    results = {}

    for name, sym_fn in D6_SYMMETRIES:
        if name == "identity":
            results[name] = True
            continue

        # Create symmetric state
        sym_state = apply_symmetry_to_state(state, sym_fn)
        sym_value = evaluate_fn(sym_state)

        results[name] = abs(sym_value - base_value) < tolerance

    return results


def apply_symmetry_to_state(
    state: GameState,
    symmetry_fn: Callable[[HexCoord], HexCoord],
) -> GameState:
    """Apply a D6 symmetry to a game state.

    Args:
        state: Original game state
        symmetry_fn: Symmetry transformation function

    Returns:
        New game state with transformed positions
    """
    size = state.board.size

    # Transform stacks
    new_stacks = apply_symmetry_to_board(state.board.stacks, symmetry_fn, size)

    # Transform markers
    new_markers = {}
    for key, marker in state.board.markers.items():
        x, y = map(int, key.split(','))
        coord = HexCoord.from_offset(x, y, size)
        new_coord = symmetry_fn(coord)
        new_x, new_y = new_coord.to_offset(size)
        new_key = f"{new_x},{new_y}"
        new_markers[new_key] = marker

    new_board = BoardState(
        type=state.board.type,
        size=state.board.size,
        stacks=new_stacks,
        markers=new_markers,
        collapsedSpaces=state.board.collapsedSpaces.copy(),
        eliminatedRings=state.board.eliminatedRings.copy(),
    )

    return state.model_copy(update={'board': new_board})


# =============================================================================
# Game State Factory Functions
# =============================================================================

def create_hex_game_state(
    size: int = 11,
    num_players: int = 2,
    current_player: int = 1,
    phase: GamePhase = GamePhase.RING_PLACEMENT,
    board_type: BoardType = BoardType.HEXAGONAL,
) -> GameState:
    """Create a basic hex game state for testing.

    Args:
        size: Board size (11 for standard hex, 8 for hex8)
        num_players: Number of players (2-4)
        current_player: Current player number (1-based)
        phase: Game phase
        board_type: Board type (HEXAGONAL or HEX8)

    Returns:
        Empty hex game state
    """
    rings_per_player = get_rings_per_player(board_type)

    players = []
    for i in range(num_players):
        players.append(Player(
            id=f"p{i+1}",
            username=f"Player {i+1}",
            type="human",
            playerNumber=i + 1,
            isReady=True,
            timeRemaining=600,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=10,
        ))

    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    total_rings = rings_per_player * num_players

    return GameState(
        id="test-hex",
        boardType=board_type,
        rngSeed=42,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=current_player,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        winner=None,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=total_rings,
        totalRingsEliminated=0,
        victoryThreshold=rings_per_player,
        territoryVictoryThreshold=33,
    )


def create_hex_board_with_stacks(
    stacks: dict[HexCoord, list[int]],
    size: int = 11,
    num_players: int = 2,
    board_type: BoardType = BoardType.HEXAGONAL,
) -> GameState:
    """Create a hex game state with specific stack positions.

    Args:
        stacks: Dict mapping HexCoord to list of player numbers (stack)
        size: Board size
        num_players: Number of players
        board_type: Board type

    Returns:
        Game state with specified stacks
    """
    state = create_hex_game_state(
        size=size,
        num_players=num_players,
        phase=GamePhase.MOVEMENT,
        board_type=board_type,
    )

    # Convert HexCoord stacks to board format
    board_stacks = {}
    rings_used = dict.fromkeys(range(1, num_players + 1), 0)

    for coord, stack_list in stacks.items():
        x, y = coord.to_offset(size)
        key = f"{x},{y}"
        controller = stack_list[0] if stack_list else 0  # Bottom ring controls
        # Calculate cap height (consecutive rings at top from same player)
        cap_height = 0
        for ring in stack_list:
            if ring == controller:
                cap_height += 1
            else:
                break
        board_stacks[key] = RingStack(
            position=Position(x=x, y=y),
            rings=stack_list,
            stackHeight=len(stack_list),
            capHeight=cap_height,
            controllingPlayer=controller,
        )
        for player in stack_list:
            rings_used[player] += 1

    # Update player rings in hand
    new_players = []
    rings_per_player = get_rings_per_player(board_type)
    for player in state.players:
        new_players.append(player.model_copy(update={
            'rings_in_hand': rings_per_player - rings_used.get(player.player_number, 0)
        }))

    new_board = state.board.model_copy(update={'stacks': board_stacks})

    return state.model_copy(update={
        'board': new_board,
        'players': new_players,
    })


def create_hex_training_sample(
    features_shape: tuple[int, ...] = (10, 21, 21),
    policy_size: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create a synthetic hex training sample for testing.

    Args:
        features_shape: Shape of feature tensor (channels, height, width)
        policy_size: Size of policy vector
        seed: Random seed for reproducibility

    Returns:
        Dict with 'features', 'values', 'policy_values' arrays
    """
    np.random.seed(seed)

    return {
        'features': np.random.randn(*features_shape).astype(np.float32),
        'values': np.array([np.tanh(np.random.randn())]).astype(np.float32),
        'policy_values': np.random.dirichlet(np.ones(policy_size)).astype(np.float32),
    }


# =============================================================================
# Standard Test Positions
# =============================================================================

def hex_center_position(size: int = 11) -> HexCoord:
    """Get the center coordinate of a hex board."""
    return HexCoord(0, 0)


def hex_corner_positions(size: int = 11) -> list[HexCoord]:
    """Get the 6 corner coordinates of a hex board."""
    # For a hex board of radius r, corners are at distance r from center
    r = size - 1
    return [
        HexCoord(r, 0),
        HexCoord(0, r),
        HexCoord(-r, r),
        HexCoord(-r, 0),
        HexCoord(0, -r),
        HexCoord(r, -r),
    ]


def hex_edge_midpoints(size: int = 11) -> list[HexCoord]:
    """Get the midpoint of each edge of the hex board.

    For a hex board of radius r, returns 6 edge midpoints
    (between consecutive corners).
    """
    r = size - 1
    mid = r // 2
    # Edge midpoints between consecutive corners, going clockwise from (r, 0)
    # Edge 1: (r, 0) to (0, r) - midpoint at (r-mid, mid)
    # Edge 2: (0, r) to (-r, r) - midpoint at (-mid, r)
    # Edge 3: (-r, r) to (-r, 0) - midpoint at (-r, mid)
    # Edge 4: (-r, 0) to (0, -r) - midpoint at (-mid, -mid)
    # Edge 5: (0, -r) to (r, -r) - midpoint at (mid, -r)
    # Edge 6: (r, -r) to (r, 0) - midpoint at (r, -mid)
    return [
        HexCoord(r - mid, mid),     # Edge between (r,0) and (0,r)
        HexCoord(-mid, r),          # Edge between (0,r) and (-r,r)
        HexCoord(-r, mid),          # Edge between (-r,r) and (-r,0)
        HexCoord(-r + mid, -mid),   # Edge between (-r,0) and (0,-r)
        HexCoord(mid, -r),          # Edge between (0,-r) and (r,-r)
        HexCoord(r, -mid),          # Edge between (r,-r) and (r,0)
    ]


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def empty_hex_state() -> GameState:
    """Empty hexagonal board game state."""
    return create_hex_game_state()


@pytest.fixture
def empty_hex8_state() -> GameState:
    """Empty hex8 board game state."""
    return create_hex_game_state(size=8, board_type=BoardType.HEXAGONAL)


@pytest.fixture
def hex_state_with_center_stack() -> GameState:
    """Hex board with a single stack at center."""
    return create_hex_board_with_stacks({
        HexCoord(0, 0): [1],
    })


@pytest.fixture
def hex_state_symmetric() -> GameState:
    """Hex board with D6-symmetric stack placement."""
    # Place stacks at all 6 corners - perfectly symmetric
    corners = hex_corner_positions(11)
    stacks = {corner: [1] for corner in corners[:3]}
    stacks.update({corner: [2] for corner in corners[3:]})
    return create_hex_board_with_stacks(stacks)


@pytest.fixture
def hex_training_sample() -> dict[str, np.ndarray]:
    """Synthetic hex training sample."""
    return create_hex_training_sample()


@pytest.fixture
def hex_coord_origin() -> HexCoord:
    """Origin hex coordinate."""
    return HexCoord(0, 0)


@pytest.fixture
def hex_coords_ring_1() -> list[HexCoord]:
    """All hex coordinates at distance 1 from origin."""
    return HexCoord(0, 0).neighbors()
