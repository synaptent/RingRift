"""
Shared pytest fixtures for ai-service tests.

This module provides common fixtures to reduce duplication across test files.
All fixtures are session-scoped by default for better performance where appropriate,
but game state fixtures are function-scoped to ensure test isolation.
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional

import pytest


# =============================================================================
# PROMETHEUS REGISTRY FIX
# =============================================================================
# Fix for "Duplicated timeseries in CollectorRegistry" errors during test collection.
# This happens when multiple test files import modules that register Prometheus metrics.
# We patch the registry to silently handle re-registration of identical metrics.


def _patch_prometheus_registry():
    """Patch Prometheus registry to handle duplicate metric registration gracefully.

    This is needed because:
    1. app/metrics.py and app/metrics/orchestrator.py register metrics at import time
    2. Different test files may import these via different paths
    3. Python's import system can re-execute module-level code in edge cases

    The patch makes re-registration of identical metrics a no-op instead of an error.
    """
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.registry import CollectorRegistry

        _original_register = CollectorRegistry.register

        def _safe_register(self, collector):
            """Register collector, ignoring duplicates."""
            try:
                return _original_register(self, collector)
            except ValueError as e:
                if "Duplicated timeseries" in str(e):
                    # Already registered, ignore
                    pass
                else:
                    raise

        # Only patch once
        if not getattr(CollectorRegistry, '_patched_for_tests', False):
            CollectorRegistry.register = _safe_register
            CollectorRegistry._patched_for_tests = True

    except ImportError:
        # prometheus_client not installed, no patching needed
        pass


# Apply patch immediately at conftest load time (before test collection)
_patch_prometheus_registry()

# Ensure ai-service root is on sys.path so `import app` works when running
# pytest either from the repository root or from the ai-service directory.
# This avoids ModuleNotFoundError in tests/conftest.py when the pytest
# rootdir is resolved above ai-service/.
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import (
    AIConfig,
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


# =============================================================================
# FACTORY FIXTURES
# =============================================================================


@pytest.fixture
def player_factory() -> Callable[..., Player]:
    """Factory for creating Player instances with customizable defaults."""

    def _create_player(
        player_number: int = 1,
        username: Optional[str] = None,
        player_type: str = "human",
        rings_in_hand: int = 10,
        eliminated_rings: int = 0,
        territory_spaces: int = 0,
        time_remaining: int = 600,
        ai_difficulty: Optional[int] = None,
    ) -> Player:
        return Player(
            id=f"p{player_number}",
            username=username or f"Player{player_number}",
            type=player_type,
            playerNumber=player_number,
            isReady=True,
            timeRemaining=time_remaining,
            ringsInHand=rings_in_hand,
            eliminatedRings=eliminated_rings,
            territorySpaces=territory_spaces,
            aiDifficulty=ai_difficulty,
        )

    return _create_player


@pytest.fixture
def move_factory() -> Callable[..., Move]:
    """Factory for creating Move instances with customizable defaults."""

    def _create_move(
        player: int = 1,
        x: int = 0,
        y: int = 0,
        move_type: MoveType = MoveType.PLACE_RING,
        move_number: int = 1,
        from_pos: Optional[Position] = None,
        count: int = 1,
    ) -> Move:
        return Move(
            id=f"m-{player}-{x}-{y}-{move_number}",
            type=move_type,
            player=player,
            to=Position(x=x, y=y),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=move_number,
            **{"from": from_pos} if from_pos else {},
            count=count if move_type == MoveType.PLACE_RING else None,
        )

    return _create_move


@pytest.fixture
def ring_stack_factory() -> Callable[..., RingStack]:
    """Factory for creating RingStack instances."""

    def _create_stack(
        x: int,
        y: int,
        rings: List[int],
        controlling_player: Optional[int] = None,
    ) -> RingStack:
        if not rings:
            raise ValueError("Stack must have at least one ring")
        controller = controlling_player if controlling_player is not None else rings[0]
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

    return _create_stack


@pytest.fixture
def board_state_factory(ring_stack_factory) -> Callable[..., BoardState]:
    """Factory for creating BoardState instances."""

    def _create_board(
        board_type: BoardType = BoardType.SQUARE8,
        stacks: Optional[Dict[str, RingStack]] = None,
        markers: Optional[Dict[str, MarkerInfo]] = None,
        collapsed_spaces: Optional[Dict[str, int]] = None,
        eliminated_rings: Optional[Dict[int, int]] = None,
    ) -> BoardState:
        size = 8 if board_type == BoardType.SQUARE8 else 19 if board_type == BoardType.SQUARE19 else 5
        return BoardState(
            type=board_type,
            size=size,
            stacks=stacks or {},
            markers=markers or {},
            collapsedSpaces=collapsed_spaces or {},
            eliminatedRings=eliminated_rings or {},
        )

    return _create_board


@pytest.fixture
def game_state_factory(player_factory, board_state_factory) -> Callable[..., GameState]:
    """Factory for creating GameState instances with full customization."""

    def _create_game_state(
        game_id: str = "test-game",
        board_type: BoardType = BoardType.SQUARE8,
        num_players: int = 2,
        current_player: int = 1,
        current_phase: GamePhase = GamePhase.RING_PLACEMENT,
        game_status: GameStatus = GameStatus.ACTIVE,
        players: Optional[List[Player]] = None,
        board: Optional[BoardState] = None,
        move_history: Optional[List[Move]] = None,
        rings_in_hand: int = 10,
        total_rings_in_play: int = 36,
        rng_seed: Optional[int] = None,
    ) -> GameState:
        if players is None:
            players = [player_factory(i, rings_in_hand=rings_in_hand) for i in range(1, num_players + 1)]

        if board is None:
            board = board_state_factory(board_type)

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

    return _create_game_state


# =============================================================================
# COMMON GAME STATE FIXTURES
# =============================================================================


@pytest.fixture
def empty_game_state(game_state_factory) -> GameState:
    """A fresh game state with empty board, ready for ring placement."""
    return game_state_factory()


@pytest.fixture
def game_state_with_stacks(game_state_factory, ring_stack_factory) -> GameState:
    """A game state with some stacks on the board in movement phase."""
    stacks = {
        "3,3": ring_stack_factory(3, 3, [1]),
        "4,4": ring_stack_factory(4, 4, [2]),
        "5,5": ring_stack_factory(5, 5, [1, 1]),
    }
    return game_state_factory(
        current_phase=GamePhase.MOVEMENT,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks=stacks,
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        ),
    )


@pytest.fixture
def game_state_mid_game(game_state_factory, ring_stack_factory) -> GameState:
    """A mid-game state with multiple stacks and markers."""
    stacks = {
        "2,2": ring_stack_factory(2, 2, [1, 1]),
        "3,3": ring_stack_factory(3, 3, [2, 2, 1]),
        "5,5": ring_stack_factory(5, 5, [1]),
        "6,6": ring_stack_factory(6, 6, [2]),
    }
    markers = {
        "4,4": MarkerInfo(player=1, position=Position(x=4, y=4), type="regular"),
    }
    return game_state_factory(
        current_phase=GamePhase.MOVEMENT,
        board=BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks=stacks,
            markers=markers,
            collapsedSpaces={},
            eliminatedRings={},
        ),
        rings_in_hand=7,
    )


# =============================================================================
# AI CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def ai_config_easy() -> AIConfig:
    """AI configuration for easy difficulty."""
    return AIConfig(difficulty=1)


@pytest.fixture
def ai_config_medium() -> AIConfig:
    """AI configuration for medium difficulty."""
    return AIConfig(difficulty=5)


@pytest.fixture
def ai_config_hard() -> AIConfig:
    """AI configuration for hard difficulty."""
    return AIConfig(difficulty=9)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear AI move caches before each test to ensure isolation."""
    from app.ai.move_cache import clear_move_cache
    clear_move_cache()
    yield
    clear_move_cache()


@pytest.fixture
def position_factory() -> Callable[..., Position]:
    """Factory for creating Position instances."""

    def _create_position(x: int, y: int, z: Optional[int] = None) -> Position:
        if z is not None:
            return Position(x=x, y=y, z=z)
        return Position(x=x, y=y)

    return _create_position


# =============================================================================
# HEX BOARD FIXTURES
# =============================================================================


@pytest.fixture
def hex_game_state(game_state_factory) -> GameState:
    """A game state for hexagonal board."""
    return game_state_factory(
        board_type=BoardType.HEXAGONAL,
        total_rings_in_play=42,
    )


# =============================================================================
# MULTIPLAYER FIXTURES
# =============================================================================


@pytest.fixture
def three_player_game_state(game_state_factory) -> GameState:
    """A 3-player game state."""
    return game_state_factory(num_players=3)


@pytest.fixture
def four_player_game_state(game_state_factory) -> GameState:
    """A 4-player game state."""
    return game_state_factory(num_players=4)
