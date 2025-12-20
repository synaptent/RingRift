"""Game-related test factories.

Provides factories for creating game states, boards, moves, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "create_board_config",
    "create_game_record",
    "create_game_state",
    "create_move",
]


@dataclass
class MockPosition:
    """Mock position for testing."""
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass
class MockMove:
    """Mock move for testing."""
    from_pos: MockPosition | None = None
    to_pos: MockPosition | None = None
    action_type: str = "move"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_pos.to_dict() if self.from_pos else None,
            "to": self.to_pos.to_dict() if self.to_pos else None,
            "type": self.action_type,
        }


@dataclass
class MockGameState:
    """Mock game state for testing."""
    board_type: str = "square8"
    num_players: int = 2
    current_player: int = 0
    turn_number: int = 1
    phase: str = "placement"
    game_over: bool = False
    winner: int | None = None
    board: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_type": self.board_type,
            "num_players": self.num_players,
            "current_player": self.current_player,
            "turn_number": self.turn_number,
            "phase": self.phase,
            "game_over": self.game_over,
            "winner": self.winner,
            "board": self.board,
        }


@dataclass
class MockBoardConfig:
    """Mock board configuration for testing."""
    board_type: str = "square8"
    num_players: int = 2
    board_size: int = 8

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class MockGameRecord:
    """Mock game record for testing."""
    game_id: str = "test_game_1"
    board_type: str = "square8"
    num_players: int = 2
    moves: list[MockMove] = field(default_factory=list)
    winner: int | None = None
    total_turns: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "moves": [m.to_dict() for m in self.moves],
            "winner": self.winner,
            "total_turns": self.total_turns,
            "duration_seconds": self.duration_seconds,
        }


def create_game_state(
    board_type: str = "square8",
    num_players: int = 2,
    current_player: int = 0,
    turn_number: int = 1,
    phase: str = "placement",
    game_over: bool = False,
    winner: int | None = None,
    **kwargs: Any,
) -> MockGameState:
    """Create a mock game state for testing.

    Args:
        board_type: Board type (default: square8)
        num_players: Number of players (default: 2)
        current_player: Current player index (default: 0)
        turn_number: Current turn number (default: 1)
        phase: Game phase (default: placement)
        game_over: Whether game is over (default: False)
        winner: Winner player index (default: None)
        **kwargs: Additional state attributes

    Returns:
        MockGameState instance
    """
    return MockGameState(
        board_type=board_type,
        num_players=num_players,
        current_player=current_player,
        turn_number=turn_number,
        phase=phase,
        game_over=game_over,
        winner=winner,
        **kwargs,
    )


def create_board_config(
    board_type: str = "square8",
    num_players: int = 2,
    board_size: int = 8,
) -> MockBoardConfig:
    """Create a mock board configuration for testing.

    Args:
        board_type: Board type (default: square8)
        num_players: Number of players (default: 2)
        board_size: Board size (default: 8)

    Returns:
        MockBoardConfig instance
    """
    return MockBoardConfig(
        board_type=board_type,
        num_players=num_players,
        board_size=board_size,
    )


def create_move(
    from_x: int | None = None,
    from_y: int | None = None,
    to_x: int = 0,
    to_y: int = 0,
    action_type: str = "move",
) -> MockMove:
    """Create a mock move for testing.

    Args:
        from_x: Source x coordinate
        from_y: Source y coordinate
        to_x: Destination x coordinate
        to_y: Destination y coordinate
        action_type: Move type (default: move)

    Returns:
        MockMove instance
    """
    from_pos = MockPosition(from_x, from_y) if from_x is not None and from_y is not None else None
    to_pos = MockPosition(to_x, to_y)

    return MockMove(
        from_pos=from_pos,
        to_pos=to_pos,
        action_type=action_type,
    )


def create_game_record(
    game_id: str = "test_game_1",
    board_type: str = "square8",
    num_players: int = 2,
    num_moves: int = 10,
    winner: int | None = 0,
    duration_seconds: float = 60.0,
) -> MockGameRecord:
    """Create a mock game record for testing.

    Args:
        game_id: Unique game ID
        board_type: Board type
        num_players: Number of players
        num_moves: Number of moves to generate
        winner: Winner player index
        duration_seconds: Game duration

    Returns:
        MockGameRecord instance
    """
    moves = [create_move(to_x=i, to_y=i) for i in range(num_moves)]

    return MockGameRecord(
        game_id=game_id,
        board_type=board_type,
        num_players=num_players,
        moves=moves,
        winner=winner,
        total_turns=num_moves,
        duration_seconds=duration_seconds,
    )
