from __future__ import annotations

from datetime import datetime
import os
import sys

# Ensure `app.*` imports resolve when running pytest from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # type: ignore  # noqa: E402
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    TimeControl,
)
from app.rules.phase_machine import PhaseTransitionInput, advance_phases  # noqa: E402


def _make_minimal_state(
    phase: GamePhase,
    current_player: int = 1,
) -> GameState:
    """Create a minimal active GameState for phase-machine tests."""
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsed_spaces={},
        territories={},
        formed_lines=[],
        eliminated_rings={"1": 0, "2": 0},
    )
    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=0,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=0,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]
    now = datetime.now()

    return GameState(
        id="phase-machine-test",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=current_player,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def _make_noop_move(move_type: MoveType, player: int = 1) -> Move:
    """Create a minimal bookkeeping-style Move for phase tests."""
    now = datetime.now()
    return Move(
        id="m1",
        type=move_type,
        player=player,
        to=Position(x=0, y=0),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )


def test_no_movement_action_advances_to_line_processing():
    """NO_MOVEMENT_ACTION should advance to LINE_PROCESSING."""
    state = _make_minimal_state(GamePhase.MOVEMENT, current_player=1)
    move = _make_noop_move(MoveType.NO_MOVEMENT_ACTION, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    assert state.current_phase == GamePhase.LINE_PROCESSING


def test_no_line_action_with_empty_board_rotates_to_next_player():
    """
    NO_LINE_ACTION on empty board should rotate to next player.

    With no territory regions and no stacks (empty board), the phase machine
    should skip TERRITORY_PROCESSING entirely and rotate to the next player
    in RING_PLACEMENT. This matches RR-CANON phase transition rules.
    """
    state = _make_minimal_state(GamePhase.LINE_PROCESSING, current_player=1)
    move = _make_noop_move(MoveType.NO_LINE_ACTION, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    # Empty board: no territory regions, no stacks → turn ends
    assert state.current_player == 2
    assert state.current_phase == GamePhase.RING_PLACEMENT


def test_no_territory_action_rotates_to_next_player_ring_placement():
    """NO_TERRITORY_ACTION should rotate to next player in RING_PLACEMENT."""
    state = _make_minimal_state(GamePhase.TERRITORY_PROCESSING, current_player=1)
    move = _make_noop_move(MoveType.NO_TERRITORY_ACTION, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    assert state.current_player == 2
    assert state.current_phase == GamePhase.RING_PLACEMENT


def test_process_line_with_empty_board_rotates_to_next_player():
    """
    PROCESS_LINE on empty board with no remaining lines should rotate to next player.

    With an empty board (no territory regions, no stacks), _get_line_processing_moves
    returns no interactive moves and _get_territory_processing_moves returns no regions,
    so the phase machine should skip TERRITORY_PROCESSING and rotate to next player.
    """
    state = _make_minimal_state(GamePhase.LINE_PROCESSING, current_player=1)
    now = datetime.now()
    move = Move(
        id="m1",
        type=MoveType.PROCESS_LINE,
        player=1,
        to=Position(x=0, y=0),
        timestamp=now,
        thinkTime=0,
        moveNumber=1,
    )

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    # Empty board: no territory regions, no stacks → turn ends
    assert state.current_player == 2
    assert state.current_phase == GamePhase.RING_PLACEMENT
