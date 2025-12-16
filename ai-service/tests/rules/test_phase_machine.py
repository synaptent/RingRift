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
    """Create a minimal active GameState for phase-machine tests.

    Default ringsInHand=18 ensures players have turn-material per RR-CANON-R201,
    avoiding degenerate edge cases where all players would be skipped in turn rotation.
    """
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
            ringsInHand=18,  # Realistic starting count for turn-material
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
            ringsInHand=18,  # Realistic starting count for turn-material
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


def test_skip_capture_advances_to_line_processing():
    """SKIP_CAPTURE should advance to LINE_PROCESSING."""
    state = _make_minimal_state(GamePhase.CAPTURE, current_player=1)
    move = _make_noop_move(MoveType.SKIP_CAPTURE, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    assert state.current_phase == GamePhase.LINE_PROCESSING


def test_entering_line_processing_clears_must_move_from_stack_key():
    """
    mustMoveFromStackKey must be cleared when transitioning into LINE_PROCESSING.

    TS clears mustMoveFromStackKey when entering line_processing because the
    constraint is only relevant during movement/capture phases. Leaving it set
    can create stale keys (e.g., when the constrained stack is eliminated by
    landing-on-marker cap removal) and cause TS↔Python ANM parity mismatches.
    """
    state = _make_minimal_state(GamePhase.MOVEMENT, current_player=1)
    state.must_move_from_stack_key = "5,0"

    move = _make_noop_move(MoveType.NO_MOVEMENT_ACTION, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    assert state.current_phase == GamePhase.LINE_PROCESSING
    assert state.must_move_from_stack_key is None


def test_no_line_action_with_empty_board_enters_territory_processing():
    """
    NO_LINE_ACTION on empty board should advance to TERRITORY_PROCESSING.

    Per RR-CANON-R075, every phase must be visited and produce a recorded
    action. After NO_LINE_ACTION, we always enter TERRITORY_PROCESSING so
    hosts can emit NO_TERRITORY_ACTION when there are no regions. The turn
    rotation happens after NO_TERRITORY_ACTION is applied.
    """
    state = _make_minimal_state(GamePhase.LINE_PROCESSING, current_player=1)
    move = _make_noop_move(MoveType.NO_LINE_ACTION, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    # Per RR-CANON-R075: always visit TERRITORY_PROCESSING
    assert state.current_player == 1
    assert state.current_phase == GamePhase.TERRITORY_PROCESSING


def test_no_territory_action_rotates_to_next_player_ring_placement():
    """NO_TERRITORY_ACTION should rotate to next player in RING_PLACEMENT."""
    state = _make_minimal_state(GamePhase.TERRITORY_PROCESSING, current_player=1)
    move = _make_noop_move(MoveType.NO_TERRITORY_ACTION, player=1)

    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)

    assert state.current_player == 2
    assert state.current_phase == GamePhase.RING_PLACEMENT


def test_process_line_with_empty_board_enters_territory_processing():
    """
    PROCESS_LINE on empty board with no remaining lines should advance to TERRITORY_PROCESSING.

    Per RR-CANON-R075, every phase must be visited and produce a recorded action.
    With an empty board (no territory regions, no stacks), _get_line_processing_moves
    returns no interactive moves, so we advance to TERRITORY_PROCESSING. Hosts will
    then emit NO_TERRITORY_ACTION, and the turn rotation happens afterward.
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

    # Per RR-CANON-R075: always visit TERRITORY_PROCESSING
    assert state.current_player == 1
    assert state.current_phase == GamePhase.TERRITORY_PROCESSING


# -----------------------------------------------------------------------------
# Regression tests for skip_placement bookkeeping classification (GH-FE-FIX)
# -----------------------------------------------------------------------------
from app.rules.phase_machine import (
    _is_no_action_bookkeeping_move,
    compute_had_any_action_this_turn,
)


def test_skip_placement_is_bookkeeping_move():
    """
    SKIP_PLACEMENT must be classified as a bookkeeping move (non-real action).

    Per RR-CANON lpsTracking.ts lines 11-12:
      "Non-real actions (that don't count for LPS): skip_placement, forced elimination,
       line/territory processing decisions."

    When a player skips placement but then has no movement/capture available,
    the S-metric (markers + collapsed + eliminated) does not increase, and
    forced elimination must trigger to ensure game termination.

    Regression test for 3p infinite loop bug where games got stuck in
    skip_placement → no_placement_action → no_placement_action cycles.
    """
    assert _is_no_action_bookkeeping_move(MoveType.SKIP_PLACEMENT) is True


def test_all_no_action_moves_are_bookkeeping():
    """All NO_*_ACTION move types should be classified as bookkeeping moves."""
    bookkeeping_types = [
        MoveType.NO_PLACEMENT_ACTION,
        MoveType.NO_MOVEMENT_ACTION,
        MoveType.NO_LINE_ACTION,
        MoveType.NO_TERRITORY_ACTION,
        MoveType.SKIP_PLACEMENT,
    ]
    for move_type in bookkeeping_types:
        assert _is_no_action_bookkeeping_move(move_type) is True, (
            f"{move_type} should be a bookkeeping move"
        )


def test_real_action_moves_are_not_bookkeeping():
    """Real action move types should NOT be classified as bookkeeping moves."""
    real_action_types = [
        MoveType.PLACE_RING,
        MoveType.MOVE_STACK,
        MoveType.OVERTAKING_CAPTURE,
        MoveType.CHAIN_CAPTURE,
        MoveType.RECOVERY_SLIDE,
        MoveType.FORCED_ELIMINATION,  # Per LPS rules: FE is a "non-real action" but NOT bookkeeping
    ]
    for move_type in real_action_types:
        assert _is_no_action_bookkeeping_move(move_type) is False, (
            f"{move_type} should NOT be a bookkeeping move"
        )


def test_compute_had_any_action_returns_false_after_skip_placement_only():
    """
    After a turn with only skip_placement + NO_*_ACTION moves, had_any_action should be False.

    This is critical for the forced elimination flow: when a player skips placement
    and has no movement/capture available, they made zero real progress and FE must
    trigger if they still have stacks on the board.

    Regression test for 3p infinite loop bug.
    """
    state = _make_minimal_state(GamePhase.TERRITORY_PROCESSING, current_player=1)
    now = datetime.now()

    # Simulate a turn where player 1 made only skip_placement + bookkeeping moves
    state.move_history = [
        Move(
            id="m1",
            type=MoveType.SKIP_PLACEMENT,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=1,
        ),
        Move(
            id="m2",
            type=MoveType.NO_MOVEMENT_ACTION,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=2,
        ),
        Move(
            id="m3",
            type=MoveType.NO_LINE_ACTION,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=3,
        ),
        Move(
            id="m4",
            type=MoveType.NO_TERRITORY_ACTION,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=4,
        ),
    ]

    # With SKIP_PLACEMENT correctly classified as bookkeeping,
    # compute_had_any_action_this_turn should return False
    assert compute_had_any_action_this_turn(state) is False


def test_compute_had_any_action_returns_true_after_place_ring():
    """
    After a turn with PLACE_RING (even followed by NO_*_ACTION), had_any_action should be True.

    PLACE_RING is a real action that counts toward progress, unlike SKIP_PLACEMENT.
    """
    state = _make_minimal_state(GamePhase.TERRITORY_PROCESSING, current_player=1)
    now = datetime.now()

    # Simulate a turn where player 1 placed a ring then had no moves
    state.move_history = [
        Move(
            id="m1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=3, y=3),
            timestamp=now,
            thinkTime=0,
            moveNumber=1,
        ),
        Move(
            id="m2",
            type=MoveType.NO_MOVEMENT_ACTION,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=2,
        ),
        Move(
            id="m3",
            type=MoveType.NO_LINE_ACTION,
            player=1,
            to=Position(x=0, y=0),
            timestamp=now,
            thinkTime=0,
            moveNumber=3,
        ),
    ]

    # PLACE_RING is a real action, so had_any_action should be True
    assert compute_had_any_action_this_turn(state) is True


# -----------------------------------------------------------------------------
# Regression test for forced_elimination → current_player parity (RR-PARITY-FIX-2025-12-16)
# -----------------------------------------------------------------------------

def test_forced_elimination_rotates_player_correctly():
    """
    After FORCED_ELIMINATION, current_player must advance to the next non-eliminated player.
    
    This is a regression test for the TS↔Python parity divergence fixed in RR-PARITY-FIX-2025-12-16.
    
    Root cause:
    - TypeScript uses `computeNextNonEliminatedPlayer(gameState, move.player, numPlayers)` to compute
      the next player after forced_elimination, starting from `move.player`.
    - Python was using `_rotate_to_next_active_player()` which starts from `game_state.current_player`.
    - When `move.player != game_state.current_player` (which can happen in certain edge cases),
      the two engines would disagree on the next player.
    
    Fix:
    - Python now uses `_end_turn()` which properly handles the turn rotation to match TS semantics.
    
    Per RR-CANON-R070: forced_elimination is the 7th and final phase of a turn.
    After FE, the turn ends and rotates to the next player.
    """
    state = _make_minimal_state(GamePhase.FORCED_ELIMINATION, current_player=1)
    
    # Ensure player 1 has stacks (needed for FE to be valid)
    from app.models import RingStack
    stack_pos = Position(x=3, y=3)
    state.board.stacks[stack_pos.to_key()] = RingStack(
        position=stack_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    
    # Create a FORCED_ELIMINATION move from player 1
    move = Move(
        id="fe1",
        type=MoveType.FORCED_ELIMINATION,
        player=1,
        to=stack_pos,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )
    
    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)
    
    # After player 1's FE, it should be player 2's turn in RING_PLACEMENT
    assert state.current_player == 2, (
        f"Expected current_player=2 after player 1's FE, got {state.current_player}"
    )
    assert state.current_phase == GamePhase.RING_PLACEMENT, (
        f"Expected phase=RING_PLACEMENT after FE, got {state.current_phase}"
    )


def test_forced_elimination_skips_eliminated_players():
    """
    FORCED_ELIMINATION rotation must skip players who are permanently eliminated.
    
    Per RR-CANON-R201: Players without ANY rings (controlled, buried, or in hand) are permanently
    eliminated and must be skipped during turn rotation.
    
    This test ensures the rotation after FE properly skips eliminated players.
    """
    state = _make_minimal_state(GamePhase.FORCED_ELIMINATION, current_player=1)
    
    # Make player 2 eliminated (no rings in hand, no stacks)
    p2 = next(p for p in state.players if p.player_number == 2)
    p2.rings_in_hand = 0
    p2.eliminated_rings = 18  # All rings eliminated
    
    # Player 1 still has a stack (needed for FE)
    from app.models import RingStack
    stack_pos = Position(x=3, y=3)
    state.board.stacks[stack_pos.to_key()] = RingStack(
        position=stack_pos,
        rings=[1, 1],
        stackHeight=2,
        capHeight=2,
        controllingPlayer=1,
    )
    
    move = Move(
        id="fe1",
        type=MoveType.FORCED_ELIMINATION,
        player=1,
        to=stack_pos,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )
    
    inp = PhaseTransitionInput(game_state=state, last_move=move, trace_mode=False)
    advance_phases(inp)
    
    # In a 2-player game with player 2 eliminated, player 1 should get the next turn.
    # But since player 1 just finished their turn, and player 2 is eliminated,
    # the game should either:
    # 1. Rotate back to player 1 (if player 1 still has material), or
    # 2. End in victory if player 1 caused player 2's elimination
    #
    # With our minimal state setup, player 1 should get the next turn since player 2
    # has no material.
    assert state.current_player == 1, (
        f"Expected current_player=1 (rotation skips eliminated player 2), got {state.current_player}"
    )
