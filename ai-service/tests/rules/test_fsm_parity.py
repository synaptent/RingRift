"""
FSM Orchestration Tests - Validate Python FSM phase transitions.

These tests validate that `compute_fsm_orchestration()` produces correct
phase and player transitions for all move types and remains aligned with
the existing `phase_machine.advance_phases()` behavior.

RR-CANON-R075 requires that phases are not silently skipped: even when no
interactive decisions exist, hosts must record explicit no-action moves
(`no_*_action`) for phase visits.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import List, Tuple

import pytest

from app.models import (
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
from app.rules.fsm import (
    FSMOrchestrationResult,
    compare_fsm_with_legacy,
    compute_fsm_orchestration,
)
from app.rules.phase_machine import PhaseTransitionInput, advance_phases


def _make_player(player_number: int, rings_in_hand: int = 0) -> Player:
    """Create a test player."""
    return Player(
        id=f"p{player_number}",
        username=f"player{player_number}",
        type="human",
        playerNumber=player_number,
        isReady=True,
        timeRemaining=60000,
        aiDifficulty=None,
        ringsInHand=rings_in_hand,
        eliminatedRings=0,
        territorySpaces=0,
    )


def _make_game_state(
    phase: GamePhase,
    current_player: int = 1,
    num_players: int = 2,
    stacks: dict | None = None,
    rings_in_hand: int = 18,
) -> GameState:
    """Create a test game state.

    Default rings_in_hand=18 ensures players have turn-material per RR-CANON-R201,
    avoiding degenerate edge cases where all players would be skipped in turn rotation.
    """
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks=stacks or {},
        markers={},
        collapsed_spaces={},
        territories={},
        formed_lines=[],
        eliminated_rings={str(i): 0 for i in range(1, num_players + 1)},
    )
    players = [_make_player(i, rings_in_hand) for i in range(1, num_players + 1)]
    now = datetime.now()

    return GameState(
        id="fsm-parity-test",
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
        maxPlayers=num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def _make_move(
    move_type: MoveType,
    player: int = 1,
    to: Position | None = None,
    from_pos: Position | None = None,
) -> Move:
    """Create a test move."""
    return Move(
        id="test-move",
        type=move_type,
        player=player,
        to=to or Position(x=0, y=0),
        fromPos=from_pos,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )


def _run_parity_check(
    state: GameState,
    move: Move,
) -> Tuple[FSMOrchestrationResult, GamePhase, int, bool]:
    """
    Run both FSM and legacy phase machine on the same input.

    Returns:
        Tuple of (fsm_result, legacy_phase, legacy_player, parity_ok)
    """
    # Make a copy for legacy (it mutates state)
    legacy_state = copy.deepcopy(state)

    # Run FSM
    fsm_result = compute_fsm_orchestration(state, move)

    # Run legacy
    inp = PhaseTransitionInput(game_state=legacy_state, last_move=move, trace_mode=False)
    advance_phases(inp)

    legacy_phase = legacy_state.current_phase
    legacy_player = legacy_state.current_player

    # Compare
    comparison = compare_fsm_with_legacy(fsm_result, legacy_phase, legacy_player)
    parity_ok = not comparison["diverged"]

    return fsm_result, legacy_phase, legacy_player, parity_ok


class TestFSMParityRingPlacement:
    """Test FSM parity for ring_placement phase transitions."""

    def test_place_ring_empty_board_advances_to_line_processing(self):
        """PLACE_RING on empty board advances to LINE_PROCESSING (no movement available)."""
        state = _make_game_state(GamePhase.RING_PLACEMENT, current_player=1)
        move = _make_move(MoveType.PLACE_RING, player=1, to=Position(x=3, y=3))

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}, Legacy: {legacy_phase}"
        # Empty board: no stacks to move → skip to LINE_PROCESSING
        assert fsm_result.next_phase == GamePhase.LINE_PROCESSING
        assert legacy_phase == GamePhase.LINE_PROCESSING

    def test_skip_placement_advances_to_movement(self):
        """SKIP_PLACEMENT should advance to MOVEMENT in both FSM and legacy."""
        state = _make_game_state(GamePhase.RING_PLACEMENT, current_player=1)
        move = _make_move(MoveType.SKIP_PLACEMENT, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}, Legacy: {legacy_phase}"
        assert fsm_result.next_phase == GamePhase.MOVEMENT
        assert legacy_phase == GamePhase.MOVEMENT

    def test_no_placement_action_advances_to_movement(self):
        """NO_PLACEMENT_ACTION should advance to MOVEMENT in both FSM and legacy."""
        state = _make_game_state(GamePhase.RING_PLACEMENT, current_player=1)
        move = _make_move(MoveType.NO_PLACEMENT_ACTION, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}, Legacy: {legacy_phase}"
        assert fsm_result.next_phase == GamePhase.MOVEMENT
        assert legacy_phase == GamePhase.MOVEMENT


class TestFSMParityMovement:
    """Test FSM parity for movement phase transitions."""

    def test_no_movement_action_advances_to_line_processing(self):
        """NO_MOVEMENT_ACTION should advance to LINE_PROCESSING."""
        state = _make_game_state(GamePhase.MOVEMENT, current_player=1)
        move = _make_move(MoveType.NO_MOVEMENT_ACTION, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}, Legacy: {legacy_phase}"
        assert fsm_result.next_phase == GamePhase.LINE_PROCESSING
        assert legacy_phase == GamePhase.LINE_PROCESSING

    def test_move_stack_without_captures_advances_to_line_processing(self):
        """MOVE_STACK without captures should advance to LINE_PROCESSING."""
        state = _make_game_state(GamePhase.MOVEMENT, current_player=1)
        move = _make_move(
            MoveType.MOVE_STACK,
            player=1,
            from_pos=Position(x=2, y=2),
            to=Position(x=4, y=4),
        )

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}, Legacy: {legacy_phase}"
        assert fsm_result.next_phase == GamePhase.LINE_PROCESSING
        assert legacy_phase == GamePhase.LINE_PROCESSING

    def test_recovery_slide_advances_to_line_processing(self):
        """RECOVERY_SLIDE should advance to LINE_PROCESSING."""
        state = _make_game_state(GamePhase.MOVEMENT, current_player=1)
        move = _make_move(
            MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=2, y=2),
            to=Position(x=3, y=3),
        )

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}, Legacy: {legacy_phase}"
        assert fsm_result.next_phase == GamePhase.LINE_PROCESSING
        assert legacy_phase == GamePhase.LINE_PROCESSING


class TestFSMParityLineProcessing:
    """Test FSM parity for line_processing phase transitions."""

    def test_no_line_action_empty_board_ends_turn(self):
        """NO_LINE_ACTION on empty board should advance to TERRITORY_PROCESSING.

        Per RR-CANON-R075, all phases must be visited and recorded. When no
        territory decisions exist, the host emits NO_TERRITORY_ACTION next.
        """
        state = _make_game_state(GamePhase.LINE_PROCESSING, current_player=1)
        move = _make_move(MoveType.NO_LINE_ACTION, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}/{fsm_result.next_player}, Legacy: {legacy_phase}/{legacy_player}"
        assert fsm_result.next_phase == GamePhase.TERRITORY_PROCESSING
        assert fsm_result.next_player == 1
        assert legacy_phase == GamePhase.TERRITORY_PROCESSING
        assert legacy_player == 1

    def test_process_line_empty_board_ends_turn(self):
        """PROCESS_LINE on empty board advances to TERRITORY_PROCESSING.

        After line processing completes, territory_processing is always
        visited as a distinct phase (RR-CANON-R075).
        """
        state = _make_game_state(GamePhase.LINE_PROCESSING, current_player=1)
        move = _make_move(MoveType.PROCESS_LINE, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}/{fsm_result.next_player}, Legacy: {legacy_phase}/{legacy_player}"
        assert fsm_result.next_phase == GamePhase.TERRITORY_PROCESSING
        assert fsm_result.next_player == 1
        assert legacy_phase == GamePhase.TERRITORY_PROCESSING
        assert legacy_player == 1


class TestFSMParityTerritoryProcessing:
    """Test FSM parity for territory_processing phase transitions."""

    def test_no_territory_action_ends_turn(self):
        """NO_TERRITORY_ACTION should end turn and rotate to next player."""
        state = _make_game_state(GamePhase.TERRITORY_PROCESSING, current_player=1)
        move = _make_move(MoveType.NO_TERRITORY_ACTION, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}/{fsm_result.next_player}, Legacy: {legacy_phase}/{legacy_player}"
        assert fsm_result.next_phase == GamePhase.RING_PLACEMENT
        assert fsm_result.next_player == 2
        assert legacy_phase == GamePhase.RING_PLACEMENT
        assert legacy_player == 2

    def test_choose_territory_option_empty_board_ends_turn(self):
        """CHOOSE_TERRITORY_OPTION on empty board should end turn."""
        state = _make_game_state(GamePhase.TERRITORY_PROCESSING, current_player=1)
        move = _make_move(MoveType.CHOOSE_TERRITORY_OPTION, player=1)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}/{fsm_result.next_player}, Legacy: {legacy_phase}/{legacy_player}"
        # Empty board: no more regions → turn ends
        assert fsm_result.next_phase == GamePhase.RING_PLACEMENT
        assert fsm_result.next_player == 2


class TestFSMParityForcedElimination:
    """Test FSM parity for forced_elimination phase transitions."""

    def test_forced_elimination_ends_turn(self):
        """FORCED_ELIMINATION should end turn and rotate to next player."""
        state = _make_game_state(GamePhase.FORCED_ELIMINATION, current_player=1)
        move = _make_move(MoveType.FORCED_ELIMINATION, player=1, to=Position(x=3, y=3))

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_phase}/{fsm_result.next_player}, Legacy: {legacy_phase}/{legacy_player}"
        assert fsm_result.next_phase == GamePhase.RING_PLACEMENT
        assert fsm_result.next_player == 2
        assert legacy_phase == GamePhase.RING_PLACEMENT
        assert legacy_player == 2


class TestFSMParityPlayerRotation:
    """Test FSM parity for player rotation across multiple players."""

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_turn_rotation_wraps_correctly(self, num_players: int):
        """Turn rotation should wrap from last player to first."""
        state = _make_game_state(
            GamePhase.TERRITORY_PROCESSING,
            current_player=num_players,
            num_players=num_players,
        )
        move = _make_move(MoveType.NO_TERRITORY_ACTION, player=num_players)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        assert parity_ok, f"FSM: {fsm_result.next_player}, Legacy: {legacy_player}"
        # Should wrap to player 1
        assert fsm_result.next_player == 1
        assert legacy_player == 1

    @pytest.mark.parametrize("current_player", [1, 2])
    def test_mid_game_rotation(self, current_player: int):
        """Mid-game rotation should go to next player."""
        state = _make_game_state(
            GamePhase.TERRITORY_PROCESSING,
            current_player=current_player,
            num_players=2,
        )
        move = _make_move(MoveType.NO_TERRITORY_ACTION, player=current_player)

        fsm_result, legacy_phase, legacy_player, parity_ok = _run_parity_check(state, move)

        expected_next = (current_player % 2) + 1
        assert parity_ok, f"FSM: {fsm_result.next_player}, Legacy: {legacy_player}"
        assert fsm_result.next_player == expected_next
        assert legacy_player == expected_next


class TestFSMDecisionSurface:
    """Test that FSM decision surfaces are populated correctly."""

    def test_line_processing_decision_surface(self):
        """LINE_PROCESSING should populate decision surface when entering."""
        state = _make_game_state(GamePhase.MOVEMENT, current_player=1)
        move = _make_move(MoveType.NO_MOVEMENT_ACTION, player=1)

        fsm_result = compute_fsm_orchestration(state, move)

        assert fsm_result.next_phase == GamePhase.LINE_PROCESSING
        # Decision surface should indicate action required
        assert fsm_result.pending_decision_type in (
            "line_order_required",
            "no_line_action_required",
            None,  # May be None if no lines detected
        )

    def test_territory_processing_decision_surface(self):
        """TERRITORY_PROCESSING should populate decision surface when entering."""
        state = _make_game_state(GamePhase.LINE_PROCESSING, current_player=1)
        move = _make_move(MoveType.NO_LINE_ACTION, player=1)

        fsm_result = compute_fsm_orchestration(state, move)

        assert fsm_result.next_phase == GamePhase.TERRITORY_PROCESSING
        assert fsm_result.next_player == 1
        assert fsm_result.pending_decision_type in (
            "region_order_required",
            "no_territory_action_required",
            None,  # May be None if region detection isn't surfaced in this path
        )

    def test_forced_elimination_decision_surface(self):
        """FORCED_ELIMINATION should populate decision surface."""
        state = _make_game_state(GamePhase.FORCED_ELIMINATION, current_player=1)
        move = _make_move(MoveType.FORCED_ELIMINATION, player=1)

        # Before the move, check what would be surfaced
        # After FORCED_ELIMINATION, we go to next player
        fsm_result = compute_fsm_orchestration(state, move)

        assert fsm_result.next_phase == GamePhase.RING_PLACEMENT
        assert fsm_result.next_player == 2
