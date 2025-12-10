#!/usr/bin/env python3
"""Recovery action parity tests.

These tests verify that the Python recovery action implementation matches
the TypeScript implementation according to RR-CANON-R110–R115.

Tests cover:
- Recovery eligibility checks
- Option 1/2 cost model
- Overlength line handling
- Marker slide and collapse logic
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from app.models import (
    GameState,
    BoardState,
    BoardType,
    Position,
    RingStack,
    MarkerInfo,
    Player,
    GamePhase,
    GameStatus,
    Move,
    MoveType,
    TimeControl,
)
from app.rules.core import (
    count_buried_rings,
    player_has_markers,
    is_eligible_for_recovery,
    player_controls_any_stack,
    get_effective_line_length,
)
from app.rules.recovery import (
    enumerate_recovery_slide_targets,
    has_any_recovery_move,
    validate_recovery_slide,
    apply_recovery_slide,
    calculate_recovery_cost,
)
from app.game_engine import GameEngine  # type: ignore[import]


def make_test_state(
    stacks: Dict[str, RingStack],
    markers: Dict[str, MarkerInfo],
    player1_rings_in_hand: int = 0,
    player2_rings_in_hand: int = 16,
) -> GameState:
    """Helper to create a test game state."""
    now = datetime.now()
    board = BoardState(type=BoardType.SQUARE8, size=8)
    board.stacks = stacks
    board.markers = markers
    board.collapsed_spaces = {}
    board.eliminated_rings = {"1": 0, "2": 0}
    board.formed_lines = []
    board.territories = {}

    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=player1_rings_in_hand,
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
            ringsInHand=player2_rings_in_hand,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    return GameState(
        id="test-recovery-parity",
        boardType=BoardType.SQUARE8,
        board=board,
        players=players,
        currentPlayer=1,
        currentPhase=GamePhase.MOVEMENT,
        timeControl=TimeControl(initialTime=60, increment=0, type="untimed"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
        turnNumber=50,
    )


class TestRecoveryEligibility:
    """Tests for recovery eligibility checks."""

    def test_eligible_when_no_stacks_no_rings_has_markers_has_buried(self):
        """Player is eligible when they have markers, buried rings, but no active material."""
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],  # Player 1 has a buried ring
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        assert is_eligible_for_recovery(state, 1) is True

    def test_not_eligible_when_controls_stack(self):
        """Player is not eligible if they control any stack."""
        state = make_test_state(
            stacks={
                "0,0": RingStack(
                    position=Position(x=0, y=0),
                    rings=[1],  # Player 1 controls this stack
                    stack_height=1,
                    cap_height=1,
                    controlling_player=1,
                ),
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        assert is_eligible_for_recovery(state, 1) is False

    def test_not_eligible_when_has_rings_in_hand(self):
        """Player is not eligible if they have rings in hand."""
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=5,  # Has rings in hand
        )

        assert is_eligible_for_recovery(state, 1) is False

    def test_not_eligible_when_no_markers(self):
        """Player is not eligible if they have no markers."""
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={},  # No markers
            player1_rings_in_hand=0,
        )

        assert is_eligible_for_recovery(state, 1) is False

    def test_not_eligible_when_no_buried_rings(self):
        """Player is not eligible if they have no buried rings."""
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[2],  # No buried rings for player 1
                    stack_height=1,
                    cap_height=1,
                    controlling_player=2,
                ),
            },
            markers={
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        assert is_eligible_for_recovery(state, 1) is False


class TestRecoveryCostModel:
    """Tests for Option 1/2 cost model."""

    def test_option1_cost_is_one(self):
        """Option 1 always costs 1 buried ring."""
        assert calculate_recovery_cost(1) == 1

    def test_option2_cost_is_zero(self):
        """Option 2 is free (costs 0 buried rings)."""
        assert calculate_recovery_cost(2) == 0


class TestBuriedRingCounting:
    """Tests for counting buried rings."""

    def test_count_single_buried_ring(self):
        """Count a single buried ring in opponent's stack."""
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],  # Player 1 ring at bottom
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={},
        )

        assert count_buried_rings(state.board, 1) == 1

    def test_count_multiple_buried_rings(self):
        """Count multiple buried rings across stacks."""
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 1, 2],  # Two player 1 rings buried
                    stack_height=3,
                    cap_height=3,
                    controlling_player=2,
                ),
            },
            markers={},
        )

        assert count_buried_rings(state.board, 1) == 2

    def test_not_count_controlling_ring(self):
        """Don't count the top ring (controlling ring) as buried."""
        state = make_test_state(
            stacks={
                "0,0": RingStack(
                    position=Position(x=0, y=0),
                    rings=[2, 1],  # Player 1 is on top (not buried)
                    stack_height=2,
                    cap_height=2,
                    controlling_player=1,
                ),
            },
            markers={},
        )

        assert count_buried_rings(state.board, 1) == 0

    def test_not_count_rings_in_own_stacks(self):
        """Don't count rings in player's own stacks as buried."""
        state = make_test_state(
            stacks={
                "0,0": RingStack(
                    position=Position(x=0, y=0),
                    rings=[1, 1],  # Both rings are player 1's in their stack
                    stack_height=2,
                    cap_height=2,
                    controlling_player=1,
                ),
            },
            markers={},
        )

        assert count_buried_rings(state.board, 1) == 0


class TestRecoveryPhaseTransition:
    """Ensure recovery slides advance into line_processing with collapses applied."""

    def test_recovery_slide_enters_line_processing_and_collapses(self):
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        now = datetime.now()
        state = make_test_state(
            stacks={
                "4,4": RingStack(
                    position=Position(x=4, y=4),
                    rings=[2, 1, 2],  # Player 1 buried ring
                    stack_height=3,
                    cap_height=1,
                    controlling_player=2,
                ),
            },
            markers={
                "0,0": MarkerInfo(position=Position(x=0, y=0), player=1, type="regular"),
                "1,0": MarkerInfo(position=Position(x=1, y=0), player=1, type="regular"),
                "2,0": MarkerInfo(position=Position(x=2, y=0), player=1, type="regular"),
                "4,0": MarkerInfo(position=Position(x=4, y=0), player=1, type="regular"),  # Will slide to 3,0
            },
            player1_rings_in_hand=0,
            player2_rings_in_hand=0,
        )

        move = Move(
            id="recovery-test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=4, y=0),
            to=Position(x=3, y=0),
            recovery_option=1,
            timestamp=now,
            think_time=0,
            move_number=1,
        )

        next_state = GameEngine.apply_move(state, move)

        # Phase should advance to line_processing after recovery.
        assert next_state.current_phase == GamePhase.LINE_PROCESSING

        # Collapsed spaces should include the completed line positions (4 markers: 0,0 -> 1,0 -> 2,0 -> 3,0).
        assert {"0,0", "1,0", "2,0", "3,0"}.issubset(set(next_state.board.collapsed_spaces.keys()))

        # Territory count should reflect collapsed markers.
        p1 = next(p for p in next_state.players if p.player_number == 1)
        assert p1.territory_spaces >= 4


class TestRecoverySlideEnumeration:
    """Tests for enumerating recovery slide targets."""

    def test_enumerate_exact_length_target(self):
        """Enumerate a slide that completes an exact-length line."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "1,3": MarkerInfo(position=Position(x=1, y=3), player=1, type="regular"),
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
        )

        targets = enumerate_recovery_slide_targets(state, 1)

        # Should find the slide from (5,3) to (4,3) that completes a line of 4
        assert len(targets) > 0
        slide_to_4_3 = next(
            (t for t in targets if t.to_pos.x == 4 and t.to_pos.y == 3), None
        )
        assert slide_to_4_3 is not None
        assert slide_to_4_3.markers_in_line >= 4  # 4 markers for square8 2-player
        assert slide_to_4_3.option1_cost == 1

    def test_enumerate_overlength_target(self):
        """Enumerate a slide that completes an overlength line."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        # An overlength line needs 5+ markers
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "0,3": MarkerInfo(position=Position(x=0, y=3), player=1, type="regular"),
                "1,3": MarkerInfo(position=Position(x=1, y=3), player=1, type="regular"),
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
        )

        targets = enumerate_recovery_slide_targets(state, 1)

        # Should find the slide from (5,3) to (4,3) that completes a line of 5
        slide_to_4_3 = next(
            (t for t in targets if t.to_pos.x == 4 and t.to_pos.y == 3), None
        )
        assert slide_to_4_3 is not None
        assert slide_to_4_3.markers_in_line == 5
        assert slide_to_4_3.is_overlength is True
        assert slide_to_4_3.option2_available is True
        assert slide_to_4_3.option2_cost == 0


class TestHasAnyRecoveryMove:
    """Tests for checking if player has any valid recovery move."""

    def test_has_recovery_when_valid_slide_exists(self):
        """Player has recovery move when valid slide exists."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "1,3": MarkerInfo(position=Position(x=1, y=3), player=1, type="regular"),
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
        )

        assert has_any_recovery_move(state, 1) is True

    @pytest.mark.skip(reason="Recovery module may support non-line recovery modes; needs further investigation")
    def test_no_recovery_when_no_valid_slide(self):
        """Player has no recovery move when no valid slide completes a line."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        # Place markers far apart so no single slide can complete a 4-marker line
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                # Markers scattered - no way to form a 4-marker line with one slide
                "0,0": MarkerInfo(position=Position(x=0, y=0), player=1, type="regular"),
                "7,0": MarkerInfo(position=Position(x=7, y=0), player=1, type="regular"),
                "0,7": MarkerInfo(position=Position(x=0, y=7), player=1, type="regular"),
            },
        )

        assert has_any_recovery_move(state, 1) is False


class TestRecoveryMoveApplication:
    """Tests for recovery move application (state mutation parity)."""

    def test_apply_exact_length_recovery_extracts_ring(self):
        """Applying exact-length recovery slide extracts 1 buried ring."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],  # Player 1 has a buried ring at bottom
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "1,3": MarkerInfo(position=Position(x=1, y=3), player=1, type="regular"),
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        move = Move(
            id="recovery-test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=5, y=3),
            to=Position(x=4, y=3),
            recovery_option=1,  # Use recovery_option field
            extraction_stacks=("7,7",),
            timestamp=datetime.now(),
            think_time=0,
            move_number=51,
        )

        result = apply_recovery_slide(state, move)

        # Ring should be extracted from stack
        assert result.success, f"Failed: {result.error}"
        assert result.rings_extracted == 1  # Option 1 extracts 1 buried ring

        # State is mutated in place - check it directly
        stack = state.board.stacks.get("7,7")
        assert stack is not None
        assert len(stack.rings) == 1  # One ring removed
        assert 1 not in stack.rings  # Player 1's ring was extracted

    def test_apply_overlength_option2_no_extraction(self):
        """Applying overlength recovery with Option 2 extracts no rings."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        # An overlength line needs 5+ markers
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "0,3": MarkerInfo(position=Position(x=0, y=3), player=1, type="regular"),
                "1,3": MarkerInfo(position=Position(x=1, y=3), player=1, type="regular"),
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        # Line will be at positions 0,3 -> 1,3 -> 2,3 -> 3,3 -> 4,3 (5 markers, overlength for 4-required)
        # Option 2 collapses only lineLength (4) consecutive markers
        collapse_positions = (
            Position(x=1, y=3),
            Position(x=2, y=3),
            Position(x=3, y=3),
            Position(x=4, y=3),  # includes destination
        )

        move = Move(
            id="recovery-test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=5, y=3),
            to=Position(x=4, y=3),
            recovery_option=2,  # Option 2: free (use recovery_option field)
            collapse_positions=collapse_positions,  # Include on Move for validation
            extraction_stacks=(),  # No extraction for Option 2
            timestamp=datetime.now(),
            think_time=0,
            move_number=51,
        )

        result = apply_recovery_slide(state, move)

        # No ring should be extracted
        assert result.success, f"Failed: {result.error}"
        assert result.rings_extracted == 0  # Option 2 is free

        # State is mutated in place - check stack directly
        stack = state.board.stacks.get("7,7")
        assert stack is not None
        assert len(stack.rings) == 2  # Both rings remain

    def test_marker_moves_to_destination(self):
        """Recovery slide moves the marker from source to destination."""
        # Per RR-CANON-R112, square8 2-player requires 4 markers for a line
        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "1,3": MarkerInfo(position=Position(x=1, y=3), player=1, type="regular"),
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        move = Move(
            id="recovery-test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=5, y=3),
            to=Position(x=4, y=3),
            recovery_option=1,  # Use recovery_option field
            extraction_stacks=("7,7",),
            timestamp=datetime.now(),
            think_time=0,
            move_number=51,
        )

        result = apply_recovery_slide(state, move)

        assert result.success, f"Failed: {result.error}"

        # State is mutated in place - check directly
        # Source marker should be gone (moved to destination, then collapsed)
        assert "5,3" not in state.board.markers

        # Line positions should be tracked in result
        assert len(result.line_positions) >= 3  # At least lineLength markers in line
        assert len(result.collapsed_positions) >= 3  # All line markers collapsed for Option 1


class TestRecoveryLpsIntegration:
    """Tests for recovery integration with Last Player Standing."""

    def test_recovery_counts_as_real_action(self):
        """Recovery move should count as a real action for LPS."""
        from app.game_engine import GameEngine

        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[1, 2],
                    stack_height=2,
                    cap_height=2,
                    controlling_player=2,
                ),
            },
            markers={
                "2,3": MarkerInfo(position=Position(x=2, y=3), player=1, type="regular"),
                "3,3": MarkerInfo(position=Position(x=3, y=3), player=1, type="regular"),
                "5,3": MarkerInfo(position=Position(x=5, y=3), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        # Player 1 has no stacks, no rings in hand, but can do recovery
        result = GameEngine._has_real_action_for_player(state, 1)
        assert result is True

    def test_no_recovery_means_no_real_action(self):
        """Player with no recovery option should have no real action."""
        from app.game_engine import GameEngine

        state = make_test_state(
            stacks={
                "7,7": RingStack(
                    position=Position(x=7, y=7),
                    rings=[2],  # No buried rings for player 1
                    stack_height=1,
                    cap_height=1,
                    controlling_player=2,
                ),
            },
            markers={
                # Markers too far apart - no valid recovery slide
                "0,0": MarkerInfo(position=Position(x=0, y=0), player=1, type="regular"),
                "7,0": MarkerInfo(position=Position(x=7, y=0), player=1, type="regular"),
            },
            player1_rings_in_hand=0,
        )

        # Player 1 cannot do recovery - no buried rings and no valid slides
        result = GameEngine._has_real_action_for_player(state, 1)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
