"""Tests for app.rules.generators module.

Tests the move generators including PlacementGenerator, MovementGenerator,
TerritoryGenerator, and CaptureGenerator.

Canonical Spec References:
- RR-CANON-R050: Ring placement rules
- RR-CANON-R055: Multi-ring placement (1-3 rings)
- RR-CANON-R060: No dead placement rule
- RR-CANON-R085: Stack movement rules
- RR-CANON-R095: Overtaking capture initiation
- RR-CANON-R114: Recovery context territory self-elimination
- RR-CANON-R145: Normal territory self-elimination

BUGS FIXED (Dec 2025):
1. PlacementGenerator._generate_all_positions - Fixed to use Position(x=x, y=y)
2. MovementGenerator._is_path_clear - Fixed to use x,y instead of q,r
"""

from unittest.mock import patch

import pytest

from app.models import BoardType, GamePhase, MarkerInfo, MoveType, Position, RingStack, Territory
from app.rules.generators.capture import CaptureGenerator
from app.rules.generators.movement import MovementGenerator
from app.rules.generators.placement import PlacementGenerator
from app.rules.generators.territory import TerritoryGenerator
from app.testing.fixtures import (
    create_board_state,
    create_game_state,
    create_player,
    create_ring_stack,
)


# ============================================================================
# PlacementGenerator Tests
# ============================================================================


class TestPlacementGeneratorBasics:
    """Basic tests for PlacementGenerator initialization and interface."""

    def test_generator_instantiation(self):
        """Test PlacementGenerator can be instantiated."""
        generator = PlacementGenerator()
        assert generator is not None

    def test_generate_returns_list(self):
        """Test generate() returns a list."""
        generator = PlacementGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert isinstance(moves, list)

    def test_no_placements_when_no_rings_in_hand(self):
        """Test no placements when player has no rings in hand."""
        generator = PlacementGenerator()
        players = [create_player(1, rings_in_hand=0), create_player(2, rings_in_hand=10)]
        state = create_game_state(players=players)
        moves = generator.generate(state, player=1)
        assert len(moves) == 0


class TestPlacementGeneratorSquare8:
    """Tests for PlacementGenerator on square8 board."""

    @pytest.fixture
    def generator(self):
        return PlacementGenerator()

    def test_generates_placements_on_empty_board(self, generator):
        """Test generates placements on empty square8 board."""
        state = create_game_state(board_type="square8", num_players=2)
        moves = generator.generate(state, player=1)
        # On empty 8x8 board with rings in hand, should generate placements
        assert len(moves) > 0
        assert all(m.type == MoveType.PLACE_RING for m in moves)


class TestPlacementGeneratorHex8:
    """Tests for PlacementGenerator on hex8 board."""

    @pytest.fixture
    def generator(self):
        return PlacementGenerator()

    def test_generates_placements_on_empty_hex_board(self, generator):
        """Test generates placements on empty hex8 board."""
        state = create_game_state(board_type="hex8", num_players=2)
        moves = generator.generate(state, player=1)
        # On empty hex board with rings in hand, should generate placements
        assert len(moves) > 0
        assert all(m.type == MoveType.PLACE_RING for m in moves)


# ============================================================================
# MovementGenerator Tests
# ============================================================================


class TestMovementGeneratorBasics:
    """Basic tests for MovementGenerator initialization and interface."""

    def test_generator_instantiation(self):
        """Test MovementGenerator can be instantiated."""
        generator = MovementGenerator()
        assert generator is not None

    def test_generate_returns_list(self):
        """Test generate() returns a list (empty board)."""
        generator = MovementGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert isinstance(moves, list)

    def test_no_movements_on_empty_board(self):
        """Test no movements when player has no stacks."""
        generator = MovementGenerator()
        state = create_game_state(board_type="square8", num_players=2)
        moves = generator.generate(state, player=1)
        assert len(moves) == 0


class TestMovementGeneratorSquare8:
    """Tests for MovementGenerator on square8 board."""

    @pytest.fixture
    def generator(self):
        return MovementGenerator()

    def test_generates_movements_for_owned_stack(self, generator):
        """Test generates movements for player's stack."""
        stack = create_ring_stack(x=3, y=3, rings=[1, 1])  # Height 2
        board = create_board_state(
            board_type="square8",
            stacks={"3,3": stack},
        )
        state = create_game_state(board=board, num_players=2)
        state.current_phase = GamePhase.MOVEMENT
        moves = generator.generate(state, player=1)

        # Stack with height 2 should generate movement moves
        move_stack_moves = [m for m in moves if m.type == MoveType.MOVE_STACK]
        assert len(move_stack_moves) > 0


class TestMovementGeneratorHex8:
    """Tests for MovementGenerator on hex8 board."""

    @pytest.fixture
    def generator(self):
        return MovementGenerator()


# ============================================================================
# TerritoryGenerator Tests
# ============================================================================


class TestTerritoryGeneratorBasics:
    """Basic tests for TerritoryGenerator initialization and interface."""

    def test_generator_instantiation(self):
        """Test TerritoryGenerator can be instantiated."""
        generator = TerritoryGenerator()
        assert generator is not None

    def test_generate_returns_list(self):
        """Test generate() returns a list."""
        generator = TerritoryGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert isinstance(moves, list)

    def test_no_territory_moves_without_regions(self):
        """Test no territory moves when no disconnected regions exist."""
        generator = TerritoryGenerator()
        state = create_game_state(board_type="square8", num_players=2)
        moves = generator.generate(state, player=1)
        # Without disconnected regions, should return empty list
        assert len(moves) == 0


class TestTerritoryGeneratorDetection:
    """Tests for territory detection and processing."""

    @pytest.fixture
    def generator(self):
        return TerritoryGenerator()

    def test_skip_territory_option_available(self, generator):
        """Test SKIP_TERRITORY_PROCESSING option when regions exist."""
        marker = MarkerInfo(
            player=1,
            position=Position(x=0, y=0),
            type="ownership",
        )
        board = create_board_state(
            board_type="square8",
            markers={"0,0": marker},
        )
        state = create_game_state(board=board, board_type="square8", num_players=2)
        region = Territory(
            spaces=[Position(x=1, y=1)],
            controllingPlayer=1,
            isDisconnected=True,
        )

        with patch(
            "app.rules.generators.territory.BoardManager.find_disconnected_regions",
            return_value=[region],
        ), patch.object(
            TerritoryGenerator,
            "_can_process_region",
            return_value=True,
        ):
            moves = generator.generate(state, player=1)

        move_types = {move.type for move in moves}
        assert MoveType.CHOOSE_TERRITORY_OPTION in move_types
        assert MoveType.SKIP_TERRITORY_PROCESSING in move_types


# ============================================================================
# CaptureGenerator Tests
# ============================================================================


class TestCaptureGeneratorBasics:
    """Basic tests for CaptureGenerator initialization and interface."""

    def test_generator_instantiation(self):
        """Test CaptureGenerator can be instantiated."""
        generator = CaptureGenerator()
        assert generator is not None

    def test_generate_returns_list(self):
        """Test generate() returns a list."""
        generator = CaptureGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert isinstance(moves, list)

    def test_no_captures_on_empty_board(self):
        """Test no captures when player has no stacks."""
        generator = CaptureGenerator()
        state = create_game_state(board_type="square8", num_players=2)
        moves = generator.generate(state, player=1)
        assert len(moves) == 0


class TestCaptureGeneratorSquare8:
    """Tests for CaptureGenerator on square8 board."""

    @pytest.fixture
    def generator(self):
        return CaptureGenerator()

    def test_can_capture_weaker_stack(self, generator):
        """Test can capture opponent's weaker stack (RR-CANON-R095)."""
        # Player 1 has height-3 stack, Player 2 has height-2 stack nearby
        attacker = create_ring_stack(x=0, y=0, rings=[1, 1, 1])
        target = create_ring_stack(x=0, y=3, rings=[2, 2])
        board = create_board_state(
            board_type="square8",
            stacks={
                "0,0": attacker,
                "0,3": target,
            },
        )
        state = create_game_state(board=board, num_players=2)
        moves = generator.generate(state, player=1)

        # Should have at least one capture move
        capture_moves = [m for m in moves if m.type == MoveType.OVERTAKING_CAPTURE]
        assert len(capture_moves) > 0

    def test_cannot_capture_equal_or_stronger_stack(self, generator):
        """Test cannot capture stack of equal or greater height."""
        # Both stacks have height 2
        attacker = create_ring_stack(x=0, y=0, rings=[1, 1])
        target = create_ring_stack(x=0, y=2, rings=[2, 2])
        board = create_board_state(
            board_type="square8",
            stacks={
                "0,0": attacker,
                "0,2": target,
            },
        )
        state = create_game_state(board=board, num_players=2)
        moves = generator.generate(state, player=1)

        # Should not be able to capture equal-height stack
        for move in moves:
            if move.type == MoveType.OVERTAKING_CAPTURE:
                # If there are any captures, they shouldn't target (0,2)
                assert not (move.to.x == 0 and move.to.y == 2)


# ============================================================================
# Edge Cases and Cross-Generator Tests
# ============================================================================


class TestGeneratorEdgeCases:
    """Edge case tests across all generators."""

    def test_placement_all_cells_filled(self):
        """Test placement when all cells are filled."""
        # TODO: Implement when needed
        pass


class TestGeneratorLimitParameter:
    """Tests for limit parameter optimization."""

    def test_placement_respects_limit(self):
        """Test PlacementGenerator respects limit parameter."""
        generator = PlacementGenerator()
        state = create_game_state(board_type="square8", num_players=2)
        moves = generator.generate(state, player=1, limit=5)
        assert len(moves) <= 5

    def test_movement_respects_limit(self):
        """Test MovementGenerator respects limit parameter."""
        generator = MovementGenerator()
        stack = create_ring_stack(x=3, y=3, rings=[1, 1, 1])  # Height 3, many moves possible
        board = create_board_state(
            board_type="square8",
            stacks={"3,3": stack},
        )
        state = create_game_state(board=board, num_players=2)
        state.current_phase = GamePhase.MOVEMENT
        moves = generator.generate(state, player=1, limit=3)
        assert len(moves) <= 3

    def test_capture_respects_limit(self):
        """Test CaptureGenerator respects limit parameter."""
        generator = CaptureGenerator()
        attacker = create_ring_stack(x=3, y=3, rings=[1, 1, 1])
        target1 = create_ring_stack(x=3, y=5, rings=[2])
        target2 = create_ring_stack(x=5, y=3, rings=[2])
        board = create_board_state(
            board_type="square8",
            stacks={
                "3,3": attacker,
                "3,5": target1,
                "5,3": target2,
            },
        )
        state = create_game_state(board=board, num_players=2)

        moves = generator.generate(state, player=1, limit=1)
        assert len(moves) <= 1
