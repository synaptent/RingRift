"""Tests for gpu_game_types module.

Tests cover:
- get_int_dtype MPS compatibility
- GameStatus, MoveType, GamePhase enum values
- DetectedLine dataclass
- get_required_line_length calculation
- Direction constants
"""

import pytest
import torch

from app.ai.gpu_game_types import (
    get_int_dtype,
    GameStatus,
    MoveType,
    GamePhase,
    DetectedLine,
    get_required_line_length,
    MAX_STACK_HEIGHT,
    SQUARE_DIRECTIONS,
    LINE_DIRECTIONS,
)


class TestGetIntDtype:
    """Tests for get_int_dtype function."""

    def test_cpu_returns_int16(self):
        """CPU device should use int16 for efficiency."""
        device = torch.device('cpu')
        assert get_int_dtype(device) == torch.int16

    def test_mps_returns_int32(self):
        """MPS device should use int32 due to limitations."""
        device = torch.device('mps')
        assert get_int_dtype(device) == torch.int32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_returns_int16(self):
        """CUDA device should use int16 for efficiency."""
        device = torch.device('cuda')
        assert get_int_dtype(device) == torch.int16


class TestGameStatus:
    """Tests for GameStatus enum."""

    def test_values(self):
        """Verify enum values match expected integers."""
        assert GameStatus.ACTIVE == 0
        assert GameStatus.COMPLETED == 1
        assert GameStatus.DRAW == 2
        assert GameStatus.MAX_MOVES == 3

    def test_is_int_enum(self):
        """Verify values can be used as integers."""
        assert int(GameStatus.ACTIVE) == 0
        assert GameStatus.COMPLETED + 1 == 2


class TestMoveType:
    """Tests for MoveType enum."""

    def test_values(self):
        """Verify enum values match app.models.MoveType."""
        assert MoveType.PLACEMENT == 0
        assert MoveType.MOVEMENT == 1
        assert MoveType.CAPTURE == 2
        assert MoveType.LINE_FORMATION == 3
        assert MoveType.TERRITORY_CLAIM == 4
        assert MoveType.SKIP == 5
        assert MoveType.NO_ACTION == 6
        assert MoveType.RECOVERY_SLIDE == 7

    def test_all_values_unique(self):
        """Verify all enum values are unique."""
        values = [m.value for m in MoveType]
        assert len(values) == len(set(values))


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_values(self):
        """Verify phase values for correct ordering."""
        assert GamePhase.RING_PLACEMENT == 0
        assert GamePhase.MOVEMENT == 1
        assert GamePhase.LINE_PROCESSING == 2
        assert GamePhase.TERRITORY_PROCESSING == 3
        assert GamePhase.END_TURN == 4

    def test_phase_ordering(self):
        """Verify phases are ordered for progression."""
        assert GamePhase.RING_PLACEMENT < GamePhase.MOVEMENT
        assert GamePhase.MOVEMENT < GamePhase.LINE_PROCESSING
        assert GamePhase.LINE_PROCESSING < GamePhase.TERRITORY_PROCESSING
        assert GamePhase.TERRITORY_PROCESSING < GamePhase.END_TURN


class TestDetectedLine:
    """Tests for DetectedLine dataclass."""

    def test_creation(self):
        """Test dataclass creation with all fields."""
        line = DetectedLine(
            positions=[(0, 0), (0, 1), (0, 2), (0, 3)],
            length=4,
            is_overlength=False,
            direction=(0, 1),
        )
        assert line.length == 4
        assert len(line.positions) == 4
        assert line.is_overlength is False
        assert line.direction == (0, 1)

    def test_overlength_line(self):
        """Test overlength line detection."""
        line = DetectedLine(
            positions=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            length=5,
            is_overlength=True,
            direction=(0, 1),
        )
        assert line.is_overlength is True
        assert line.length == 5


class TestGetRequiredLineLength:
    """Tests for get_required_line_length function."""

    def test_2_player_8x8(self):
        """8x8 board with 2 players requires 4."""
        assert get_required_line_length(8, 2) == 4

    def test_3_player_8x8(self):
        """8x8 board with 3 players requires 3 (per RR-CANON-R120)."""
        assert get_required_line_length(8, 3) == 3

    def test_4_player_8x8(self):
        """8x8 board with 4 players requires 3."""
        assert get_required_line_length(8, 4) == 3

    def test_2_player_10x10(self):
        """10x10 board with 2 players requires 4."""
        assert get_required_line_length(10, 2) == 4

    def test_3_player_10x10(self):
        """10x10 board with 3 players requires 4 (not 8x8)."""
        assert get_required_line_length(10, 3) == 4

    def test_2_player_12x12(self):
        """12x12 board requires 4."""
        assert get_required_line_length(12, 2) == 4


class TestConstants:
    """Tests for module constants."""

    def test_max_stack_height(self):
        """Verify max stack height constant."""
        assert MAX_STACK_HEIGHT == 8

    def test_square_directions_count(self):
        """Square boards have 8 directions."""
        assert len(SQUARE_DIRECTIONS) == 8

    def test_square_directions_content(self):
        """Verify all 8 directions are present."""
        expected = {
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1), (1, 0), (1, 1),
        }
        assert set(SQUARE_DIRECTIONS) == expected

    def test_line_directions_count(self):
        """Line checking uses 4 directions (no duplicates)."""
        assert len(LINE_DIRECTIONS) == 4

    def test_line_directions_content(self):
        """Verify line directions cover all axes."""
        expected = {
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (1, -1),  # Diagonal down-left
        }
        assert set(LINE_DIRECTIONS) == expected
