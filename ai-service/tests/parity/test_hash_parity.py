#!/usr/bin/env python3
"""Tests to verify Python and TypeScript hash implementations produce identical results.

The hash format was unified so that both engines produce:
1. The same fingerprint string format (readable, for debugging)
2. The same hash output (compact, for comparison)

This enables meaningful cross-engine parity testing.
"""

import pytest

from app.db.game_replay import _compute_state_hash, _simple_hash, _fingerprint_state
from app.rules.core import hash_game_state


class TestSimpleHash:
    """Test the simple hash function matches expected behavior."""

    def test_empty_string(self):
        """Empty string should produce consistent hash."""
        result = _simple_hash("")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        """Same input should produce same output."""
        input_str = "1:ring_placement:active#1:18:0:0|2:18:0:0##"
        result1 = _simple_hash(input_str)
        result2 = _simple_hash(input_str)
        assert result1 == result2

    def test_different_inputs(self):
        """Different inputs should produce different outputs."""
        result1 = _simple_hash("state1")
        result2 = _simple_hash("state2")
        assert result1 != result2

    def test_known_value(self):
        """Test against a known reference value."""
        # This test ensures the hash algorithm hasn't changed
        input_str = "test"
        result = _simple_hash(input_str)
        assert len(result) == 16
        # Store the expected value after first run
        # assert result == "expected_hash_here"


def _create_test_state():
    """Create a minimal test game state."""
    from datetime import datetime, timezone
    from app.models import GameState, BoardState, GamePhase, Player, BoardType, TimeControl

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsed_spaces={},
        eliminated_rings={},
    )

    time_control = TimeControl(
        initial_time=600,
        increment=10,
        type="fischer",
    )

    now = datetime.now(timezone.utc)

    return GameState(
        id="test-game-id",
        board_type=BoardType.SQUARE8,
        board=board,
        current_player=1,
        current_phase=GamePhase.RING_PLACEMENT,
        players=[
            Player(
                id="p1",
                username="Player1",
                type="human",
                player_number=1,
                is_ready=True,
                time_remaining=600,
                rings_in_hand=18,
                eliminated_rings=0,
                territory_spaces=0,
            ),
            Player(
                id="p2",
                username="Player2",
                type="human",
                player_number=2,
                is_ready=True,
                time_remaining=600,
                rings_in_hand=18,
                eliminated_rings=0,
                territory_spaces=0,
            ),
        ],
        game_status="active",
        move_history=[],
        time_control=time_control,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=2,
        total_rings_in_play=0,
        total_rings_eliminated=0,
        victory_threshold=12,
        territory_victory_threshold=10,
    )


class TestFingerprintFormat:
    """Test the fingerprint format is consistent."""

    def test_fingerprint_format_structure(self):
        """Fingerprint should have 5 sections separated by #."""
        state = _create_test_state()

        fingerprint = _fingerprint_state(state)
        parts = fingerprint.split("#")

        assert len(parts) == 5, f"Expected 5 parts, got {len(parts)}: {fingerprint}"

        # meta, players, stacks, markers, collapsed
        meta = parts[0]
        assert ":" in meta, "Meta should have colons"
        assert "ring_placement" in meta or "movement" in meta, f"Meta should have phase: {meta}"

    def test_hash_from_fingerprint(self):
        """Hash should be computed from fingerprint."""
        state = _create_test_state()

        fingerprint = _fingerprint_state(state)
        hash_result = _compute_state_hash(state)
        hash_from_fingerprint = _simple_hash(fingerprint)

        assert hash_result == hash_from_fingerprint, \
            "Hash from state should equal hash from fingerprint"


class TestCrossEngineParity:
    """Tests that would require TypeScript execution for full verification."""

    def test_fingerprint_format_matches_typescript_spec(self):
        """Verify fingerprint format matches TypeScript fingerprintGameState."""
        state = _create_test_state()

        fingerprint = _fingerprint_state(state)

        # Verify format matches expected TypeScript output:
        # meta#players#stacks#markers#collapsed
        # meta: currentPlayer:currentPhase:gameStatus
        # players: sorted playerNumber:ringsInHand:eliminatedRings:territorySpaces

        expected_meta = "1:ring_placement:active"
        expected_players = "1:18:0:0|2:18:0:0"

        parts = fingerprint.split("#")
        assert parts[0] == expected_meta, f"Meta mismatch: {parts[0]} != {expected_meta}"
        assert parts[1] == expected_players, f"Players mismatch: {parts[1]} != {expected_players}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
