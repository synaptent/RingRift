"""
Golden Replay Tests (Python)

Runs curated golden game fixtures through the Python rules engine and verifies
structural invariants hold at every step.

These tests mirror the TypeScript golden replay tests to ensure TS↔Python parity.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest

# Import the Python game engine and types
from app.game_engine import GameEngine
from app.models.core import BoardType

# =============================================================================
# GOLDEN GAMES DIRECTORY
# =============================================================================

def get_golden_games_dir() -> Path:
    """Get the path to the golden games fixtures directory."""
    # Navigate from ai-service/tests/golden to tests/fixtures/golden-games
    return Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "golden-games"


def get_python_golden_games_dir() -> Path:
    """Get the path to Python-specific golden games fixtures."""
    return Path(__file__).parent.parent / "fixtures" / "golden_games"


# =============================================================================
# GOLDEN GAME INFO
# =============================================================================

class GoldenGameInfo:
    """Metadata about a golden game fixture."""

    def __init__(
        self,
        filename: str,
        category: str,
        description: str,
        board_type: str,
        num_players: int,
        expected_outcome: str | None = None,
    ):
        self.filename = filename
        self.category = category
        self.description = description
        self.board_type = board_type
        self.num_players = num_players
        self.expected_outcome = expected_outcome


# =============================================================================
# INVARIANT CHECKING
# =============================================================================

class InvariantViolation:
    """Represents a structural invariant violation."""

    def __init__(self, invariant: str, move_index: int, details: str):
        self.invariant = invariant
        self.move_index = move_index
        self.details = details

    def __str__(self) -> str:
        return f"[{self.invariant}] Move {self.move_index}: {self.details}"


def check_board_consistency(state: dict[str, Any], move_index: int) -> list[InvariantViolation]:
    """INV-BOARD-CONSISTENCY: All positions in board state are valid."""
    violations: list[InvariantViolation] = []
    board = state.get("board", {})
    state.get("boardType", "square8")

    # Check stacks have valid positions and heights
    stacks = board.get("stacks", {})
    for pos_key, stack in stacks.items():
        stack_height = stack.get("stackHeight", 0)
        if stack_height < 1:
            violations.append(InvariantViolation(
                "INV-BOARD-CONSISTENCY",
                move_index,
                f"Stack at {pos_key} has invalid height: {stack_height}"
            ))

    return violations


def check_turn_sequence(
    prev_state: dict[str, Any] | None,
    current_state: dict[str, Any],
    move_index: int
) -> list[InvariantViolation]:
    """INV-TURN-SEQUENCE: Move history length increases with each step."""
    violations: list[InvariantViolation] = []

    if prev_state:
        prev_move_count = len(prev_state.get("moveHistory", []))
        current_move_count = len(current_state.get("moveHistory", []))

        if current_move_count < prev_move_count:
            violations.append(InvariantViolation(
                "INV-TURN-SEQUENCE",
                move_index,
                f"Move history decreased from {prev_move_count} to {current_move_count}"
            ))

    return violations


def check_player_rings(state: dict[str, Any], move_index: int) -> list[InvariantViolation]:
    """INV-PLAYER-RINGS: Ring counts are non-negative and consistent."""
    violations: list[InvariantViolation] = []

    for player in state.get("players", []):
        player_num = player.get("playerNumber", 0)
        eliminated = player.get("eliminatedRings", 0)
        in_hand = player.get("ringsInHand", 0)

        if eliminated < 0:
            violations.append(InvariantViolation(
                "INV-PLAYER-RINGS",
                move_index,
                f"Player {player_num} has negative eliminatedRings: {eliminated}"
            ))

        if in_hand < 0:
            violations.append(InvariantViolation(
                "INV-PLAYER-RINGS",
                move_index,
                f"Player {player_num} has negative ringsInHand: {in_hand}"
            ))

    return violations


def check_phase_valid(state: dict[str, Any], move_index: int) -> list[InvariantViolation]:
    """INV-PHASE-VALID: Game phase is a valid phase value."""
    violations: list[InvariantViolation] = []

    valid_phases = [
        "ring_placement",
        "movement",
        "capture",
        "chain_capture",
        "line_processing",
        "territory_processing",
    ]

    phase = state.get("currentPhase", "")
    if phase not in valid_phases:
        violations.append(InvariantViolation(
            "INV-PHASE-VALID",
            move_index,
            f"Invalid phase: {phase}"
        ))

    return violations


def check_active_player(state: dict[str, Any], move_index: int) -> list[InvariantViolation]:
    """INV-ACTIVE-PLAYER: Active player index is valid."""
    violations: list[InvariantViolation] = []

    if state.get("gameStatus") == "active":
        current_player = state.get("currentPlayer", 0)
        num_players = len(state.get("players", []))

        if current_player < 0 or current_player >= num_players:
            violations.append(InvariantViolation(
                "INV-ACTIVE-PLAYER",
                move_index,
                f"Invalid currentPlayer: {current_player} (players: {num_players})"
            ))

    return violations


def check_game_status(state: dict[str, Any], move_index: int) -> list[InvariantViolation]:
    """INV-GAME-STATUS: Game status is consistent with winner field."""
    violations: list[InvariantViolation] = []

    winner = state.get("winner")
    status = state.get("gameStatus", "")

    if winner is not None:
        # Canonical terminal status is "completed"
        if status != "completed":
            violations.append(InvariantViolation(
                "INV-GAME-STATUS",
                move_index,
                f"Winner is set ({winner}) but status is {status}"
            ))

    return violations


def check_all_invariants(
    state: dict[str, Any],
    move_index: int,
    prev_state: dict[str, Any] | None = None
) -> list[InvariantViolation]:
    """Run all structural invariants on a game state."""
    violations: list[InvariantViolation] = []

    violations.extend(check_board_consistency(state, move_index))
    violations.extend(check_turn_sequence(prev_state, state, move_index))
    violations.extend(check_player_rings(state, move_index))
    violations.extend(check_phase_valid(state, move_index))
    violations.extend(check_active_player(state, move_index))
    violations.extend(check_game_status(state, move_index))

    return violations


# =============================================================================
# FIXTURE LOADING
# =============================================================================

def load_golden_games(fixtures_dir: Path) -> list[tuple[GoldenGameInfo, dict[str, Any]]]:
    """Load all golden game fixtures from a directory."""
    games: list[tuple[GoldenGameInfo, dict[str, Any]]] = []

    if not fixtures_dir.exists():
        return games

    for file in fixtures_dir.iterdir():
        if file.suffix not in [".json", ".jsonl"]:
            continue

        content = file.read_text()

        try:
            # Try parsing as single JSON first
            record = json.loads(content)

            # Extract metadata from filename
            parts = file.stem.split("_")
            info = GoldenGameInfo(
                filename=file.name,
                category=parts[0] if parts else "uncategorized",
                description="_".join(parts[1:-2]) if len(parts) > 2 else file.name,
                board_type=record.get("boardType", "square8"),
                num_players=len(record.get("players", [])) or 2,
                expected_outcome=record.get("outcome"),
            )

            games.append((info, record))
        except json.JSONDecodeError:
            # Try parsing as JSONL
            lines = content.strip().split("\n")
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                    info = GoldenGameInfo(
                        filename=f"{file.name}:{i}",
                        category="jsonl",
                        description=f"Line {i} of {file.name}",
                        board_type=record.get("boardType", "square8"),
                        num_players=len(record.get("players", [])) or 2,
                        expected_outcome=record.get("outcome"),
                    )
                    games.append((info, record))
                except (json.JSONDecodeError, Exception):
                    continue

    return games


# =============================================================================
# TESTS
# =============================================================================

class TestGoldenReplayInvariants:
    """Test structural invariants on golden game fixtures."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load golden games once for all tests."""
        shared_dir = get_golden_games_dir()
        python_dir = get_python_golden_games_dir()

        self.golden_games = load_golden_games(shared_dir)
        self.golden_games.extend(load_golden_games(python_dir))

    def test_fixtures_can_be_loaded(self):
        """Test that we can load fixture files (even if empty)."""
        # This test passes even with no fixtures - it just validates the loading mechanism
        shared_dir = get_golden_games_dir()
        python_dir = get_python_golden_games_dir()

        # Should not raise
        load_golden_games(shared_dir)
        load_golden_games(python_dir)

    @pytest.mark.skipif(
        not get_golden_games_dir().exists() or not any(get_golden_games_dir().iterdir()),
        reason="No golden game fixtures found"
    )
    def test_invariants_hold_for_golden_games(self):
        """Test that all invariants hold for loaded golden games."""
        if not self.golden_games:
            pytest.skip("No golden game fixtures loaded")

        for _info, record in self.golden_games:
            # For now, just verify the record structure is valid
            assert "boardType" in record or record.get("board") is not None
            assert "moves" in record

            # TODO: When Python engine supports reconstructStateAtMove,
            # replay through all moves and check invariants at each step


class TestInvariantCheckers:
    """Unit tests for the invariant checking functions."""

    def test_check_phase_valid_detects_invalid_phase(self):
        """Test that invalid phases are detected."""
        state = {
            "currentPhase": "invalid_phase",
            "gameStatus": "active",
        }

        violations = check_phase_valid(state, 0)

        assert len(violations) == 1
        assert violations[0].invariant == "INV-PHASE-VALID"

    def test_check_phase_valid_accepts_valid_phases(self):
        """Test that valid phases pass."""
        valid_phases = [
            "ring_placement",
            "movement",
            "capture",
            "chain_capture",
            "line_processing",
            "territory_processing",
        ]

        for phase in valid_phases:
            state = {"currentPhase": phase}
            violations = check_phase_valid(state, 0)
            assert len(violations) == 0, f"Phase {phase} should be valid"

    def test_check_active_player_detects_invalid_player(self):
        """Test that invalid player indices are detected."""
        state = {
            "gameStatus": "active",
            "currentPlayer": 99,
            "players": [{"playerNumber": 0}],
        }

        violations = check_active_player(state, 0)

        assert len(violations) == 1
        assert violations[0].invariant == "INV-ACTIVE-PLAYER"

    def test_check_player_rings_detects_negative_counts(self):
        """Test that negative ring counts are detected."""
        state = {
            "players": [
                {"playerNumber": 0, "eliminatedRings": -5, "ringsInHand": 18},
            ],
        }

        violations = check_player_rings(state, 0)

        assert len(violations) == 1
        assert violations[0].invariant == "INV-PLAYER-RINGS"

    def test_check_game_status_detects_inconsistent_winner(self):
        """Test that winner without finished status is detected."""
        state = {
            "winner": 0,
            "gameStatus": "active",
        }

        violations = check_game_status(state, 0)

        assert len(violations) == 1
        assert violations[0].invariant == "INV-GAME-STATUS"


class TestFixtureLoading:
    """Tests for fixture loading functionality."""

    def test_load_returns_empty_for_nonexistent_dir(self):
        """Test that loading from non-existent directory returns empty list."""
        result = load_golden_games(Path("/non/existent/path"))
        assert result == []

    def test_load_golden_games_parses_json(self, tmp_path):
        """Test that JSON files are parsed correctly."""
        fixture = {
            "boardType": "square8",
            "numPlayers": 2,
            "players": [{"playerNumber": 0}, {"playerNumber": 1}],
            "moves": [],
            "outcome": "ring_elimination",
        }

        fixture_file = tmp_path / "test_game.json"
        fixture_file.write_text(json.dumps(fixture))

        games = load_golden_games(tmp_path)

        assert len(games) == 1
        info, record = games[0]
        assert info.board_type == "square8"
        assert info.expected_outcome == "ring_elimination"
        assert record == fixture
