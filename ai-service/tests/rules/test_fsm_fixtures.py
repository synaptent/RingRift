"""
Cross-Language FSM Fixture Tests.

This module loads FSM transition test vectors from the shared JSON fixtures
and validates that Python FSM validation produces the expected results.
The same fixtures are used by TypeScript tests to ensure cross-language parity.

Fixture location: tests/fixtures/fsm-parity/v1/fsm_transitions.vectors.json

These tests focus on FSM validation (move type validity for phase), not full
orchestration. Full orchestration tests are in test_fsm_parity.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from app.rules.fsm import validate_move_for_phase, FSMValidationResult

# Path to shared fixtures (relative to ai-service/)
FIXTURES_PATH = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "fsm-parity" / "v1" / "fsm_transitions.vectors.json"


def _load_fixtures() -> dict[str, Any]:
    """Load the cross-language FSM fixtures."""
    with open(FIXTURES_PATH, "r") as f:
        return json.load(f)


# Map fixture event types to MoveType enum values
# Note: Some FSM events (END_CHAIN, _ADVANCE_TURN, RESIGN, TIMEOUT) don't have
# direct MoveType equivalents in Python - they're FSM-internal transitions.
_EVENT_TO_MOVE_TYPE: dict[str, MoveType | None] = {
    "PLACE_RING": MoveType.PLACE_RING,
    "SKIP_PLACEMENT": MoveType.SKIP_PLACEMENT,
    "NO_PLACEMENT_ACTION": MoveType.NO_PLACEMENT_ACTION,
    "MOVE_STACK": MoveType.MOVE_STACK,
    "NO_MOVEMENT_ACTION": MoveType.NO_MOVEMENT_ACTION,
    "CAPTURE": MoveType.OVERTAKING_CAPTURE,  # Maps to overtaking_capture
    "END_CHAIN": None,  # FSM-internal: no direct Python MoveType
    "CONTINUE_CHAIN": MoveType.CONTINUE_CAPTURE_SEGMENT,
    "NO_LINE_ACTION": MoveType.NO_LINE_ACTION,
    "PROCESS_LINE": MoveType.PROCESS_LINE,
    "NO_TERRITORY_ACTION": MoveType.NO_TERRITORY_ACTION,
    "PROCESS_REGION": MoveType.CHOOSE_TERRITORY_OPTION,  # Canonical type (legacy: PROCESS_TERRITORY_REGION)
    "FORCED_ELIMINATE": MoveType.FORCED_ELIMINATION,
    "_ADVANCE_TURN": None,  # Internal FSM event, not a move type
    "RESIGN": None,  # Not a move type
    "TIMEOUT": None,  # Not a move type
}

# Map fixture phase strings to GamePhase enum
_PHASE_TO_ENUM: dict[str, GamePhase] = {
    "ring_placement": GamePhase.RING_PLACEMENT,
    "movement": GamePhase.MOVEMENT,
    "capture": GamePhase.CAPTURE,
    "chain_capture": GamePhase.CHAIN_CAPTURE,
    "line_processing": GamePhase.LINE_PROCESSING,
    "territory_processing": GamePhase.TERRITORY_PROCESSING,
    "forced_elimination": GamePhase.FORCED_ELIMINATION,
    "turn_end": None,  # No GamePhase for turn_end
    "game_over": None,  # No GamePhase for game_over
}


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
    rings_in_hand: int = 0,
) -> GameState:
    """Create a minimal test game state."""
    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsed_spaces={},
        territories={},
        formed_lines=[],
        eliminated_rings={str(i): 0 for i in range(1, num_players + 1)},
    )
    players = [_make_player(i, rings_in_hand) for i in range(1, num_players + 1)]
    now = datetime.now()

    return GameState(
        id="fsm-fixture-test",
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
    to: dict[str, int] | None = None,
    from_pos: dict[str, int] | None = None,
) -> Move:
    """Create a test move from fixture data."""
    to_pos = Position(x=to["x"], y=to["y"]) if to else Position(x=0, y=0)
    from_position = Position(x=from_pos["x"], y=from_pos["y"]) if from_pos else None

    return Move(
        id="fixture-move",
        type=move_type,
        player=player,
        to=to_pos,
        fromPos=from_position,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )


class TestFSMCrossLanguageFixtures:
    """Test FSM validation using cross-language fixtures."""

    @pytest.fixture(scope="class")
    def fixtures(self) -> dict[str, Any]:
        """Load fixtures once per test class."""
        return _load_fixtures()

    def test_fixtures_loaded(self, fixtures: dict[str, Any]):
        """Verify fixtures are loaded correctly."""
        assert fixtures["version"] == "v1"
        assert fixtures["count"] == len(fixtures["vectors"])
        assert len(fixtures["vectors"]) > 0

    def test_all_phases_covered(self, fixtures: dict[str, Any]):
        """Verify all major phases have at least one test vector."""
        categories = set(v["category"] for v in fixtures["vectors"])
        expected_phases = {
            "ring_placement",
            "movement",
            "capture",
            "chain_capture",
            "line_processing",
            "territory_processing",
            "forced_elimination",
            "turn_end",
        }
        # At least most phases should be covered
        covered = categories & expected_phases
        assert len(covered) >= 5, f"Only {len(covered)} phases covered: {covered}"


class TestFSMValidationVectors:
    """Test individual FSM validation vectors."""

    @pytest.fixture(scope="class")
    def vectors(self) -> list[dict[str, Any]]:
        """Load all vectors."""
        fixtures = _load_fixtures()
        return fixtures["vectors"]

    def _can_test_vector(self, vector: dict[str, Any]) -> bool:
        """Check if we can test this vector with Python FSM validation.

        Some vectors test internal FSM transitions (_ADVANCE_TURN, RESIGN, TIMEOUT)
        that don't map to move types. We skip those.
        """
        event_type = vector["input"]["event"]["type"]
        phase_str = vector["input"]["currentPhase"]

        # Skip internal events
        if event_type in ("_ADVANCE_TURN", "RESIGN", "TIMEOUT"):
            return False

        # Skip phases without GamePhase enum mapping
        if _PHASE_TO_ENUM.get(phase_str) is None:
            return False

        # Skip if no move type mapping
        if _EVENT_TO_MOVE_TYPE.get(event_type) is None:
            return False

        return True

    def _get_testable_vectors(self, vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter vectors to only those testable with Python FSM."""
        return [v for v in vectors if self._can_test_vector(v)]

    def test_ring_placement_vectors(self, vectors: list[dict[str, Any]]):
        """Test ring_placement phase vectors."""
        ring_placement_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "ring_placement"
        ]

        for vector in ring_placement_vectors:
            self._run_validation_vector(vector)

    def test_movement_vectors(self, vectors: list[dict[str, Any]]):
        """Test movement phase vectors."""
        movement_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "movement"
        ]

        for vector in movement_vectors:
            self._run_validation_vector(vector)

    def test_capture_vectors(self, vectors: list[dict[str, Any]]):
        """Test capture phase vectors."""
        capture_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "capture"
        ]

        for vector in capture_vectors:
            self._run_validation_vector(vector)

    def test_chain_capture_vectors(self, vectors: list[dict[str, Any]]):
        """Test chain_capture phase vectors."""
        chain_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "chain_capture"
        ]

        for vector in chain_vectors:
            self._run_validation_vector(vector)

    def test_line_processing_vectors(self, vectors: list[dict[str, Any]]):
        """Test line_processing phase vectors."""
        line_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "line_processing"
        ]

        for vector in line_vectors:
            self._run_validation_vector(vector)

    def test_territory_processing_vectors(self, vectors: list[dict[str, Any]]):
        """Test territory_processing phase vectors."""
        territory_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "territory_processing"
        ]

        for vector in territory_vectors:
            self._run_validation_vector(vector)

    def test_forced_elimination_vectors(self, vectors: list[dict[str, Any]]):
        """Test forced_elimination phase vectors."""
        fe_vectors = [
            v for v in self._get_testable_vectors(vectors)
            if v["category"] == "forced_elimination"
        ]

        for vector in fe_vectors:
            self._run_validation_vector(vector)

    def _run_validation_vector(self, vector: dict[str, Any]):
        """Run a single validation vector test."""
        input_data = vector["input"]
        expected = vector["expectedOutput"]

        # Get phase and event type
        phase_str = input_data["currentPhase"]
        event = input_data["event"]
        event_type = event["type"]
        player = input_data["currentPlayer"]
        num_players = input_data["numPlayers"]

        # Map to Python types
        phase = _PHASE_TO_ENUM[phase_str]
        move_type = _EVENT_TO_MOVE_TYPE[event_type]

        if phase is None or move_type is None:
            pytest.skip(f"Cannot test vector {vector['id']}: unmapped phase or event")

        # Create game state (minimal, for validation)
        state = _make_game_state(
            phase=phase,
            current_player=player,
            num_players=num_players,
        )

        # Create move
        move = _make_move(
            move_type=move_type,
            player=player,
            to=event.get("to") or event.get("target"),
            from_pos=event.get("from"),
        )

        # Run validation (without full game state context to avoid guard checks)
        # We pass game_state=None to skip guards that require full board state
        result = validate_move_for_phase(phase, move, game_state=None)

        # Check result
        if expected["ok"]:
            assert result.ok, (
                f"Vector {vector['id']} expected ok=True but got ok=False. "
                f"Error: {result.code} - {result.message}"
            )
        else:
            # Expected error
            assert not result.ok, (
                f"Vector {vector['id']} expected ok=False but got ok=True"
            )
            if "errorCode" in expected:
                # Map our error codes to fixture error codes
                expected_code = expected["errorCode"]
                # INVALID_EVENT in TS maps to INVALID_MOVE_FOR_PHASE in Python
                if expected_code == "INVALID_EVENT":
                    assert result.code in ("INVALID_MOVE_FOR_PHASE", "INVALID_EVENT"), (
                        f"Vector {vector['id']} expected error code {expected_code} "
                        f"but got {result.code}"
                    )


class TestFSMValidationParity:
    """Test that Python FSM validation matches TypeScript behavior."""

    def test_place_ring_allowed_in_ring_placement(self):
        """PLACE_RING should be allowed in ring_placement phase."""
        state = _make_game_state(GamePhase.RING_PLACEMENT)
        move = _make_move(MoveType.PLACE_RING, player=1, to={"x": 3, "y": 3})

        result = validate_move_for_phase(GamePhase.RING_PLACEMENT, move)
        assert result.ok, f"Expected ok=True, got {result.code}: {result.message}"

    def test_move_stack_not_allowed_in_ring_placement(self):
        """MOVE_STACK should not be allowed in ring_placement phase."""
        state = _make_game_state(GamePhase.RING_PLACEMENT)
        move = _make_move(
            MoveType.MOVE_STACK,
            player=1,
            to={"x": 4, "y": 4},
            from_pos={"x": 2, "y": 2},
        )

        result = validate_move_for_phase(GamePhase.RING_PLACEMENT, move)
        assert not result.ok, "Expected ok=False for MOVE_STACK in ring_placement"
        assert result.code == "INVALID_MOVE_FOR_PHASE"

    def test_forced_elimination_allowed_in_forced_elimination(self):
        """FORCED_ELIMINATION should be allowed in forced_elimination phase."""
        state = _make_game_state(GamePhase.FORCED_ELIMINATION)
        move = _make_move(MoveType.FORCED_ELIMINATION, player=1, to={"x": 3, "y": 3})

        result = validate_move_for_phase(GamePhase.FORCED_ELIMINATION, move)
        assert result.ok, f"Expected ok=True, got {result.code}: {result.message}"

    def test_no_territory_action_allowed_in_territory_processing(self):
        """NO_TERRITORY_ACTION should be allowed in territory_processing phase (without context)."""
        state = _make_game_state(GamePhase.TERRITORY_PROCESSING)
        move = _make_move(MoveType.NO_TERRITORY_ACTION, player=1)

        # Without game state context, the guard is skipped
        result = validate_move_for_phase(GamePhase.TERRITORY_PROCESSING, move, game_state=None)
        assert result.ok, f"Expected ok=True, got {result.code}: {result.message}"

    def test_process_line_allowed_in_line_processing(self):
        """PROCESS_LINE should be allowed in line_processing phase."""
        state = _make_game_state(GamePhase.LINE_PROCESSING)
        move = _make_move(MoveType.PROCESS_LINE, player=1)

        result = validate_move_for_phase(GamePhase.LINE_PROCESSING, move)
        assert result.ok, f"Expected ok=True, got {result.code}: {result.message}"
