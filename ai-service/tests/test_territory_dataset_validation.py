from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import pytest

# Ensure app package is importable when running tests directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.models import (  # type: ignore  # noqa: E402
    BoardState,
    BoardType,
    GameState,
    Player,
    TimeControl,
)
from app.training.territory_dataset_validation import (  # type: ignore  # noqa: E402
    iter_territory_dataset_errors,
    validate_territory_example,
)


def _make_minimal_game_state(board_type: BoardType = BoardType.SQUARE8) -> dict[str, object]:
    """Construct a minimal, JSON-serialisable GameState payload."""
    board = BoardState(type=board_type, size=8 if board_type == BoardType.SQUARE8 else 19)
    players = [
        Player(
            id="p1",
            username="Player1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=None,
        ),
        Player(
            id="p2",
            username="Player2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=None,
        ),
    ]

    state = GameState(
        id="g1",
        boardType=board_type,
        board=board,
        players=players,
        currentPlayer=1,
        currentPhase="ring_placement",
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus="active",
        createdAt=datetime.now(timezone.utc),
        lastMoveAt=datetime.now(timezone.utc),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
    )
    # Use the same JSON mode as the dataset generator.
    return state.model_dump(by_alias=True, mode="json")  # type: ignore[attr-defined]


def test_validate_territory_example_accepts_minimal_valid_example() -> None:
    game_state = _make_minimal_game_state()
    example = {
        "game_state": game_state,
        "player_number": 1,
        "target": 0.0,
        "time_weight": 1.0,
        "engine_mode": "mixed",
        "num_players": 2,
        "ai_type_p1": "descent",
        "ai_difficulty_p1": 5,
        "ai_type_p2": "descent",
        "ai_difficulty_p2": 5,
    }

    errors = validate_territory_example(example)
    assert errors == []


@pytest.mark.parametrize(
    "field",
    ["game_state", "player_number", "target", "time_weight", "engine_mode", "num_players"],
)
def test_validate_territory_example_flags_missing_required_fields(field: str) -> None:
    game_state = _make_minimal_game_state()
    example = {
        "game_state": game_state,
        "player_number": 1,
        "target": 0.0,
        "time_weight": 1.0,
        "engine_mode": "mixed",
        "num_players": 2,
        "ai_type_p1": "descent",
        "ai_difficulty_p1": 5,
        "ai_type_p2": "descent",
        "ai_difficulty_p2": 5,
    }
    example.pop(field)

    errors = validate_territory_example(example)
    assert any(field in msg for msg in errors)


def test_iter_territory_dataset_errors_reports_line_numbers(tmp_path) -> None:
    game_state = _make_minimal_game_state()

    valid_example = {
        "game_state": game_state,
        "player_number": 1,
        "target": 0.0,
        "time_weight": 0.5,
        "engine_mode": "mixed",
        "num_players": 2,
        "ai_type_p1": "descent",
        "ai_difficulty_p1": 5,
        "ai_type_p2": "descent",
        "ai_difficulty_p2": 5,
    }

    invalid_example = {
        # Missing game_state and engine_mode; invalid time_weight and num_players.
        "player_number": 3,
        "target": "not-a-number",
        "time_weight": 1.5,
        "num_players": 1,
    }

    path = tmp_path / "sample.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(valid_example) + "\n")
        f.write(json.dumps(invalid_example) + "\n")

    with path.open("r", encoding="utf-8") as f:
        errors = iter_territory_dataset_errors(f, max_errors=10)

    # No errors on the first line.
    assert all(line_no != 1 for line_no, _ in errors)

    # Multiple errors should be reported for the second line (missing required fields).
    messages_for_second: list[str] = [msg for line_no, msg in errors if line_no == 2]
    assert messages_for_second, "Expected errors for line 2"
    # Validator only checks for missing required fields, not value validation
    assert any("game_state" in m for m in messages_for_second)
    assert any("engine_mode" in m for m in messages_for_second)

