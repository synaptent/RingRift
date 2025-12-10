"""
Tests for GameRecord JSONL loader and adapters.

These tests exercise the offline-only helpers in
:mod:`app.training.game_record_loader` that ingest backend-exported
GameRecord JSONL files.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (  # type: ignore  # noqa: E402
    BoardType,
    MoveType,
)
from app.models.game_record import (  # type: ignore  # noqa: E402
    GameOutcome,
    GameRecord,
)
from app.training.game_record_loader import (  # type: ignore  # noqa: E402
    RecordedEpisode,
    game_record_to_recorded_episode,
    iter_game_records_from_jsonl,
    load_game_records,
)


def _make_minimal_game_record(
    game_id: str = "g1",
    winner: int | None = 1,
) -> GameRecord:
    """Construct a minimal but structurally valid GameRecord instance."""
    now = datetime.now(timezone.utc)
    # Use replace to remove timezone info, then add Z suffix for ISO format
    now_iso = now.replace(tzinfo=None).isoformat() + "Z"

    payload = {
        "id": game_id,
        "boardType": BoardType.SQUARE8.value,
        "numPlayers": 2,
        "rngSeed": 42,
        "isRated": False,
        "players": [
            {
                "playerNumber": 1,
                "username": "P1",
                "playerType": "human",
            },
            {
                "playerNumber": 2,
                "username": "P2",
                "playerType": "human",
            },
        ],
        "winner": winner,
        "outcome": GameOutcome.RING_ELIMINATION.value,
        "finalScore": {
            "ringsEliminated": {1: 1, 2: 0},
            "territorySpaces": {1: 10, 2: 5},
            "ringsRemaining": {1: 10, 2: 9},
        },
        "startedAt": now_iso,
        "endedAt": now_iso,
        "totalMoves": 1,
        "totalDurationMs": 1000,
        "moves": [
            {
                "moveNumber": 1,
                "player": 1,
                "type": MoveType.PLACE_RING.value,
                "from": None,
                "to": {"x": 0, "y": 0},
                "thinkTimeMs": 123,
            }
        ],
        "metadata": {
            "createdAt": now_iso,
            "source": "self_play",
        },
    }

    return GameRecord.model_validate(payload)


def test_iter_game_records_round_trip_jsonl(tmp_path: Path) -> None:
    """JSONL round-trip should preserve key GameRecord fields."""
    rec1 = _make_minimal_game_record("g1", winner=1)
    rec2 = _make_minimal_game_record("g2", winner=2)

    path = tmp_path / "records.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(rec1.to_jsonl_line() + "\n")
        f.write(rec2.to_jsonl_line() + "\n")

    loaded = list(iter_game_records_from_jsonl(path))

    assert len(loaded) == 2
    ids = [r.id for r in loaded]
    assert ids == ["g1", "g2"]

    assert all(r.board_type == BoardType.SQUARE8 for r in loaded)
    assert all(r.num_players == 2 for r in loaded)
    assert [r.winner for r in loaded] == [1, 2]
    assert all(r.outcome == GameOutcome.RING_ELIMINATION for r in loaded)

    # Moves and final_score should also round-trip.
    assert all(len(r.moves) == 1 for r in loaded)
    assert all(
        r.final_score.territory_spaces.get(1) == 10 for r in loaded
    )


def test_load_game_records_limit_and_blank_lines(tmp_path: Path) -> None:
    """Loader should skip blank lines and honour the optional limit."""
    rec1 = _make_minimal_game_record("g1", winner=1)
    rec2 = _make_minimal_game_record("g2", winner=2)

    path = tmp_path / "records_with_blanks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(rec1.to_jsonl_line() + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(rec2.to_jsonl_line() + "\n")

    all_records = load_game_records(path)
    assert [r.id for r in all_records] == ["g1", "g2"]

    limited = load_game_records(path, limit=1)
    assert len(limited) == 1
    assert limited[0].id == "g1"

    # A non-positive limit should return an empty list.
    assert load_game_records(path, limit=0) == []


def test_iter_game_records_raises_on_invalid_line(tmp_path: Path) -> None:
    """By default the loader should fail fast on malformed JSON."""
    rec1 = _make_minimal_game_record("g1", winner=1)
    rec2 = _make_minimal_game_record("g2", winner=2)

    path = tmp_path / "records_invalid.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(rec1.to_jsonl_line() + "\n")
        f.write("not-json\n")
        f.write(rec2.to_jsonl_line() + "\n")

    with pytest.raises(ValueError) as excinfo:
        list(iter_game_records_from_jsonl(path))

    message = str(excinfo.value)
    # Error message should mention the file path and offending line number.
    assert str(path) in message
    assert "line 2" in message


def test_iter_game_records_skip_invalid_logs_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When skip_invalid=True, loader should log and skip bad lines."""
    rec1 = _make_minimal_game_record("g1", winner=1)
    rec2 = _make_minimal_game_record("g2", winner=2)

    path = tmp_path / "records_skip_invalid.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(rec1.to_jsonl_line() + "\n")
        f.write("not-json\n")
        f.write(rec2.to_jsonl_line() + "\n")

    caplog.set_level(logging.WARNING)

    loaded = list(iter_game_records_from_jsonl(path, skip_invalid=True))
    assert [r.id for r in loaded] == ["g1", "g2"]

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert warnings, "Expected at least one warning for invalid JSON line"
    assert any(
        "Failed to parse GameRecord" in w.getMessage()
        for w in warnings
    )


def test_game_record_to_recorded_episode_basic() -> None:
    """Adapter should expose core summary fields from GameRecord."""
    record = _make_minimal_game_record("episode-game", winner=1)
    episode: RecordedEpisode = game_record_to_recorded_episode(record)

    assert episode.record is record
    assert episode.board_type == record.board_type
    assert episode.num_players == record.num_players
    assert episode.winner == record.winner
    assert episode.outcome == record.outcome
    assert episode.num_moves == record.total_moves
    assert len(episode.moves) == len(record.moves)