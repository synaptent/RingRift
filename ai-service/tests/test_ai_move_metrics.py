"""AI move endpoint metrics and error-path tests.

These tests exercise the `/ai/move` FastAPI route and assert that:

* `ai_move_requests_total{ai_type,difficulty,outcome}` is incremented for both
  success and error outcomes.
* `ai_move_latency_seconds{ai_type,difficulty}` observes a new sample for the
  same label set.
* Error responses surface a clear `detail` field that matches what the
  TypeScript `AIServiceClient` expects when mapping service failures into
  structured error codes and metrics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from app.main import app  # type: ignore
from app.metrics import AI_MOVE_LATENCY, AI_MOVE_REQUESTS, observe_ai_move_start
from app.models import (  # type: ignore
    AIType,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)


def _make_minimal_state() -> GameState:
    """Construct a minimal SQUARE8 GameState suitable for /ai/move tests."""
    return GameState(
        id="ai-move-metrics-test-game",
        boardType=BoardType.SQUARE8,
        board=BoardState(type=BoardType.SQUARE8, size=8),
        players=[
            Player(
                id="p1",
                username="P1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p2",
                username="P2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=20,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ],
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,  # RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
    )


def _get_counter_value(counter, labels: Dict[str, str]) -> float:
    """Return the current value for a Counter with the given label set.

    Prometheus client libraries may expose Counter samples either under the
    bare metric name (for example ``ai_move_requests_total``) or with an
    additional ``_total`` suffix depending on the version and how the metric
    was originally registered. To keep this helper robust across versions, we
    accept both forms.
    """
    metric_name = counter._name  # type: ignore[attr-defined]
    candidate_names = {metric_name, f"{metric_name}_total"}

    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name not in candidate_names:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


def _get_histogram_count(histogram, labels: Dict[str, str]) -> float:
    """Return the current count for a Histogram with the given label set."""
    metric_name = f"{histogram._name}_count"  # type: ignore[attr-defined]
    for metric in histogram.collect():
        for sample in metric.samples:
            if sample.name != metric_name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


TEST_TIMEOUT_SECONDS = 30
client = TestClient(app)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_ai_move_success_increments_metrics() -> None:
    """Successful /ai/move calls increment requests and latency metrics."""
    state = _make_minimal_state()
    payload: Dict[str, Any] = {
        "game_state": jsonable_encoder(state, by_alias=True),
        "player_number": 1,
        "difficulty": 1,
        # Force a cheap RandomAI path and avoid heavier search engines.
        "ai_type": AIType.RANDOM.value,
        "seed": 123,
    }

    labels = {"ai_type": "random", "difficulty": "1"}

    before_requests = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "success"})
    before_errors = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "error"})
    before_latency_count = _get_histogram_count(AI_MOVE_LATENCY, labels)

    response = client.post("/ai/move", json=payload)
    assert response.status_code == 200, f"status={response.status_code}, body={response.text}"
    body = response.json()
    assert "move" in body
    assert "evaluation" in body
    assert body["ai_type"] == "random"
    assert body["difficulty"] == 1

    after_requests = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "success"})
    after_errors = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "error"})
    after_latency_count = _get_histogram_count(AI_MOVE_LATENCY, labels)

    # One successful request, no errors, one new latency observation.
    assert after_requests == before_requests + 1
    assert after_errors == before_errors
    assert after_latency_count == before_latency_count + 1


def test_observe_ai_move_start_normalises_labels() -> None:
    """observe_ai_move_start returns stringified difficulty labels."""
    ai_type_label, difficulty_label = observe_ai_move_start("mcts", 6)

    assert ai_type_label == "mcts"
    assert difficulty_label == "6"


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_ai_move_error_increments_error_metrics_and_surfaces_detail() -> None:
    """Failing /ai/move calls increment error metrics and return clear detail."""
    state = _make_minimal_state()
    payload: Dict[str, Any] = {
        "game_state": jsonable_encoder(state, by_alias=True),
        "player_number": 1,
        "difficulty": 1,
        "ai_type": AIType.RANDOM.value,
        "seed": 123,
    }

    labels = {"ai_type": "random", "difficulty": "1"}

    before_success = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "success"})
    before_error = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "error"})
    before_latency_count = _get_histogram_count(AI_MOVE_LATENCY, labels)

    # Induce a controlled failure before the AI is created so that the
    # service returns a 500 with a simple string detail. This matches what
    # the TS AIServiceClient expects when mapping service failures into its
    # own error codes.
    with patch("app.main._create_ai_instance", side_effect=RuntimeError("boom")):
        response = client.post("/ai/move", json=payload)

    assert response.status_code == 500
    body = response.json()
    assert body.get("detail") == "boom"

    after_success = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "success"})
    after_error = _get_counter_value(AI_MOVE_REQUESTS, {**labels, "outcome": "error"})
    after_latency_count = _get_histogram_count(AI_MOVE_LATENCY, labels)

    # No successful requests recorded for this call, one error, and a
    # corresponding latency sample.
    assert after_success == before_success
    assert after_error == before_error + 1
    assert after_latency_count == before_latency_count + 1
