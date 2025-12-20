"""
API Error Path Tests for AI Service

Tests edge cases and error handling for AI service endpoints:
- Invalid player_number validation
- Malformed game_state handling
- Timeout and failure scenarios
- Choice endpoint edge cases

These tests ensure robust error handling without UI component changes.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

# Ensure app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    AIType,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    TimeControl,
)


def _make_minimal_state(num_players: int = 2) -> GameState:
    """Construct a minimal GameState for testing."""
    players = [
        Player(
            id=f"p{i}",
            username=f"P{i}",
            type="human",
            playerNumber=i,
            isReady=True,
            timeRemaining=600,
            ringsInHand=20 if num_players == 2 else 18,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=None,
        )
        for i in range(1, num_players + 1)
    ]

    return GameState(
        id="error-path-test-game",
        boardType=BoardType.SQUARE8,
        board=BoardState(type=BoardType.SQUARE8, size=8),
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=18,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=0,
    )


def _make_state_with_empty_players() -> dict[str, Any]:
    """Create a game state dict with empty players list."""
    state = _make_minimal_state()
    state_dict = jsonable_encoder(state, by_alias=True)
    state_dict["players"] = []
    return state_dict


client = TestClient(app)
TEST_TIMEOUT_SECONDS = 30


class TestPlayerNumberValidation:
    """Tests for player_number validation in /ai/move endpoint."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_player_number_zero_fails_gracefully(self) -> None:
        """Player number 0 should not crash the service."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 0,
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
            "seed": 123,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_player_number_negative_fails_gracefully(self) -> None:
        """Negative player number should not crash the service."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": -1,
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
            "seed": 123,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_player_number_exceeds_player_count(self) -> None:
        """Player number larger than players list should fail gracefully."""
        state = _make_minimal_state(num_players=2)
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 5,  # Only 2 players exist
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
            "seed": 123,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_player_number_valid_boundary(self) -> None:
        """Valid player numbers at boundary should work."""
        state = _make_minimal_state(num_players=2)
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 2,  # Max valid player number
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
            "seed": 123,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 200


class TestMalformedGameState:
    """Tests for malformed game_state handling."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_empty_players_list_fails_gracefully(self) -> None:
        """Empty players list should fail gracefully, not crash."""
        state_dict = _make_state_with_empty_players()
        payload = {
            "game_state": state_dict,
            "player_number": 1,
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
            "seed": 123,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_missing_board_fails_with_validation_error(self) -> None:
        """Missing board in game_state should return validation error."""
        state = _make_minimal_state()
        state_dict = jsonable_encoder(state, by_alias=True)
        del state_dict["board"]

        payload = {
            "game_state": state_dict,
            "player_number": 1,
            "difficulty": 1,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422  # Validation error


class TestDifficultyValidation:
    """Tests for difficulty parameter validation."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_difficulty_zero_rejected(self) -> None:
        """Difficulty 0 should be rejected (min is 1)."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 1,
            "difficulty": 0,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_difficulty_eleven_rejected(self) -> None:
        """Difficulty 11 should be rejected (max is 10)."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 1,
            "difficulty": 11,
        }

        response = client.post("/ai/move", json=payload)
        assert response.status_code == 422

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_difficulty_boundaries_accepted(self) -> None:
        """Difficulty 1 and 10 should be accepted."""
        state = _make_minimal_state()

        for difficulty in (1, 10):
            payload = {
                "game_state": jsonable_encoder(state, by_alias=True),
                "player_number": 1,
                "difficulty": difficulty,
                "ai_type": AIType.RANDOM.value,
                "seed": 123,
            }

            response = client.post("/ai/move", json=payload)
            assert response.status_code == 200, f"difficulty={difficulty} failed"


class TestAIFailureHandling:
    """Tests for AI internal failure handling."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_ai_creation_failure_returns_500_with_detail(self) -> None:
        """AI creation failure should return 500 with clear detail message."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 1,
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
        }

        with patch("app.main._create_ai_instance", side_effect=RuntimeError("AI init failed")):
            response = client.post("/ai/move", json=payload)

        assert response.status_code == 500
        body = response.json()
        assert body.get("detail") == "AI init failed"

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_ai_select_move_failure_returns_500(self) -> None:
        """AI select_move failure should return 500 with detail."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 1,
            "difficulty": 1,
            "ai_type": AIType.RANDOM.value,
        }

        mock_ai = MagicMock()
        mock_ai.select_move.side_effect = ValueError("No valid moves")

        with patch("app.main._create_ai_instance", return_value=mock_ai):
            response = client.post("/ai/move", json=payload)

        assert response.status_code == 500
        body = response.json()
        assert "detail" in body


class TestEvaluateEndpointErrorPaths:
    """Tests for /ai/evaluate error handling (single-player evaluation)."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_evaluate_invalid_player_number(self) -> None:
        """Invalid player_number should fail gracefully."""
        state = _make_minimal_state()
        payload = {
            "game_state": jsonable_encoder(state, by_alias=True),
            "player_number": 99,
        }

        # Use /ai/evaluate which has player_number validation
        response = client.post("/ai/evaluate", json=payload)
        # Should fail but not crash
        assert response.status_code in (422, 500)

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_evaluate_empty_players(self) -> None:
        """Empty players list should fail gracefully."""
        state_dict = _make_state_with_empty_players()
        payload = {
            "game_state": state_dict,
            "player_number": 1,
        }

        # Use /ai/evaluate which has player_number validation
        response = client.post("/ai/evaluate", json=payload)
        assert response.status_code in (422, 500)


class TestEvaluatePositionMultiErrorPaths:
    """Tests for /ai/evaluate_position error handling (multi-player evaluation)."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_evaluate_position_empty_players(self) -> None:
        """Empty players list should fail gracefully."""
        state_dict = _make_state_with_empty_players()
        payload = {
            "game_state": state_dict,
        }

        response = client.post("/ai/evaluate_position", json=payload)
        # API returns 400 for empty players (explicit validation)
        assert response.status_code in (400, 422, 500)


class TestChoiceEndpointEdgeCases:
    """Tests for choice endpoint edge cases."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_line_reward_empty_options_rejected(self) -> None:
        """Empty options list should be rejected with validation error."""
        payload = {
            "difficulty": 5,
            "options": [],
        }

        response = client.post("/ai/choice/line_reward_option", json=payload)
        # Empty options is rejected by validation
        assert response.status_code == 422

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_line_order_empty_options_rejected(self) -> None:
        """Empty line order options should be rejected."""
        payload = {
            "difficulty": 5,
            "options": [],
        }

        response = client.post("/ai/choice/line_order", json=payload)
        assert response.status_code == 422

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_region_order_empty_options_rejected(self) -> None:
        """Empty region order options should be rejected."""
        payload = {
            "difficulty": 5,
            "options": [],
        }

        response = client.post("/ai/choice/region_order", json=payload)
        assert response.status_code == 422

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_invalid_difficulty_in_choice_rejected(self) -> None:
        """Invalid difficulty in choice endpoints should be rejected."""
        payload = {
            "difficulty": 0,  # Invalid
            "options": ["option1", "option2"],
        }

        response = client.post("/ai/choice/line_reward_option", json=payload)
        assert response.status_code == 422

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_valid_choice_request_succeeds(self) -> None:
        """Valid choice request should return selected option."""
        # Use actual enum values and include required playerNumber
        payload = {
            "playerNumber": 1,
            "difficulty": 5,
            "options": [
                "option_1_collapse_all_and_eliminate",
                "option_2_min_collapse_no_elimination",
            ],
        }

        response = client.post("/ai/choice/line_reward_option", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "selectedOption" in body
        # API prefers OPTION_2 when available
        assert body["selectedOption"] == "option_2_min_collapse_no_elimination"


class TestMultiPlayerGameStates:
    """Tests for 3-4 player game state handling."""

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_three_player_valid_player_numbers(self) -> None:
        """3-player game should accept player numbers 1-3."""
        state = _make_minimal_state(num_players=3)

        for player_num in (1, 2, 3):
            payload = {
                "game_state": jsonable_encoder(state, by_alias=True),
                "player_number": player_num,
                "difficulty": 1,
                "ai_type": AIType.RANDOM.value,
                "seed": 123,
            }

            response = client.post("/ai/move", json=payload)
            assert response.status_code == 200, f"player_number={player_num} failed"

    @pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
    def test_four_player_valid_player_numbers(self) -> None:
        """4-player game should accept player numbers 1-4."""
        state = _make_minimal_state(num_players=4)

        for player_num in (1, 4):  # Test boundaries
            payload = {
                "game_state": jsonable_encoder(state, by_alias=True),
                "player_number": player_num,
                "difficulty": 1,
                "ai_type": AIType.RANDOM.value,
                "seed": 123,
            }

            response = client.post("/ai/move", json=payload)
            assert response.status_code == 200, f"player_number={player_num} failed"
