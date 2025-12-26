"""Tests for app.routes.training - Training Status API Routes.

This module tests the FastAPI training endpoints including:
- GET /api/training/{config_key}/velocity - Elo velocity and trend
- GET /api/training/{config_key}/momentum - Full momentum status
- GET /api/training/status - Aggregate training status
- GET /api/training/feedback - FeedbackAccelerator status
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.routes.training import (
    EloVelocityResponse,
    FeedbackStatusResponse,
    MomentumResponse,
    TrainingStatusResponse,
    router,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_momentum():
    """Create a mock ConfigMomentum object."""
    momentum = MagicMock()
    momentum.current_elo = 1650.0
    momentum.momentum_state.value = "improving"
    momentum.intensity.value = "high"
    momentum.games_since_training = 250
    momentum.consecutive_improvements = 3
    momentum.consecutive_plateaus = 0
    momentum.get_elo_trend.return_value = 2.5
    momentum.get_improvement_rate.return_value = 15.0
    momentum.elo_history = [
        MagicMock(games_played=100),
        MagicMock(games_played=350),
    ]
    return momentum


@pytest.fixture
def mock_accelerator(mock_momentum):
    """Create a mock FeedbackAccelerator."""
    accelerator = MagicMock()
    accelerator.get_config_momentum.return_value = mock_momentum
    accelerator.get_status_summary.return_value = {
        "total_configs": 4,
        "improving_configs": 2,
        "plateau_configs": 1,
        "total_improvement_rate_per_hour": 45.0,
        "configs": {
            "hex8_2p": {"elo": 1650.0, "state": "improving"},
            "square8_2p": {"elo": 1520.0, "state": "plateau"},
        },
    }
    accelerator.get_improving_configs.return_value = ["hex8_2p", "square8_4p"]
    accelerator.get_plateau_configs.return_value = ["square8_2p"]
    accelerator.get_aggregate_selfplay_recommendation.return_value = {
        "aggregate_momentum": "improving",
        "recommended_multiplier": 1.5,
    }
    return accelerator


@pytest.fixture
def test_client(mock_accelerator):
    """Create a test client with mocked FeedbackAccelerator."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Patch the import inside the route functions
    with patch("app.training.feedback_accelerator.get_feedback_accelerator", return_value=mock_accelerator):
        yield TestClient(app)


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestEloVelocityResponse:
    """Tests for EloVelocityResponse model."""

    def test_required_fields(self):
        """Should accept required fields."""
        response = EloVelocityResponse(
            config_key="hex8_2p",
            elo_per_hour=15.0,
            trend="improving",
        )
        assert response.config_key == "hex8_2p"
        assert response.elo_per_hour == 15.0
        assert response.trend == "improving"

    def test_default_fields(self):
        """Should have correct default values."""
        response = EloVelocityResponse(
            config_key="hex8_2p",
            elo_per_hour=15.0,
            trend="improving",
        )
        assert response.elo_per_game == 0.0
        assert response.lookback_hours == 24.0
        assert response.games_in_period == 0
        assert response.current_elo == 1500.0

    def test_optional_fields(self):
        """Should allow overriding optional fields."""
        response = EloVelocityResponse(
            config_key="hex8_2p",
            elo_per_hour=15.0,
            elo_per_game=2.5,
            trend="improving",
            lookback_hours=48.0,
            games_in_period=500,
            current_elo=1650.0,
        )
        assert response.elo_per_game == 2.5
        assert response.lookback_hours == 48.0
        assert response.games_in_period == 500
        assert response.current_elo == 1650.0


class TestMomentumResponse:
    """Tests for MomentumResponse model."""

    def test_required_fields(self):
        """Should accept required fields."""
        response = MomentumResponse(
            config_key="hex8_2p",
            momentum_state="improving",
            intensity="high",
        )
        assert response.config_key == "hex8_2p"
        assert response.momentum_state == "improving"
        assert response.intensity == "high"

    def test_default_fields(self):
        """Should have correct default values."""
        response = MomentumResponse(
            config_key="hex8_2p",
            momentum_state="improving",
            intensity="high",
        )
        assert response.current_elo == 1500.0
        assert response.games_since_training == 0
        assert response.consecutive_improvements == 0
        assert response.consecutive_plateaus == 0
        assert response.elo_trend == 0.0
        assert response.improvement_rate_per_hour == 0.0


class TestTrainingStatusResponse:
    """Tests for TrainingStatusResponse model."""

    def test_structure(self):
        """Should have correct structure."""
        response = TrainingStatusResponse(
            total_configs=4,
            improving_configs=2,
            plateau_configs=1,
            total_improvement_rate_per_hour=45.0,
            configs={
                "hex8_2p": {"elo": 1650.0},
                "square8_2p": {"elo": 1520.0},
            },
        )
        assert response.total_configs == 4
        assert response.improving_configs == 2
        assert response.plateau_configs == 1
        assert response.total_improvement_rate_per_hour == 45.0
        assert len(response.configs) == 2

    def test_default_values(self):
        """Should have correct defaults."""
        response = TrainingStatusResponse()
        assert response.total_configs == 0
        assert response.improving_configs == 0
        assert response.plateau_configs == 0
        assert response.total_improvement_rate_per_hour == 0.0
        assert response.configs == {}


class TestFeedbackStatusResponse:
    """Tests for FeedbackStatusResponse model."""

    def test_structure(self):
        """Should have correct structure."""
        response = FeedbackStatusResponse(
            improving=["hex8_2p", "square8_4p"],
            plateau=["square8_2p"],
            aggregate_momentum="improving",
            recommended_multiplier=1.5,
        )
        assert len(response.improving) == 2
        assert len(response.plateau) == 1
        assert response.aggregate_momentum == "improving"
        assert response.recommended_multiplier == 1.5

    def test_default_values(self):
        """Should have correct defaults."""
        response = FeedbackStatusResponse()
        assert response.improving == []
        assert response.plateau == []
        assert response.aggregate_momentum == "unknown"
        assert response.recommended_multiplier == 1.0


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestGetEloVelocityEndpoint:
    """Tests for GET /api/training/{config_key}/velocity endpoint."""

    def test_get_velocity_with_data(self, test_client, mock_accelerator, mock_momentum):
        """Should return velocity when data exists."""
        mock_accelerator.get_config_momentum.return_value = mock_momentum
        mock_momentum.get_improvement_rate.return_value = 15.0
        mock_momentum.get_elo_trend.return_value = 2.5
        mock_momentum.current_elo = 1650.0

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["config_key"] == "hex8_2p"
        assert data["elo_per_hour"] == 15.0
        assert data["elo_per_game"] == 2.5
        assert data["trend"] == "improving"
        assert data["current_elo"] == 1650.0
        assert data["games_in_period"] == 250

    def test_get_velocity_no_data(self, test_client, mock_accelerator):
        """Should return zero values when no momentum data."""
        mock_accelerator.get_config_momentum.return_value = None

        response = test_client.get("/training/nonexistent/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["config_key"] == "nonexistent"
        assert data["elo_per_hour"] == 0.0
        assert data["trend"] == "unknown"
        assert data["current_elo"] == 1500.0

    def test_get_velocity_improving_trend(self, test_client, mock_accelerator, mock_momentum):
        """Should detect improving trend."""
        mock_momentum.get_improvement_rate.return_value = 20.0

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["trend"] == "improving"

    def test_get_velocity_declining_trend(self, test_client, mock_accelerator, mock_momentum):
        """Should detect declining trend."""
        mock_momentum.get_improvement_rate.return_value = -10.0

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["trend"] == "declining"

    def test_get_velocity_stable_trend(self, test_client, mock_accelerator, mock_momentum):
        """Should detect stable trend."""
        mock_momentum.get_improvement_rate.return_value = 5.0

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["trend"] == "stable"

    def test_get_velocity_custom_lookback(self, test_client, mock_accelerator, mock_momentum):
        """Should accept custom lookback hours."""
        response = test_client.get("/training/hex8_2p/velocity?lookback_hours=48")
        assert response.status_code == 200

        data = response.json()
        assert data["lookback_hours"] == 48.0

    def test_get_velocity_accelerator_not_available(self, test_client):
        """Should return 503 when accelerator not available."""
        with patch(
            "app.training.feedback_accelerator.get_feedback_accelerator",
            side_effect=ImportError("Module not found"),
        ):
            app = test_client.app
            response = test_client.get("/training/hex8_2p/velocity")
            assert response.status_code == 503
            assert "FeedbackAccelerator not available" in response.json()["detail"]

    def test_get_velocity_unexpected_error(self, test_client, mock_accelerator):
        """Should return 500 on unexpected error."""
        mock_accelerator.get_config_momentum.side_effect = RuntimeError("Database error")

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_get_velocity_empty_elo_history(self, test_client, mock_accelerator, mock_momentum):
        """Should handle empty elo history."""
        mock_momentum.elo_history = []

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["games_in_period"] == 0

    def test_get_velocity_single_elo_entry(self, test_client, mock_accelerator, mock_momentum):
        """Should handle single elo history entry."""
        mock_momentum.elo_history = [MagicMock(games_played=100)]

        response = test_client.get("/training/hex8_2p/velocity")
        assert response.status_code == 200

        data = response.json()
        assert data["games_in_period"] == 0


class TestGetMomentumEndpoint:
    """Tests for GET /api/training/{config_key}/momentum endpoint."""

    def test_get_momentum_with_data(self, test_client, mock_accelerator, mock_momentum):
        """Should return full momentum when data exists."""
        response = test_client.get("/training/hex8_2p/momentum")
        assert response.status_code == 200

        data = response.json()
        assert data["config_key"] == "hex8_2p"
        assert data["current_elo"] == 1650.0
        assert data["momentum_state"] == "improving"
        assert data["intensity"] == "high"
        assert data["games_since_training"] == 250
        assert data["consecutive_improvements"] == 3
        assert data["consecutive_plateaus"] == 0
        assert data["elo_trend"] == 2.5
        assert data["improvement_rate_per_hour"] == 15.0

    def test_get_momentum_no_data(self, test_client, mock_accelerator):
        """Should return 404 when no momentum data."""
        mock_accelerator.get_config_momentum.return_value = None

        response = test_client.get("/training/nonexistent/momentum")
        assert response.status_code == 404
        assert "No momentum data" in response.json()["detail"]

    def test_get_momentum_accelerator_not_available(self, test_client):
        """Should return 503 when accelerator not available."""
        with patch(
            "app.training.feedback_accelerator.get_feedback_accelerator",
            side_effect=ImportError("Module not found"),
        ):
            app = test_client.app
            response = test_client.get("/training/hex8_2p/momentum")
            assert response.status_code == 503

    def test_get_momentum_unexpected_error(self, test_client, mock_accelerator, mock_momentum):
        """Should return 500 on unexpected error."""
        mock_momentum.get_elo_trend.side_effect = RuntimeError("Calculation error")

        response = test_client.get("/training/hex8_2p/momentum")
        assert response.status_code == 500


class TestGetTrainingStatusEndpoint:
    """Tests for GET /api/training/status endpoint."""

    def test_get_status_with_data(self, test_client, mock_accelerator):
        """Should return aggregate status."""
        response = test_client.get("/training/status")
        assert response.status_code == 200

        data = response.json()
        assert data["total_configs"] == 4
        assert data["improving_configs"] == 2
        assert data["plateau_configs"] == 1
        assert data["total_improvement_rate_per_hour"] == 45.0
        assert len(data["configs"]) == 2

    def test_get_status_empty(self, test_client, mock_accelerator):
        """Should handle empty status."""
        mock_accelerator.get_status_summary.return_value = {}

        response = test_client.get("/training/status")
        assert response.status_code == 200

        data = response.json()
        assert data["total_configs"] == 0
        assert data["improving_configs"] == 0
        assert data["plateau_configs"] == 0
        assert data["total_improvement_rate_per_hour"] == 0.0
        assert data["configs"] == {}

    def test_get_status_accelerator_not_available(self, test_client):
        """Should return 503 when accelerator not available."""
        with patch(
            "app.training.feedback_accelerator.get_feedback_accelerator",
            side_effect=ImportError("Module not found"),
        ):
            app = test_client.app
            response = test_client.get("/training/status")
            assert response.status_code == 503

    def test_get_status_unexpected_error(self, test_client, mock_accelerator):
        """Should return 500 on unexpected error."""
        mock_accelerator.get_status_summary.side_effect = RuntimeError("Database error")

        response = test_client.get("/training/status")
        assert response.status_code == 500


class TestGetFeedbackStatusEndpoint:
    """Tests for GET /api/training/feedback endpoint."""

    def test_get_feedback_with_data(self, test_client, mock_accelerator):
        """Should return feedback status."""
        response = test_client.get("/training/feedback")
        assert response.status_code == 200

        data = response.json()
        assert len(data["improving"]) == 2
        assert "hex8_2p" in data["improving"]
        assert "square8_4p" in data["improving"]
        assert len(data["plateau"]) == 1
        assert "square8_2p" in data["plateau"]
        assert data["aggregate_momentum"] == "improving"
        assert data["recommended_multiplier"] == 1.5

    def test_get_feedback_empty(self, test_client, mock_accelerator):
        """Should handle empty feedback."""
        mock_accelerator.get_improving_configs.return_value = []
        mock_accelerator.get_plateau_configs.return_value = []
        mock_accelerator.get_aggregate_selfplay_recommendation.return_value = {}

        response = test_client.get("/training/feedback")
        assert response.status_code == 200

        data = response.json()
        assert data["improving"] == []
        assert data["plateau"] == []
        assert data["aggregate_momentum"] == "unknown"
        assert data["recommended_multiplier"] == 1.0

    def test_get_feedback_accelerator_not_available(self, test_client):
        """Should return 503 when accelerator not available."""
        with patch(
            "app.training.feedback_accelerator.get_feedback_accelerator",
            side_effect=ImportError("Module not found"),
        ):
            app = test_client.app
            response = test_client.get("/training/feedback")
            assert response.status_code == 503

    def test_get_feedback_unexpected_error(self, test_client, mock_accelerator):
        """Should return 500 on unexpected error."""
        mock_accelerator.get_improving_configs.side_effect = RuntimeError("Database error")

        response = test_client.get("/training/feedback")
        assert response.status_code == 500

    def test_get_feedback_partial_data(self, test_client, mock_accelerator):
        """Should handle partial recommendation data."""
        mock_accelerator.get_aggregate_selfplay_recommendation.return_value = {
            "aggregate_momentum": "plateau"
            # Missing recommended_multiplier
        }

        response = test_client.get("/training/feedback")
        assert response.status_code == 200

        data = response.json()
        assert data["aggregate_momentum"] == "plateau"
        assert data["recommended_multiplier"] == 1.0  # Default
