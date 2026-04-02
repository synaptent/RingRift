"""
Unit tests for app.main module (FastAPI application).

Tests cover:
- Health check endpoints
- AI cache management
- Request validation models
- Root endpoint
- Metrics endpoint

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_game_state():
    """Create a mock game state."""
    state = MagicMock()
    state.board.board_type.value = "square8"
    state.board.size = 8
    state.current_player = 1
    state.players = [
        MagicMock(player_number=1),
        MagicMock(player_number=2),
    ]
    state.status.value = "active"
    return state


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "RingRift AI Service"
        assert "status" in data
        assert "version" in data

    def test_health_check(self, client):
        """Health check returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check(self, client):
        """Readiness check returns ready status."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    def test_liveness_check(self, client):
        """Liveness check returns live status."""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert "live" in data

    def test_metrics_endpoint(self, client):
        """Metrics endpoint returns prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus format starts with # HELP or metric names
        content = response.text
        assert "ai_move_requests" in content or "#" in content


# =============================================================================
# AI Cache Tests
# =============================================================================


class TestAICache:
    """Tests for AI instance caching constants and behavior."""

    def test_cache_enabled_flag_exists(self):
        """AI cache enabled flag exists."""
        from app.main import AI_INSTANCE_CACHE_ENABLED
        assert isinstance(AI_INSTANCE_CACHE_ENABLED, bool)

    def test_cache_ttl_exists(self):
        """AI cache TTL constant exists."""
        from app.main import AI_INSTANCE_CACHE_TTL_SEC
        assert isinstance(AI_INSTANCE_CACHE_TTL_SEC, int)
        assert AI_INSTANCE_CACHE_TTL_SEC > 0

    def test_cache_max_exists(self):
        """AI cache max size constant exists."""
        from app.main import AI_INSTANCE_CACHE_MAX
        assert isinstance(AI_INSTANCE_CACHE_MAX, int)
        assert AI_INSTANCE_CACHE_MAX > 0


# =============================================================================
# Request/Response Model Tests
# =============================================================================


class TestRequestModels:
    """Tests for request model existence and structure."""

    def test_move_request_exists(self):
        """MoveRequest model exists."""
        from app.main import MoveRequest
        assert MoveRequest is not None

    def test_move_response_exists(self):
        """MoveResponse model exists."""
        from app.main import MoveResponse
        assert MoveResponse is not None

    def test_batch_move_request_exists(self):
        """BatchMoveRequest model exists."""
        from app.main import BatchMoveRequest
        assert BatchMoveRequest is not None

    def test_batch_move_response_exists(self):
        """BatchMoveResponse model exists."""
        from app.main import BatchMoveResponse
        assert BatchMoveResponse is not None


# =============================================================================
# Ladder Config Tests
# =============================================================================


class TestLadderConfig:
    """Tests for ladder tier configuration."""

    def test_get_ladder_tier_config_function_exists(self):
        """get_ladder_tier_config function exists."""
        from app.main import get_ladder_tier_config
        assert callable(get_ladder_tier_config)

    def test_get_ladder_tier_config_invalid(self):
        """Invalid tier raises KeyError."""
        from app.main import get_ladder_tier_config

        with pytest.raises(KeyError):
            get_ladder_tier_config("nonexistent_tier_12345", "hex8", 2)


# =============================================================================
# Seed Selection Tests
# =============================================================================


class TestSeedSelection:
    """Tests for RNG seed selection functions."""

    def test_derive_seed_function_exists(self):
        """_derive_non_ssot_seed function exists."""
        from app.main import _derive_non_ssot_seed
        assert callable(_derive_non_ssot_seed)

    def test_select_effective_seed_function_exists(self):
        """_select_effective_seed function exists."""
        from app.main import _select_effective_seed
        assert callable(_select_effective_seed)


class TestStartupMode:
    """Tests for startup-mode environment helpers."""

    def test_coordination_disabled_by_default(self):
        from app.main import _should_run_coordination_daemons

        with patch.dict("os.environ", {}, clear=True):
            assert _should_run_coordination_daemons() is False

    def test_coordination_enabled_when_orchestration_flag_is_set(self):
        from app.main import _should_run_coordination_daemons

        with patch.dict("os.environ", {"RINGRIFT_ORCHESTRATION": "true"}, clear=True):
            assert _should_run_coordination_daemons() is True

    def test_inference_only_overrides_orchestration(self):
        from app.main import _should_run_coordination_daemons

        with patch.dict(
            "os.environ",
            {
                "RINGRIFT_ORCHESTRATION": "true",
                "RINGRIFT_INFERENCE_ONLY": "true",
            },
            clear=True,
        ):
            assert _should_run_coordination_daemons() is False


# =============================================================================
# Admin Endpoint Tests
# =============================================================================


class TestAdminEndpoints:
    """Tests for admin endpoints (require API key)."""

    def test_admin_health_requires_key(self, client):
        """Admin health endpoint requires API key."""
        response = client.get("/admin/health/coordinators")
        # Should be 401 or 403 without key
        assert response.status_code in (401, 403, 422)

    def test_admin_sync_status_requires_key(self, client):
        """Admin sync status requires API key."""
        response = client.get("/admin/sync/status")
        assert response.status_code in (401, 403, 422)


# =============================================================================
# AI Type Tests
# =============================================================================


class TestAITypes:
    """Tests for AI type support."""

    def test_ai_types_imported(self):
        """AIType enum is accessible."""
        from app.main import AIType
        assert AIType is not None

    def test_heuristic_ai_imported(self):
        """HeuristicAI class is accessible."""
        from app.main import HeuristicAI
        assert HeuristicAI is not None

    def test_random_ai_imported(self):
        """RandomAI class is accessible."""
        from app.main import RandomAI
        assert RandomAI is not None

    def test_descent_ai_imported(self):
        """DescentAI class is accessible."""
        from app.main import DescentAI
        assert DescentAI is not None


# =============================================================================
# GameEngine Tests
# =============================================================================


class TestGameEngine:
    """Tests for game engine access."""

    def test_game_engine_imported(self):
        """GameEngine is accessible from main."""
        from app.main import GameEngine
        assert GameEngine is not None

    def test_board_manager_imported(self):
        """BoardManager is accessible from main."""
        from app.main import BoardManager
        assert BoardManager is not None


# =============================================================================
# Heuristic Weights Tests
# =============================================================================


class TestHeuristicWeights:
    """Tests for heuristic weight profiles."""

    def test_weight_profiles_imported(self):
        """HEURISTIC_WEIGHT_PROFILES is accessible."""
        from app.main import HEURISTIC_WEIGHT_PROFILES
        assert isinstance(HEURISTIC_WEIGHT_PROFILES, dict)
