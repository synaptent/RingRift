"""Unit tests for AutoPromotionDaemon.

December 2025: Tests for automatic model promotion based on evaluation results.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.auto_promotion_daemon import (
    AutoPromotionConfig,
    AutoPromotionDaemon,
    PromotionCandidate,
    get_auto_promotion_daemon,
    reset_auto_promotion_daemon,
)


# =============================================================================
# Test AutoPromotionConfig
# =============================================================================


class TestAutoPromotionConfig:
    """Tests for AutoPromotionConfig dataclass."""

    def test_default_values(self):
        """Default configuration has expected values."""
        config = AutoPromotionConfig()
        assert config.enabled is True
        assert config.min_games_vs_random == 50
        assert config.min_games_vs_heuristic == 50
        assert config.promotion_cooldown_seconds == 300.0
        assert config.require_both_baselines is True
        assert config.consecutive_passes_required == 1
        assert config.dry_run is False
        assert config.min_elo_improvement == 10.0
        # Quality gate settings (Dec 2025)
        assert config.quality_gate_enabled is True
        assert config.min_training_games == 100
        assert config.min_quality_score == 0.55
        assert config.require_parity_validation is True

    def test_custom_values(self):
        """Configuration accepts custom values."""
        config = AutoPromotionConfig(
            enabled=False,
            min_games_vs_random=10,
            min_games_vs_heuristic=15,
            promotion_cooldown_seconds=60.0,
            require_both_baselines=False,
            consecutive_passes_required=3,
            dry_run=True,
        )
        assert config.enabled is False
        assert config.min_games_vs_random == 10
        assert config.min_games_vs_heuristic == 15
        assert config.promotion_cooldown_seconds == 60.0
        assert config.require_both_baselines is False
        assert config.consecutive_passes_required == 3
        assert config.dry_run is True


# =============================================================================
# Test PromotionCandidate
# =============================================================================


class TestPromotionCandidate:
    """Tests for PromotionCandidate dataclass."""

    def test_default_values(self):
        """Candidate has expected defaults."""
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/path/to/model.pth",
        )
        assert candidate.config_key == "hex8_2p"
        assert candidate.model_path == "/path/to/model.pth"
        assert candidate.evaluation_results == {}
        assert candidate.evaluation_games == {}
        assert candidate.consecutive_passes == 0
        assert candidate.last_evaluation_time == 0.0
        assert candidate.last_promotion_time == 0.0

    def test_mutable_defaults_are_isolated(self):
        """Each candidate gets its own dict instances."""
        c1 = PromotionCandidate(config_key="hex8_2p", model_path="/m1.pth")
        c2 = PromotionCandidate(config_key="square8_2p", model_path="/m2.pth")

        c1.evaluation_results["RANDOM"] = 0.9
        assert "RANDOM" not in c2.evaluation_results


# =============================================================================
# Test AutoPromotionDaemon Initialization
# =============================================================================


class TestAutoPromotionDaemonInit:
    """Tests for AutoPromotionDaemon initialization."""

    def test_default_initialization(self):
        """Daemon initializes with default config."""
        daemon = AutoPromotionDaemon()
        assert daemon.config.enabled is True
        assert daemon._running is False
        assert daemon._subscribed is False
        assert daemon._candidates == {}
        assert daemon._promotion_history == []

    def test_custom_config(self):
        """Daemon accepts custom config."""
        config = AutoPromotionConfig(dry_run=True)
        daemon = AutoPromotionDaemon(config=config)
        assert daemon.config.dry_run is True


# =============================================================================
# Test Daemon Lifecycle
# =============================================================================


class TestAutoPromotionDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.fixture
    def daemon(self):
        """Create a fresh daemon for each test."""
        return AutoPromotionDaemon()

    @pytest.mark.asyncio
    async def test_start_sets_running(self, daemon):
        """Start sets running flag."""
        with patch.object(daemon, "_subscribe_to_events", new_callable=AsyncMock):
            await daemon.start()
            assert daemon._running is True

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, daemon):
        """Multiple starts don't cause issues."""
        call_count = 0

        async def mock_subscribe():
            nonlocal call_count
            call_count += 1

        with patch.object(daemon, "_subscribe_to_events", side_effect=mock_subscribe):
            await daemon.start()
            await daemon.start()
            assert call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, daemon):
        """Stop clears running flag."""
        daemon._running = True
        await daemon.stop()
        assert daemon._running is False


# =============================================================================
# Test Event Subscription
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_to_events_success(self):
        """Subscribes to EVALUATION_COMPLETED on success."""
        daemon = AutoPromotionDaemon()

        mock_router = MagicMock()
        mock_router.subscribe = MagicMock()  # subscribe is synchronous, not async

        # Dec 29, 2025: Must also patch DataEventType to be non-None
        mock_event_type = MagicMock()
        mock_event_type.EVALUATION_COMPLETED = "evaluation_completed"

        with patch(
            "app.coordination.event_router.get_router", return_value=mock_router
        ), patch(
            "app.coordination.event_router.DataEventType", mock_event_type
        ):
            await daemon._subscribe_to_events()
            assert daemon._subscribed is True
            mock_router.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_handles_import_error(self):
        """Handles ImportError gracefully."""
        daemon = AutoPromotionDaemon()

        # Simulate the entire import failing
        with patch.dict("sys.modules", {"app.coordination.event_router": None}):
            await daemon._subscribe_to_events()
            # Won't subscribe because import fails
            assert daemon._subscribed is False


# =============================================================================
# Test Evaluation Processing
# =============================================================================


class TestEvaluationProcessing:
    """Tests for processing evaluation events."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with mocked check_promotion."""
        d = AutoPromotionDaemon()
        d._check_promotion = AsyncMock()
        return d

    @pytest.mark.asyncio
    async def test_process_evaluation_creates_candidate(self, daemon):
        """Processing creates candidate if not exists."""
        payload = {
            "config_key": "hex8_2p",
            "model_path": "/path/to/model.pth",
            "opponent_type": "RANDOM",
            "win_rate": 0.90,
            "games_played": 25,
        }
        await daemon._process_evaluation(payload)

        assert "hex8_2p" in daemon._candidates
        candidate = daemon._candidates["hex8_2p"]
        assert candidate.evaluation_results["RANDOM"] == 0.90
        assert candidate.evaluation_games["RANDOM"] == 25

    @pytest.mark.asyncio
    async def test_process_evaluation_updates_existing_candidate(self, daemon):
        """Processing updates existing candidate."""
        # First evaluation
        await daemon._process_evaluation({
            "config_key": "hex8_2p",
            "model_path": "/path/model.pth",
            "opponent_type": "RANDOM",
            "win_rate": 0.90,
            "games_played": 25,
        })

        # Second evaluation (different opponent)
        await daemon._process_evaluation({
            "config_key": "hex8_2p",
            "model_path": "/path/model.pth",
            "opponent_type": "HEURISTIC",
            "win_rate": 0.70,
            "games_played": 30,
        })

        candidate = daemon._candidates["hex8_2p"]
        assert candidate.evaluation_results["RANDOM"] == 0.90
        assert candidate.evaluation_results["HEURISTIC"] == 0.70

    @pytest.mark.asyncio
    async def test_process_evaluation_ignores_missing_fields(self, daemon):
        """Ignores events with missing required fields."""
        await daemon._process_evaluation({})
        assert daemon._candidates == {}

        await daemon._process_evaluation({"config_key": "hex8_2p"})  # No model_path
        assert daemon._candidates == {}


# =============================================================================
# Test Promotion Logic
# =============================================================================


class TestPromotionLogic:
    """Tests for promotion decision logic."""

    @pytest.fixture
    def daemon(self):
        """Create daemon with minimal thresholds."""
        config = AutoPromotionConfig(
            min_games_vs_random=5,
            min_games_vs_heuristic=5,
            consecutive_passes_required=1,
            min_elo_improvement=0.0,  # Dec 28: Disable Elo check for threshold tests
        )
        return AutoPromotionDaemon(config=config)

    @pytest.mark.asyncio
    async def test_requires_both_baselines_by_default(self, daemon):
        """Waits for both RANDOM and HEURISTIC results."""
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results["RANDOM"] = 0.95
        candidate.evaluation_games["RANDOM"] = 10

        daemon._promote_model = AsyncMock()

        with patch(
            "app.config.thresholds.get_promotion_thresholds",
            return_value={"vs_random": 0.85, "vs_heuristic": 0.60},
        ):
            await daemon._check_promotion(candidate)
            daemon._promote_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_promotes_when_thresholds_met(self, daemon):
        """Promotes when all thresholds are met."""
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results = {"RANDOM": 0.95, "HEURISTIC": 0.70}
        candidate.evaluation_games = {"RANDOM": 10, "HEURISTIC": 10}

        daemon._promote_model = AsyncMock()

        # Dec 28, 2025: Mock should_promote_model (two-tier promotion system)
        with patch(
            "app.config.thresholds.should_promote_model",
            return_value=(True, "Meets aspirational thresholds"),
        ):
            await daemon._check_promotion(candidate)
            daemon._promote_model.assert_called_once_with(candidate)

    @pytest.mark.asyncio
    async def test_rejects_when_random_threshold_not_met(self, daemon):
        """Rejects when RANDOM threshold not met."""
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results = {"RANDOM": 0.80, "HEURISTIC": 0.70}  # 0.80 < 0.85
        candidate.evaluation_games = {"RANDOM": 10, "HEURISTIC": 10}

        daemon._promote_model = AsyncMock()

        # Dec 28, 2025: Mock should_promote_model (two-tier promotion system)
        with patch(
            "app.config.thresholds.should_promote_model",
            return_value=(False, "Below minimum thresholds"),
        ):
            await daemon._check_promotion(candidate)
            daemon._promote_model.assert_not_called()
            assert candidate.consecutive_passes == 0  # Reset on failure

    @pytest.mark.asyncio
    async def test_rejects_insufficient_games(self, daemon):
        """Rejects when not enough games played."""
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results = {"RANDOM": 0.95, "HEURISTIC": 0.70}
        candidate.evaluation_games = {"RANDOM": 3, "HEURISTIC": 3}  # Below min of 5

        daemon._promote_model = AsyncMock()

        with patch(
            "app.config.thresholds.get_promotion_thresholds",
            return_value={"vs_random": 0.85, "vs_heuristic": 0.60},
        ):
            await daemon._check_promotion(candidate)
            daemon._promote_model.assert_not_called()


# =============================================================================
# Test Cooldown
# =============================================================================


class TestCooldown:
    """Tests for promotion cooldown."""

    @pytest.mark.asyncio
    async def test_respects_cooldown_period(self):
        """Doesn't promote during cooldown period."""
        config = AutoPromotionConfig(
            promotion_cooldown_seconds=300.0,
            min_games_vs_random=1,
            min_games_vs_heuristic=1,
            consecutive_passes_required=1,
        )
        daemon = AutoPromotionDaemon(config=config)

        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results = {"RANDOM": 0.95, "HEURISTIC": 0.70}
        candidate.evaluation_games = {"RANDOM": 10, "HEURISTIC": 10}
        candidate.last_promotion_time = time.time() - 60  # 60s ago (still in 300s cooldown)

        daemon._promote_model = AsyncMock()

        # Dec 28, 2025: Mock should_promote_model (two-tier promotion system)
        with patch(
            "app.config.thresholds.should_promote_model",
            return_value=(True, "Meets aspirational thresholds"),
        ):
            await daemon._check_promotion(candidate)
            daemon._promote_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_promotes_after_cooldown(self):
        """Promotes after cooldown expires."""
        config = AutoPromotionConfig(
            promotion_cooldown_seconds=60.0,
            min_games_vs_random=1,
            min_games_vs_heuristic=1,
            consecutive_passes_required=1,
            min_elo_improvement=0.0,  # Dec 28: Disable Elo check for cooldown tests
        )
        daemon = AutoPromotionDaemon(config=config)

        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results = {"RANDOM": 0.95, "HEURISTIC": 0.70}
        candidate.evaluation_games = {"RANDOM": 10, "HEURISTIC": 10}
        candidate.last_promotion_time = time.time() - 120  # 120s ago (past 60s cooldown)

        daemon._promote_model = AsyncMock()

        # Dec 28, 2025: Mock should_promote_model (two-tier promotion system)
        with patch(
            "app.config.thresholds.should_promote_model",
            return_value=(True, "Meets aspirational thresholds"),
        ):
            await daemon._check_promotion(candidate)
            daemon._promote_model.assert_called_once()


# =============================================================================
# Test Dry Run Mode
# =============================================================================


class TestDryRunMode:
    """Tests for dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_promote(self):
        """Dry run logs but doesn't actually promote."""
        config = AutoPromotionConfig(dry_run=True)
        daemon = AutoPromotionDaemon(config=config)

        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )

        # Mock the promotion controller
        with patch(
            "app.training.promotion_controller.PromotionController"
        ) as mock_controller:
            await daemon._promote_model(candidate)
            mock_controller.assert_not_called()


# =============================================================================
# Test Event Emission
# =============================================================================


class TestEventEmission:
    """Tests for event emission."""

    @pytest.mark.asyncio
    async def test_emits_model_promoted_event(self):
        """Emits MODEL_PROMOTED event on success."""
        daemon = AutoPromotionDaemon()
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.evaluation_results = {"RANDOM": 0.95, "HEURISTIC": 0.70}

        # Dec 29, 2025: Must also patch DataEventType to be non-None
        mock_event_type = MagicMock()
        mock_event_type.MODEL_PROMOTED = "model_promoted"

        with patch(
            "app.coordination.event_router.publish",
            new_callable=AsyncMock,
        ) as mock_publish, patch(
            "app.coordination.event_router.DataEventType",
            mock_event_type,
        ), patch(
            "app.coordination.event_router.emit_curriculum_advanced",
            new_callable=AsyncMock,
        ):
            await daemon._emit_promotion_event(candidate)
            mock_publish.assert_awaited_once()
            call_kwargs = mock_publish.await_args.kwargs
            assert call_kwargs["event_type"] == "model_promoted"
            assert call_kwargs["payload"]["config_key"] == "hex8_2p"
            assert call_kwargs["source"] == "auto_promotion_daemon"

    @pytest.mark.asyncio
    async def test_emits_promotion_failed_event(self):
        """Emits PROMOTION_FAILED event on failure."""
        daemon = AutoPromotionDaemon()
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )

        with patch(
            "app.coordination.event_router.publish",
            new_callable=AsyncMock,
        ) as mock_publish:
            await daemon._emit_promotion_failed(candidate, error="Test error")
            mock_publish.assert_awaited_once()
            call_kwargs = mock_publish.await_args.kwargs
            assert call_kwargs["payload"]["error"] == "Test error"
            assert call_kwargs["source"] == "auto_promotion_daemon"

    @pytest.mark.asyncio
    async def test_emits_promotion_completed_event(self):
        """Emits PROMOTION_COMPLETED event after promotion decision."""
        daemon = AutoPromotionDaemon()
        candidate = PromotionCandidate(
            config_key="hex8_2p",
            model_path="/model.pth",
        )
        candidate.previous_elo = 1200.0
        candidate.estimated_elo = 1235.0
        candidate.consecutive_failures = 1
        candidate.consecutive_passes = 2
        candidate.evaluation_results = {"RANDOM": 0.9, "HEURISTIC": 0.68}

        with patch(
            "app.coordination.event_router.publish",
            new_callable=AsyncMock,
        ) as mock_publish:
            await daemon._emit_promotion_completed(candidate, success=True)
            mock_publish.assert_awaited_once()
            call_kwargs = mock_publish.await_args.kwargs
            assert call_kwargs["event_type"] == "PROMOTION_COMPLETED"
            assert call_kwargs["payload"]["config_key"] == "hex8_2p"
            assert call_kwargs["payload"]["success"] is True
            assert call_kwargs["payload"]["elo_change"] == 35.0
            assert call_kwargs["source"] == "auto_promotion_daemon"


# =============================================================================
# Test Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_when_stopped(self):
        """Health check reports unhealthy when stopped."""
        daemon = AutoPromotionDaemon()
        result = daemon.health_check()
        assert result.healthy is False
        assert result.status.value == "stopped"

    def test_health_check_when_not_subscribed(self):
        """Health check reports degraded when not subscribed."""
        daemon = AutoPromotionDaemon()
        daemon._running = True
        result = daemon.health_check()
        assert result.healthy is False
        assert result.status.value == "degraded"

    def test_health_check_when_healthy(self):
        """Health check reports healthy when running and subscribed."""
        daemon = AutoPromotionDaemon()
        daemon._running = True
        daemon._subscribed = True
        result = daemon.health_check()
        assert result.healthy is True
        assert result.status.value == "running"


# =============================================================================
# Test Velocity Gate (P4 - Jan 26, 2026)
# =============================================================================


class TestVelocityGate:
    """Tests for Elo velocity gate functionality."""

    @pytest.mark.asyncio
    async def test_velocity_gate_passes_when_velocity_positive(self):
        """Velocity gate passes when Elo is increasing."""
        config = AutoPromotionConfig(velocity_gate_enabled=True)
        daemon = AutoPromotionDaemon(config=config)
        candidate = PromotionCandidate(config_key="hex8_2p", model_path="/model.pth")

        mock_trend = {"slope": 5.0, "is_declining": False}
        with patch(
            "app.coordination.auto_promotion_daemon.get_elo_trend_for_config",
            return_value=mock_trend,
        ), patch("app.coordination.auto_promotion_daemon.HAS_ELO_TREND", True):
            passed, reason = await daemon._check_velocity_gate(candidate)

        assert passed is True
        assert "velocity_ok" in reason

    @pytest.mark.asyncio
    async def test_velocity_gate_fails_when_declining(self):
        """Velocity gate fails when Elo is declining."""
        config = AutoPromotionConfig(velocity_gate_enabled=True)
        daemon = AutoPromotionDaemon(config=config)
        candidate = PromotionCandidate(config_key="hex8_2p", model_path="/model.pth")

        mock_trend = {"slope": -2.0, "is_declining": True}
        with patch(
            "app.coordination.auto_promotion_daemon.get_elo_trend_for_config",
            return_value=mock_trend,
        ), patch("app.coordination.auto_promotion_daemon.HAS_ELO_TREND", True):
            passed, reason = await daemon._check_velocity_gate(candidate)

        assert passed is False
        assert "declining" in reason

    @pytest.mark.asyncio
    async def test_velocity_gate_passes_when_unavailable(self):
        """Velocity gate passes gracefully when elo_service unavailable."""
        config = AutoPromotionConfig(velocity_gate_enabled=True)
        daemon = AutoPromotionDaemon(config=config)
        candidate = PromotionCandidate(config_key="hex8_2p", model_path="/model.pth")

        with patch("app.coordination.auto_promotion_daemon.HAS_ELO_TREND", False):
            passed, reason = await daemon._check_velocity_gate(candidate)

        assert passed is True
        assert "unavailable" in reason

    @pytest.mark.asyncio
    async def test_velocity_gate_passes_when_no_trend_data(self):
        """Velocity gate passes when no trend data available (bootstrap)."""
        config = AutoPromotionConfig(velocity_gate_enabled=True)
        daemon = AutoPromotionDaemon(config=config)
        candidate = PromotionCandidate(config_key="hex8_2p", model_path="/model.pth")

        with patch(
            "app.coordination.auto_promotion_daemon.get_elo_trend_for_config",
            return_value=None,
        ), patch("app.coordination.auto_promotion_daemon.HAS_ELO_TREND", True):
            passed, reason = await daemon._check_velocity_gate(candidate)

        assert passed is True
        assert "no_trend_data" in reason

    @pytest.mark.asyncio
    async def test_velocity_gate_handles_errors_gracefully(self):
        """Velocity gate passes on errors (don't block on check failures)."""
        config = AutoPromotionConfig(velocity_gate_enabled=True)
        daemon = AutoPromotionDaemon(config=config)
        candidate = PromotionCandidate(config_key="hex8_2p", model_path="/model.pth")

        with patch(
            "app.coordination.auto_promotion_daemon.get_elo_trend_for_config",
            side_effect=Exception("DB connection failed"),
        ), patch("app.coordination.auto_promotion_daemon.HAS_ELO_TREND", True):
            passed, reason = await daemon._check_velocity_gate(candidate)

        assert passed is True
        assert "error" in reason


# =============================================================================
# Test Status
# =============================================================================


class TestStatus:
    """Tests for status reporting."""

    def test_get_status_returns_dict(self):
        """Status returns expected dictionary structure."""
        daemon = AutoPromotionDaemon()
        daemon._running = True
        daemon._subscribed = True

        status = daemon.get_status()

        assert status["running"] is True
        assert status["subscribed"] is True
        assert status["enabled"] is True
        assert status["dry_run"] is False
        assert "candidates" in status
        assert "promotion_history_count" in status
        assert "recent_promotions" in status


# =============================================================================
# Test Singleton
# =============================================================================


class TestSingleton:
    """Tests for singleton behavior."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_auto_promotion_daemon()

    def test_get_returns_singleton(self):
        """get_auto_promotion_daemon returns same instance."""
        d1 = get_auto_promotion_daemon()
        d2 = get_auto_promotion_daemon()
        assert d1 is d2

    def test_reset_clears_singleton(self):
        """reset clears the singleton."""
        d1 = get_auto_promotion_daemon()
        reset_auto_promotion_daemon()
        d2 = get_auto_promotion_daemon()
        assert d1 is not d2
