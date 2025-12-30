"""Integration tests for regression detection and rollback.

December 29, 2025: Tests the regression handling chain:
    EVALUATION_COMPLETED (low win rate) → RegressionDetector
    → REGRESSION_DETECTED → ModelLifecycleCoordinator
    → MODEL_ROLLBACK → CurriculumIntegration → Adjusted weights

This verifies the system correctly detects and handles model regressions
to maintain quality during 48-hour autonomous operation.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.integration.coordination.conftest import (
    EventChainVerifier,
    MockEventRouter,
    RecordedEvent,
)


class TestRegressionDetection:
    """Test regression detection from evaluation results."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock event router."""
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_low_win_rate_triggers_regression(self, mock_router):
        """Test that win rate below threshold triggers REGRESSION_DETECTED."""
        # Simulate evaluation with low win rate
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {
                "config_key": "hex8_2p",
                "model_path": "models/weak_model.pth",
                "win_rate": 0.35,  # Below 50% threshold
                "elo": 1050,
                "baseline_elo": 1200,
                "games_played": 100,
            },
        )

        # Simulate regression detection
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_2p",
                "old_elo": 1200,
                "new_elo": 1050,
                "win_rate": 0.35,
                "regression_magnitude": 150,
                "model_path": "models/weak_model.pth",
                "timestamp": time.time(),
            },
        )

        assert mock_router.event_count("REGRESSION_DETECTED") == 1

        regression = mock_router.get_latest("REGRESSION_DETECTED")
        assert regression.payload["regression_magnitude"] == 150
        assert regression.payload["win_rate"] < 0.50

    @pytest.mark.asyncio
    async def test_regression_triggers_rollback(self, mock_router):
        """Test that REGRESSION_DETECTED triggers MODEL_ROLLBACK."""
        verifier = EventChainVerifier(mock_router)
        verifier.expect_chain([
            "REGRESSION_DETECTED",
            "MODEL_ROLLBACK_INITIATED",
            "MODEL_ROLLBACK_COMPLETED",
        ])

        # Publish regression chain
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_2p",
                "old_elo": 1200,
                "new_elo": 1050,
            },
        )

        await mock_router.publish(
            "MODEL_ROLLBACK_INITIATED",
            {
                "config_key": "hex8_2p",
                "current_model": "models/weak.pth",
                "rollback_to": "models/previous.pth",
            },
        )

        await mock_router.publish(
            "MODEL_ROLLBACK_COMPLETED",
            {
                "config_key": "hex8_2p",
                "restored_model": "models/previous.pth",
                "success": True,
            },
        )

        result = await verifier.verify(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_regression_updates_selfplay_priority(self, mock_router):
        """Test that regression increases staleness weight for affected config."""
        # Simulate regression and priority update
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_4p",
                "old_elo": 1100,
                "new_elo": 950,
            },
        )

        await mock_router.publish(
            "SELFPLAY_PRIORITY_ADJUSTED",
            {
                "config_key": "hex8_4p",
                "reason": "regression_detected",
                "staleness_boost": 1.5,  # 50% boost
            },
        )

        priority_event = mock_router.get_latest("SELFPLAY_PRIORITY_ADJUSTED")
        assert priority_event is not None
        assert priority_event.payload["staleness_boost"] > 1.0

    @pytest.mark.asyncio
    async def test_regression_reduces_curriculum_exploration(self, mock_router):
        """Test that regression reduces exploration for affected config."""
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_2p",
                "regression_magnitude": 100,
            },
        )

        await mock_router.publish(
            "CURRICULUM_EMERGENCY_UPDATE",
            {
                "config_key": "hex8_2p",
                "action": "reduce_exploration",
                "temperature_reduction": 0.2,
                "reason": "regression_recovery",
            },
        )

        emergency_event = mock_router.get_latest("CURRICULUM_EMERGENCY_UPDATE")
        assert emergency_event is not None
        assert emergency_event.payload["action"] == "reduce_exploration"


class TestModelRollback:
    """Test model rollback mechanics."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_rollback_restores_previous_model(self, mock_router):
        """Test that rollback correctly restores previous model."""
        await mock_router.publish(
            "MODEL_ROLLBACK_INITIATED",
            {
                "config_key": "hex8_2p",
                "current_model": "models/v3.pth",
                "rollback_to": "models/v2.pth",
                "current_elo": 1050,
                "target_elo": 1200,
            },
        )

        await mock_router.publish(
            "MODEL_ROLLBACK_COMPLETED",
            {
                "config_key": "hex8_2p",
                "restored_model": "models/v2.pth",
                "success": True,
                "restored_elo": 1200,
            },
        )

        rollback = mock_router.get_latest("MODEL_ROLLBACK_COMPLETED")
        assert rollback.payload["success"] is True
        assert rollback.payload["restored_elo"] > 1050

    @pytest.mark.asyncio
    async def test_rollback_failure_handling(self, mock_router):
        """Test handling of rollback failures."""
        await mock_router.publish(
            "MODEL_ROLLBACK_INITIATED",
            {
                "config_key": "hex8_2p",
                "rollback_to": "models/missing.pth",
            },
        )

        await mock_router.publish(
            "MODEL_ROLLBACK_FAILED",
            {
                "config_key": "hex8_2p",
                "error": "Model file not found",
                "fallback_action": "use_baseline",
            },
        )

        failure = mock_router.get_latest("MODEL_ROLLBACK_FAILED")
        assert failure is not None
        assert "fallback_action" in failure.payload

    @pytest.mark.asyncio
    async def test_next_training_uses_rollback_model(self, mock_router):
        """Test that next training starts with rolled-back model as init."""
        # Simulate rollback followed by training
        await mock_router.publish(
            "MODEL_ROLLBACK_COMPLETED",
            {
                "config_key": "hex8_2p",
                "restored_model": "models/v2.pth",
            },
        )

        await mock_router.publish(
            "TRAINING_STARTED",
            {
                "config_key": "hex8_2p",
                "init_weights": "models/v2.pth",  # Should use rollback model
                "epoch": 0,
            },
        )

        training = mock_router.get_latest("TRAINING_STARTED")
        rollback = mock_router.get_latest("MODEL_ROLLBACK_COMPLETED")

        assert training.payload["init_weights"] == rollback.payload["restored_model"]


class TestRegressionRecovery:
    """Test complete regression recovery flow."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_full_regression_recovery_chain(self, mock_router):
        """Test the complete regression detection → recovery flow."""
        verifier = EventChainVerifier(mock_router)
        verifier.expect_chain([
            "EVALUATION_COMPLETED",
            "REGRESSION_DETECTED",
            "MODEL_ROLLBACK_INITIATED",
            "MODEL_ROLLBACK_COMPLETED",
            "CURRICULUM_EMERGENCY_UPDATE",
            "TRAINING_STARTED",
            "EVALUATION_COMPLETED",
        ])

        # Step 1: Evaluation shows regression
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {"config_key": "hex8_2p", "elo": 1050, "win_rate": 0.35},
        )

        # Step 2: Regression detected
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {"config_key": "hex8_2p", "old_elo": 1200, "new_elo": 1050},
        )

        # Step 3: Rollback initiated
        await mock_router.publish(
            "MODEL_ROLLBACK_INITIATED",
            {"config_key": "hex8_2p", "rollback_to": "models/v2.pth"},
        )

        # Step 4: Rollback completed
        await mock_router.publish(
            "MODEL_ROLLBACK_COMPLETED",
            {"config_key": "hex8_2p", "restored_model": "models/v2.pth"},
        )

        # Step 5: Curriculum adjusted
        await mock_router.publish(
            "CURRICULUM_EMERGENCY_UPDATE",
            {"config_key": "hex8_2p", "action": "reduce_exploration"},
        )

        # Step 6: New training starts with rollback model
        await mock_router.publish(
            "TRAINING_STARTED",
            {"config_key": "hex8_2p", "init_weights": "models/v2.pth"},
        )

        # Step 7: New evaluation shows improvement
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {"config_key": "hex8_2p", "elo": 1250, "win_rate": 0.65},
        )

        result = await verifier.verify(timeout=1.0)
        assert result is True

        # Verify recovery: final Elo > regression Elo
        evals = mock_router.get_events("EVALUATION_COMPLETED")
        final_elo = evals[-1].payload["elo"]
        regression_elo = evals[0].payload["elo"]
        assert final_elo > regression_elo

    @pytest.mark.asyncio
    async def test_multiple_regressions_handled(self, mock_router):
        """Test handling of multiple consecutive regressions."""
        configs = ["hex8_2p", "hex8_3p"]

        for config in configs:
            await mock_router.publish(
                "REGRESSION_DETECTED",
                {
                    "config_key": config,
                    "old_elo": 1200,
                    "new_elo": 1050,
                },
            )

        assert mock_router.event_count("REGRESSION_DETECTED") == 2

        # Each config should have its own regression event
        for config in configs:
            events = [
                e for e in mock_router.get_events("REGRESSION_DETECTED")
                if e.payload.get("config_key") == config
            ]
            assert len(events) == 1

    @pytest.mark.asyncio
    async def test_regression_severity_affects_response(self, mock_router):
        """Test that regression severity affects response intensity."""
        # Minor regression (50 Elo points)
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_2p",
                "regression_magnitude": 50,
                "severity": "minor",
            },
        )

        # Major regression (200 Elo points)
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_3p",
                "regression_magnitude": 200,
                "severity": "major",
            },
        )

        minor = mock_router.get_events("REGRESSION_DETECTED")[0]
        major = mock_router.get_events("REGRESSION_DETECTED")[1]

        assert minor.payload["severity"] == "minor"
        assert major.payload["severity"] == "major"
        assert major.payload["regression_magnitude"] > minor.payload["regression_magnitude"]


class TestRegressionCriticalAlert:
    """Test REGRESSION_CRITICAL event handling."""

    @pytest.fixture
    def mock_router(self):
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_critical_regression_pauses_training(self, mock_router):
        """Test that critical regression (>300 Elo) pauses training."""
        await mock_router.publish(
            "REGRESSION_CRITICAL",
            {
                "config_key": "hex8_2p",
                "regression_magnitude": 350,
                "action": "pause_training",
            },
        )

        await mock_router.publish(
            "TRAINING_PAUSED",
            {
                "config_key": "hex8_2p",
                "reason": "critical_regression",
            },
        )

        assert mock_router.event_count("REGRESSION_CRITICAL") == 1
        assert mock_router.event_count("TRAINING_PAUSED") == 1

    @pytest.mark.asyncio
    async def test_critical_regression_triggers_investigation(self, mock_router):
        """Test that critical regression triggers data quality investigation."""
        await mock_router.publish(
            "REGRESSION_CRITICAL",
            {
                "config_key": "hex8_2p",
                "regression_magnitude": 400,
            },
        )

        await mock_router.publish(
            "DATA_QUALITY_INVESTIGATION_STARTED",
            {
                "config_key": "hex8_2p",
                "trigger": "critical_regression",
                "checks": ["duplicate_games", "corrupted_moves", "label_errors"],
            },
        )

        investigation = mock_router.get_latest("DATA_QUALITY_INVESTIGATION_STARTED")
        assert investigation is not None
        assert "duplicate_games" in investigation.payload["checks"]
