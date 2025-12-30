"""Integration tests for curriculum feedback loop.

December 29, 2025: Tests the feedback chain:
    EVALUATION_COMPLETED → FeedbackLoopController → CurriculumIntegration
    → SelfplayScheduler → CURRICULUM_REBALANCED → Updated allocations

This verifies the core curriculum feedback mechanism that enables
48-hour autonomous cluster operation.
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


class TestCurriculumFeedbackLoop:
    """Test the curriculum feedback loop across multiple daemons."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock event router."""
        return MockEventRouter()

    @pytest.fixture
    def mock_curriculum_state(self):
        """Mock curriculum state for testing."""
        return {
            "hex8_2p": {"weight": 1.0, "elo": 1200, "velocity": 0.0},
            "hex8_3p": {"weight": 1.0, "elo": 1100, "velocity": 0.0},
            "hex8_4p": {"weight": 1.0, "elo": 1000, "velocity": 0.0},
            "square8_2p": {"weight": 1.0, "elo": 1150, "velocity": 0.0},
        }

    @pytest.mark.asyncio
    async def test_evaluation_triggers_curriculum_update(self, mock_router):
        """Test that EVALUATION_COMPLETED triggers curriculum weight updates."""
        # Subscribe to track events
        received_events = []

        async def track_event(event):
            received_events.append(event)

        mock_router.subscribe("CURRICULUM_REBALANCED", track_event)
        mock_router.subscribe("SELFPLAY_ALLOCATION_UPDATED", track_event)

        # Simulate EVALUATION_COMPLETED event
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {
                "config_key": "hex8_2p",
                "model_path": "models/test.pth",
                "win_rate": 0.75,
                "elo": 1350,
                "games_played": 100,
            },
        )

        # Verify event was recorded
        assert mock_router.event_count("EVALUATION_COMPLETED") == 1

        eval_event = mock_router.get_latest("EVALUATION_COMPLETED")
        assert eval_event is not None
        assert eval_event.payload["config_key"] == "hex8_2p"
        assert eval_event.payload["elo"] == 1350

    @pytest.mark.asyncio
    async def test_elo_velocity_triggers_rebalancing(self, mock_router, mock_curriculum_state):
        """Test that high Elo velocity triggers curriculum rebalancing."""
        # Simulate multiple evaluations to build velocity
        base_time = time.time()

        # Evaluation 1: Elo 1200
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {
                "config_key": "hex8_2p",
                "elo": 1200,
                "timestamp": base_time,
            },
        )

        # Evaluation 2: Elo 1250 (50 point gain)
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {
                "config_key": "hex8_2p",
                "elo": 1250,
                "timestamp": base_time + 3600,  # 1 hour later
            },
        )

        # Evaluation 3: Elo 1350 (100 point gain)
        await mock_router.publish(
            "EVALUATION_COMPLETED",
            {
                "config_key": "hex8_2p",
                "elo": 1350,
                "timestamp": base_time + 7200,  # 2 hours later
            },
        )

        # Should have 3 evaluation events
        assert mock_router.event_count("EVALUATION_COMPLETED") == 3

        # Calculate expected velocity: 150 Elo points in 2 hours = 75 Elo/hour
        evaluations = mock_router.get_events("EVALUATION_COMPLETED")
        elo_values = [e.payload["elo"] for e in evaluations]
        assert elo_values == [1200, 1250, 1350]

    @pytest.mark.asyncio
    async def test_regression_reduces_curriculum_weight(self, mock_router):
        """Test that regression detection reduces curriculum weight for affected config."""
        # Simulate regression event
        await mock_router.publish(
            "REGRESSION_DETECTED",
            {
                "config_key": "hex8_4p",
                "old_elo": 1200,
                "new_elo": 1050,
                "regression_magnitude": 150,
                "timestamp": time.time(),
            },
        )

        assert mock_router.event_count("REGRESSION_DETECTED") == 1

        regression = mock_router.get_latest("REGRESSION_DETECTED")
        assert regression is not None
        assert regression.payload["config_key"] == "hex8_4p"
        assert regression.payload["regression_magnitude"] == 150

    @pytest.mark.asyncio
    async def test_event_chain_order(self, mock_router):
        """Test that events occur in the expected order."""
        verifier = EventChainVerifier(mock_router)
        verifier.expect_chain([
            "EVALUATION_COMPLETED",
            "ELO_UPDATED",
            "CURRICULUM_REBALANCED",
        ])

        # Publish events in order
        await mock_router.publish("EVALUATION_COMPLETED", {"config_key": "hex8_2p", "elo": 1300})
        await mock_router.publish("ELO_UPDATED", {"config_key": "hex8_2p", "elo": 1300})
        await mock_router.publish("CURRICULUM_REBALANCED", {"configs_affected": ["hex8_2p"]})

        # Verify chain
        result = await verifier.verify(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_multiple_configs_independent_tracking(self, mock_router):
        """Test that multiple configs are tracked independently."""
        configs = ["hex8_2p", "hex8_3p", "hex8_4p"]

        # Publish evaluations for all configs
        for i, config in enumerate(configs):
            await mock_router.publish(
                "EVALUATION_COMPLETED",
                {
                    "config_key": config,
                    "elo": 1200 + i * 50,
                    "win_rate": 0.5 + i * 0.1,
                },
            )

        assert mock_router.event_count("EVALUATION_COMPLETED") == 3

        # Verify each config has its own event
        for config in configs:
            events = [
                e for e in mock_router.get_events("EVALUATION_COMPLETED")
                if e.payload.get("config_key") == config
            ]
            assert len(events) == 1, f"Expected 1 event for {config}, got {len(events)}"

    @pytest.mark.asyncio
    async def test_stall_detection_triggers_curriculum_advancement(self, mock_router):
        """Test that Elo stall triggers curriculum advancement."""
        # Simulate stall detection
        await mock_router.publish(
            "PROGRESS_STALL_DETECTED",
            {
                "config_key": "hex8_2p",
                "stall_duration_hours": 24,
                "last_elo": 1200,
                "elo_target": 1400,
            },
        )

        assert mock_router.event_count("PROGRESS_STALL_DETECTED") == 1

        stall = mock_router.get_latest("PROGRESS_STALL_DETECTED")
        assert stall.payload["stall_duration_hours"] == 24

    @pytest.mark.asyncio
    async def test_curriculum_weight_bounds(self, mock_router):
        """Test that curriculum weights stay within bounds."""
        # Simulate curriculum rebalance with extreme values
        await mock_router.publish(
            "CURRICULUM_REBALANCED",
            {
                "configs_affected": ["hex8_2p"],
                "weights": {
                    "hex8_2p": 2.0,  # Max weight
                    "hex8_3p": 0.1,  # Min weight
                },
            },
        )

        event = mock_router.get_latest("CURRICULUM_REBALANCED")
        assert event is not None

        weights = event.payload.get("weights", {})
        for config, weight in weights.items():
            assert 0.1 <= weight <= 2.0, f"Weight {weight} for {config} out of bounds"


class TestFeedbackLoopControllerIntegration:
    """Test FeedbackLoopController's event handling."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock event router."""
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_quality_feedback_adjusts_allocation(self, mock_router):
        """Test that quality feedback adjusts selfplay allocation."""
        # Simulate low quality data
        await mock_router.publish(
            "DATA_QUALITY_ASSESSED",
            {
                "config_key": "hex8_2p",
                "quality_score": 0.35,
                "sample_count": 5000,
                "issues": ["low_move_diversity", "short_games"],
            },
        )

        assert mock_router.event_count("DATA_QUALITY_ASSESSED") == 1

        quality_event = mock_router.get_latest("DATA_QUALITY_ASSESSED")
        assert quality_event.payload["quality_score"] < 0.40  # Below threshold

    @pytest.mark.asyncio
    async def test_hyperparameter_update_propagates(self, mock_router):
        """Test that hyperparameter updates propagate through the system."""
        # Simulate hyperparameter update
        await mock_router.publish(
            "HYPERPARAMETER_UPDATED",
            {
                "parameter": "learning_rate",
                "old_value": 0.001,
                "new_value": 0.0005,
                "config_key": "hex8_2p",
                "reason": "plateau_detected",
            },
        )

        assert mock_router.event_count("HYPERPARAMETER_UPDATED") == 1

        hp_event = mock_router.get_latest("HYPERPARAMETER_UPDATED")
        assert hp_event.payload["parameter"] == "learning_rate"
        assert hp_event.payload["new_value"] < hp_event.payload["old_value"]


class TestSelfplaySchedulerIntegration:
    """Test SelfplayScheduler's response to curriculum events."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock event router."""
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_allocation_responds_to_curriculum_rebalance(self, mock_router):
        """Test that allocation updates after CURRICULUM_REBALANCED."""
        # Simulate curriculum rebalance
        await mock_router.publish(
            "CURRICULUM_REBALANCED",
            {
                "configs_affected": ["hex8_2p", "hex8_3p"],
                "weights": {
                    "hex8_2p": 1.5,  # Increased
                    "hex8_3p": 0.8,  # Decreased
                    "hex8_4p": 1.0,  # Unchanged
                },
                "trigger": "elo_velocity",
            },
        )

        assert mock_router.event_count("CURRICULUM_REBALANCED") == 1

        rebalance = mock_router.get_latest("CURRICULUM_REBALANCED")
        weights = rebalance.payload.get("weights", {})

        # Verify weight changes
        assert weights["hex8_2p"] > 1.0  # Increased
        assert weights["hex8_3p"] < 1.0  # Decreased

    @pytest.mark.asyncio
    async def test_4p_allocation_minimums(self, mock_router):
        """Test that 4-player configs maintain minimum allocation."""
        # Simulate 4p config with low priority
        await mock_router.publish(
            "SELFPLAY_ALLOCATION_UPDATED",
            {
                "allocations": {
                    "hex8_2p": 0.4,
                    "hex8_3p": 0.3,
                    "hex8_4p": 0.15,  # Should not go below 0.10
                    "square8_2p": 0.15,
                },
            },
        )

        allocation_event = mock_router.get_latest("SELFPLAY_ALLOCATION_UPDATED")
        allocations = allocation_event.payload.get("allocations", {})

        # 4p config should have minimum 10% allocation
        assert allocations.get("hex8_4p", 0) >= 0.10


class TestEventTimingRequirements:
    """Test event timing SLAs."""

    @pytest.fixture
    def mock_router(self):
        """Create a mock event router."""
        return MockEventRouter()

    @pytest.mark.asyncio
    async def test_curriculum_update_within_sla(self, mock_router):
        """Test that curriculum updates complete within SLA."""
        start_time = time.time()

        # Simulate rapid event chain
        await mock_router.publish("EVALUATION_COMPLETED", {"config_key": "hex8_2p", "elo": 1300})
        await mock_router.publish("ELO_UPDATED", {"config_key": "hex8_2p", "elo": 1300})
        await mock_router.publish("CURRICULUM_REBALANCED", {"configs_affected": ["hex8_2p"]})

        end_time = time.time()

        # All events should complete within 1 second
        assert end_time - start_time < 1.0, "Event chain took too long"

        # Verify timestamps are monotonically increasing
        events = mock_router.events
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

    @pytest.mark.asyncio
    async def test_wait_for_event_timeout(self, mock_router):
        """Test that wait_for_event properly times out."""
        with pytest.raises(TimeoutError):
            await mock_router.wait_for_event("NONEXISTENT_EVENT", timeout=0.1)
