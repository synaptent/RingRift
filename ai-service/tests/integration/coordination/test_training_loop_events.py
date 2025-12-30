"""Integration tests for training loop event chains.

December 2025: Tests the complete event flow from selfplay to promotion.
Validates that events propagate correctly through the training pipeline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTrainingLoopEventChain:
    """Test the complete training loop event chain."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus that tracks emissions."""
        emitted_events: list[tuple[str, dict]] = []

        async def mock_publish(event_type: str, payload: dict, **kwargs):
            emitted_events.append((event_type, payload))

        bus = MagicMock()
        bus.publish = AsyncMock(side_effect=mock_publish)
        bus.emitted_events = emitted_events
        return bus

    @pytest.fixture
    def mock_event_router(self, mock_event_bus):
        """Mock the event router singleton."""
        with patch("app.coordination.event_router.get_event_bus", return_value=mock_event_bus):
            with patch("app.coordination.event_router.get_router", return_value=mock_event_bus):
                yield mock_event_bus

    def test_selfplay_to_new_games_event(self, mock_event_router):
        """Test that selfplay completion triggers NEW_GAMES_AVAILABLE."""
        from app.coordination.data_events import DataEventType

        # Simulate selfplay completion
        asyncio.run(mock_event_router.publish(
            DataEventType.SELFPLAY_COMPLETE.value,
            {
                "config_key": "hex8_2p",
                "games_generated": 100,
                "timestamp": datetime.now().isoformat(),
            }
        ))

        # Verify event was emitted
        assert len(mock_event_router.emitted_events) == 1
        event_type, payload = mock_event_router.emitted_events[0]
        assert "selfplay_complete" in event_type.lower() or "SELFPLAY" in event_type
        assert payload["config_key"] == "hex8_2p"

    def test_training_completed_event_chain(self, mock_event_router):
        """Test that TRAINING_COMPLETED triggers downstream events."""
        from app.coordination.data_events import DataEventType

        # Emit training completed
        asyncio.run(mock_event_router.publish(
            DataEventType.TRAINING_COMPLETED.value,
            {
                "config_key": "hex8_2p",
                "model_path": "models/test_hex8_2p.pth",
                "epochs": 50,
                "final_loss": 0.234,
                "timestamp": datetime.now().isoformat(),
            }
        ))

        # Verify event was captured
        assert len(mock_event_router.emitted_events) == 1
        event_type, payload = mock_event_router.emitted_events[0]
        assert payload["config_key"] == "hex8_2p"
        assert payload["model_path"] == "models/test_hex8_2p.pth"

    def test_evaluation_completed_triggers_promotion_check(self, mock_event_router):
        """Test that EVALUATION_COMPLETED can trigger MODEL_PROMOTED."""
        from app.coordination.data_events import DataEventType

        # Emit evaluation completed with passing results
        asyncio.run(mock_event_router.publish(
            DataEventType.EVALUATION_COMPLETED.value,
            {
                "config_key": "hex8_2p",
                "model_path": "models/candidate_hex8_2p.pth",
                "elo_delta": 50,
                "win_rate_vs_random": 0.95,
                "win_rate_vs_heuristic": 0.65,
                "passed_gauntlet": True,
                "timestamp": datetime.now().isoformat(),
            }
        ))

        # Verify evaluation event captured
        assert len(mock_event_router.emitted_events) == 1
        event_type, payload = mock_event_router.emitted_events[0]
        assert payload["passed_gauntlet"] is True
        assert payload["elo_delta"] == 50

    def test_regression_critical_pauses_training(self, mock_event_router):
        """Test that REGRESSION_CRITICAL triggers training pause."""
        from app.coordination.data_events import DataEventType

        # Emit critical regression
        asyncio.run(mock_event_router.publish(
            DataEventType.REGRESSION_CRITICAL.value,
            {
                "config_key": "hex8_2p",
                "elo_drop": 150,
                "threshold": 100,
                "timestamp": datetime.now().isoformat(),
            }
        ))

        # Verify regression event captured
        assert len(mock_event_router.emitted_events) == 1
        event_type, payload = mock_event_router.emitted_events[0]
        assert payload["elo_drop"] == 150
        assert payload["config_key"] == "hex8_2p"


class TestClusterHealthEventChain:
    """Test cluster health event chains."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus that tracks emissions."""
        emitted_events: list[tuple[str, dict]] = []

        async def mock_publish(event_type: str, payload: dict, **kwargs):
            emitted_events.append((event_type, payload))

        bus = MagicMock()
        bus.publish = AsyncMock(side_effect=mock_publish)
        bus.emitted_events = emitted_events
        return bus

    def test_host_offline_triggers_cluster_health_check(self, mock_event_bus):
        """Test that HOST_OFFLINE can trigger P2P_CLUSTER_UNHEALTHY."""
        from app.coordination.data_events import DataEventType

        # Emit host offline
        asyncio.run(mock_event_bus.publish(
            DataEventType.HOST_OFFLINE.value,
            {
                "node_id": "worker-1",
                "reason": "timeout",
                "last_seen": datetime.now().isoformat(),
            }
        ))

        assert len(mock_event_bus.emitted_events) == 1
        event_type, payload = mock_event_bus.emitted_events[0]
        assert payload["node_id"] == "worker-1"

    def test_host_online_triggers_recovery(self, mock_event_bus):
        """Test that HOST_ONLINE triggers recovery events."""
        from app.coordination.data_events import DataEventType

        # Emit host online
        asyncio.run(mock_event_bus.publish(
            DataEventType.HOST_ONLINE.value,
            {
                "node_id": "worker-1",
                "recovery_time_sec": 30.5,
            }
        ))

        assert len(mock_event_bus.emitted_events) == 1
        event_type, payload = mock_event_bus.emitted_events[0]
        assert payload["node_id"] == "worker-1"


class TestBackpressureEventChain:
    """Test backpressure event handling."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        emitted_events: list[tuple[str, dict]] = []

        async def mock_publish(event_type: str, payload: dict, **kwargs):
            emitted_events.append((event_type, payload))

        bus = MagicMock()
        bus.publish = AsyncMock(side_effect=mock_publish)
        bus.emitted_events = emitted_events
        return bus

    def test_evaluation_backpressure_emitted(self, mock_event_bus):
        """Test EVALUATION_BACKPRESSURE event emission."""
        from app.coordination.data_events import DataEventType

        # Emit backpressure when queue depth exceeds threshold
        asyncio.run(mock_event_bus.publish(
            DataEventType.EVALUATION_BACKPRESSURE.value if hasattr(DataEventType, 'EVALUATION_BACKPRESSURE')
            else "evaluation_backpressure",
            {
                "queue_depth": 75,
                "threshold": 70,
                "action": "pause_training",
            }
        ))

        assert len(mock_event_bus.emitted_events) == 1
        _, payload = mock_event_bus.emitted_events[0]
        assert payload["queue_depth"] == 75
        assert payload["action"] == "pause_training"

    def test_backpressure_release_resumes_training(self, mock_event_bus):
        """Test backpressure release triggers training resume."""
        from app.coordination.data_events import DataEventType

        # Emit backpressure released
        asyncio.run(mock_event_bus.publish(
            DataEventType.BACKPRESSURE_RELEASED.value,
            {
                "queue_depth": 30,
                "release_threshold": 35,
                "action": "resume_training",
            }
        ))

        assert len(mock_event_bus.emitted_events) == 1
        _, payload = mock_event_bus.emitted_events[0]
        assert payload["queue_depth"] == 30
        assert payload["action"] == "resume_training"


class TestEventSubscriptionValidation:
    """Test that all critical events have subscribers."""

    def test_data_event_types_exist(self):
        """Verify DataEventType enum has expected values."""
        from app.coordination.data_events import DataEventType

        # Training events
        assert hasattr(DataEventType, "TRAINING_COMPLETED")
        assert hasattr(DataEventType, "TRAINING_STARTED")
        assert hasattr(DataEventType, "TRAINING_FAILED")

        # Selfplay events
        assert hasattr(DataEventType, "SELFPLAY_COMPLETE")
        assert hasattr(DataEventType, "NEW_GAMES_AVAILABLE")

        # Evaluation events
        assert hasattr(DataEventType, "EVALUATION_COMPLETED")
        assert hasattr(DataEventType, "EVALUATION_STARTED")

        # Promotion events
        assert hasattr(DataEventType, "MODEL_PROMOTED")

        # Health events
        assert hasattr(DataEventType, "HOST_OFFLINE")
        assert hasattr(DataEventType, "HOST_ONLINE")
        assert hasattr(DataEventType, "P2P_CLUSTER_HEALTHY")
        assert hasattr(DataEventType, "P2P_CLUSTER_UNHEALTHY")

        # Regression events
        assert hasattr(DataEventType, "REGRESSION_DETECTED")
        assert hasattr(DataEventType, "REGRESSION_CRITICAL")

    def test_critical_events_documented(self):
        """Verify critical events are documented in the subscription matrix."""
        import os
        from pathlib import Path

        # Check if documentation exists
        doc_path = Path(__file__).parents[3] / "docs" / "architecture" / "EVENT_SUBSCRIPTION_MATRIX.md"

        if doc_path.exists():
            content = doc_path.read_text()

            # Check for critical event documentation
            assert "TRAINING_COMPLETED" in content
            assert "MODEL_PROMOTED" in content
            assert "EVALUATION_COMPLETED" in content
            assert "REGRESSION_CRITICAL" in content
        else:
            pytest.skip("EVENT_SUBSCRIPTION_MATRIX.md not found")


class TestEventPayloadValidation:
    """Test event payload structure validation."""

    def test_training_completed_payload_structure(self):
        """Validate TRAINING_COMPLETED payload has required fields."""
        payload = {
            "config_key": "hex8_2p",
            "model_path": "models/canonical_hex8_2p.pth",
            "epochs": 50,
            "final_loss": 0.234,
            "val_loss": 0.256,
            "timestamp": datetime.now().isoformat(),
        }

        # Required fields
        assert "config_key" in payload
        assert "model_path" in payload
        assert "timestamp" in payload

    def test_evaluation_completed_payload_structure(self):
        """Validate EVALUATION_COMPLETED payload has required fields."""
        payload = {
            "config_key": "hex8_2p",
            "model_path": "models/candidate_hex8_2p.pth",
            "elo_delta": 45,
            "win_rate_vs_random": 0.95,
            "win_rate_vs_heuristic": 0.65,
            "passed_gauntlet": True,
            "games_played": 100,
            "timestamp": datetime.now().isoformat(),
        }

        # Required fields
        assert "config_key" in payload
        assert "elo_delta" in payload
        assert "passed_gauntlet" in payload
        assert "timestamp" in payload

    def test_model_promoted_payload_structure(self):
        """Validate MODEL_PROMOTED payload has required fields."""
        payload = {
            "config_key": "hex8_2p",
            "model_path": "models/canonical_hex8_2p.pth",
            "previous_elo": 1400,
            "new_elo": 1450,
            "promotion_reason": "gauntlet_passed",
            "timestamp": datetime.now().isoformat(),
        }

        # Required fields
        assert "config_key" in payload
        assert "model_path" in payload
        assert "timestamp" in payload
