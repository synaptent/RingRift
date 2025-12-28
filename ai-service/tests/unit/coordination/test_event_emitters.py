"""Tests for event_emitters module (December 2025).

Comprehensive tests for the centralized event emission infrastructure.

Tests cover:
1. All emit_* functions (70+ typed emitters)
2. Event payload validation
3. Error handling when event bus is unavailable
4. Return values and side effects
5. Helper functions (_emit_data_event, is_critical_event, etc.)
6. Critical event retry logic
7. Stage event emission
8. Sync vs async emission patterns
"""

from __future__ import annotations

import asyncio
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


class MockDataEventType(Enum):
    """Mock DataEventType for testing without importing the real module."""

    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    NEW_GAMES_AVAILABLE = "new_games"
    DATA_SYNC_COMPLETED = "sync_completed"
    QUALITY_SCORE_UPDATED = "quality_score_updated"
    QUALITY_PENALTY_APPLIED = "quality_penalty_applied"
    TASK_SPAWNED = "task_spawned"
    TASK_CANCELLED = "task_cancelled"
    TASK_HEARTBEAT = "task_heartbeat"
    TASK_FAILED = "task_failed"
    TASK_ABANDONED = "task_abandoned"
    TASK_ORPHANED = "task_orphaned"
    CMAES_TRIGGERED = "cmaes_triggered"
    NAS_TRIGGERED = "nas_triggered"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"
    REGRESSION_DETECTED = "regression_detected"
    BACKPRESSURE_ACTIVATED = "backpressure_activated"
    BACKPRESSURE_RELEASED = "backpressure_released"
    CACHE_INVALIDATED = "cache_invalidated"
    HOST_ONLINE = "host_online"
    HOST_OFFLINE = "host_offline"
    NODE_RECOVERED = "node_recovered"
    NODE_ACTIVATED = "node_activated"
    TRAINING_ROLLBACK_NEEDED = "training_rollback_needed"
    TRAINING_ROLLBACK_COMPLETED = "training_rollback_completed"
    HANDLER_FAILED = "handler_failed"
    HANDLER_TIMEOUT = "handler_timeout"
    COORDINATOR_HEALTH_DEGRADED = "coordinator_health_degraded"
    COORDINATOR_SHUTDOWN = "coordinator_shutdown"
    COORDINATOR_HEARTBEAT = "coordinator_heartbeat"
    COORDINATOR_HEALTHY = "coordinator_healthy"
    COORDINATOR_UNHEALTHY = "coordinator_unhealthy"
    MODEL_CORRUPTED = "model_corrupted"
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    NODE_UNHEALTHY = "node_unhealthy"
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    SPLIT_BRAIN_DETECTED = "split_brain_detected"
    P2P_NODE_DEAD = "p2p_node_dead"
    REPAIR_COMPLETED = "repair_completed"
    REPAIR_FAILED = "repair_failed"


class MockStageEvent(Enum):
    """Mock StageEvent for testing."""

    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    SELFPLAY_COMPLETE = "selfplay_complete"
    GPU_SELFPLAY_COMPLETE = "gpu_selfplay_complete"
    CANONICAL_SELFPLAY_COMPLETE = "canonical_selfplay_complete"
    EVALUATION_COMPLETE = "evaluation_complete"
    PROMOTION_COMPLETE = "promotion_complete"
    SYNC_COMPLETE = "sync_complete"
    NPZ_EXPORT_COMPLETE = "npz_export_complete"


@dataclass
class MockDataEvent:
    """Mock DataEvent for testing."""

    event_type: MockDataEventType
    payload: dict[str, Any]
    source: str = "test"


@dataclass
class MockStageCompletionResult:
    """Mock StageCompletionResult for testing."""

    event: MockStageEvent
    success: bool
    iteration: int
    timestamp: str
    board_type: str = "square8"
    num_players: int = 2
    games_generated: int = 0
    model_path: str | None = None
    val_loss: float | None = None
    elo_delta: float | None = None
    win_rate: float | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


class MockDataEventBus:
    """Mock DataEventBus for testing."""

    def __init__(self):
        self.published_events: list[MockDataEvent] = []
        self.should_fail = False
        self.fail_count = 0
        self.max_fail_count = 0

    async def publish(self, event: MockDataEvent) -> None:
        if self.should_fail:
            if self.fail_count < self.max_fail_count:
                self.fail_count += 1
                raise Exception("Mock publish failure")
            else:
                self.should_fail = False
        self.published_events.append(event)

    def clear(self):
        self.published_events.clear()
        self.fail_count = 0


class MockStageEventBus:
    """Mock StageEventBus for testing."""

    def __init__(self):
        self.emitted_events: list[MockStageCompletionResult] = []
        self.should_fail = False
        self.fail_count = 0
        self.max_fail_count = 0

    async def emit(self, result: MockStageCompletionResult) -> None:
        if self.should_fail:
            if self.fail_count < self.max_fail_count:
                self.fail_count += 1
                raise Exception("Mock emit failure")
            else:
                self.should_fail = False
        self.emitted_events.append(result)

    def emit_sync(self, result: MockStageCompletionResult) -> None:
        if self.should_fail:
            raise Exception("Mock emit failure")
        self.emitted_events.append(result)

    def clear(self):
        self.emitted_events.clear()
        self.fail_count = 0


class MockEventRouter:
    """Mock unified event router for testing."""

    def __init__(self):
        self.published_events: list[tuple[str, dict, str]] = []
        self.should_fail = False

    async def publish(
        self, event_type: str, payload: dict[str, Any], source: str = "test"
    ) -> None:
        if self.should_fail:
            raise Exception("Mock router failure")
        self.published_events.append((event_type, payload, source))

    def publish_sync(
        self, event_type: str, payload: dict[str, Any], source: str = "test"
    ) -> None:
        if self.should_fail:
            raise Exception("Mock router failure")
        self.published_events.append((event_type, payload, source))

    def clear(self):
        self.published_events.clear()


@pytest.fixture
def mock_data_bus():
    """Provide a mock data event bus."""
    return MockDataEventBus()


@pytest.fixture
def mock_stage_bus():
    """Provide a mock stage event bus."""
    return MockStageEventBus()


@pytest.fixture
def mock_router():
    """Provide a mock unified event router."""
    return MockEventRouter()


@pytest.fixture
def mock_event_state():
    """Create a mock _EventState for testing."""

    class MockEventState:
        def __init__(self):
            self.stage_available = True
            self.data_available = True
            self.router_available = True
            self.event_type = MockDataEventType
            self.data_event = MockDataEvent
            self.stage_event = MockStageEvent
            self.stage_result = MockStageCompletionResult
            self.get_data_bus = None
            self.get_stage_bus = None
            self.get_router = None

    return MockEventState()


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetTimestamp:
    """Tests for _get_timestamp helper function."""

    def test_returns_iso_format(self):
        """Test timestamp is in ISO format."""
        # Import directly to test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _get_timestamp

        ts = _get_timestamp()
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_returns_current_time(self):
        """Test timestamp is approximately current time."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _get_timestamp

        before = datetime.now()
        ts = _get_timestamp()
        after = datetime.now()

        parsed = datetime.fromisoformat(ts)
        assert before <= parsed <= after


class TestIsCriticalEvent:
    """Tests for is_critical_event helper function."""

    def test_critical_events_detected(self):
        """Test known critical events are detected."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import (
                is_critical_event,
                CRITICAL_EVENT_TYPES,
            )

        for event_str in CRITICAL_EVENT_TYPES:
            assert is_critical_event(event_str) is True

    def test_non_critical_events(self):
        """Test non-critical events return False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import is_critical_event

        assert is_critical_event("random_event") is False
        assert is_critical_event("test_event") is False
        assert is_critical_event("") is False

    def test_accepts_enum_with_value(self):
        """Test accepts enum with .value attribute."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import is_critical_event

        mock_enum = MagicMock()
        mock_enum.value = "training_completed"
        assert is_critical_event(mock_enum) is True

        mock_enum.value = "random_event"
        assert is_critical_event(mock_enum) is False


class TestCriticalEventTypes:
    """Tests for CRITICAL_EVENT_TYPES constant."""

    def test_contains_training_events(self):
        """Test training events are included."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import CRITICAL_EVENT_TYPES

        assert "training_started" in CRITICAL_EVENT_TYPES
        assert "training_completed" in CRITICAL_EVENT_TYPES

    def test_contains_promotion_events(self):
        """Test promotion events are included."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import CRITICAL_EVENT_TYPES

        assert "model_promoted" in CRITICAL_EVENT_TYPES
        assert "promotion_failed" in CRITICAL_EVENT_TYPES

    def test_contains_sync_events(self):
        """Test sync events are included."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import CRITICAL_EVENT_TYPES

        assert "data_sync_completed" in CRITICAL_EVENT_TYPES


class TestCriticalStageEvents:
    """Tests for CRITICAL_STAGE_EVENTS constant."""

    def test_contains_training_events(self):
        """Test training stage events are included."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import CRITICAL_STAGE_EVENTS

        assert "training_complete" in CRITICAL_STAGE_EVENTS
        assert "training_failed" in CRITICAL_STAGE_EVENTS

    def test_contains_promotion_events(self):
        """Test promotion stage events are included."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import CRITICAL_STAGE_EVENTS

        assert "promotion_complete" in CRITICAL_STAGE_EVENTS
        assert "promotion_failed" in CRITICAL_STAGE_EVENTS

    def test_contains_evaluation_events(self):
        """Test evaluation stage events are included."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import CRITICAL_STAGE_EVENTS

        assert "evaluation_complete" in CRITICAL_STAGE_EVENTS


# =============================================================================
# _EventState Tests
# =============================================================================


class TestEventState:
    """Tests for _EventState class."""

    def test_initial_state(self):
        """Test _EventState initializes correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _EventState

        state = _EventState()
        assert state.stage_available is False
        assert state.data_available is False
        assert state.router_available is False
        assert state.event_type is None
        assert state.data_event is None
        assert state.stage_event is None
        assert state.stage_result is None
        assert state.get_data_bus is None
        assert state.get_stage_bus is None
        assert state.get_router is None

    def test_global_state_initialized(self):
        """Test global _state is properly initialized on import."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _state

        # Global state should be initialized with actual imports
        # These will be True if event_router imports succeed
        assert hasattr(_state, "stage_available")
        assert hasattr(_state, "data_available")
        assert hasattr(_state, "router_available")


# =============================================================================
# Data Event Emission Tests
# =============================================================================


class TestEmitDataEvent:
    """Tests for _emit_data_event helper function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", False
        ):
            result = await _emit_data_event(
                MockDataEventType.TRAINING_STARTED,
                {"job_id": "test"},
                source="test",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_bus_is_none(self):
        """Test returns False when get_data_bus returns None."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus", return_value=None
        ):
            result = await _emit_data_event(
                MockDataEventType.TRAINING_STARTED,
                {"job_id": "test"},
                source="test",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_adds_timestamp_if_missing(self, mock_data_bus):
        """Test auto-adds timestamp to payload if not present."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event, DataEvent

        mock_data_bus.published_events = []

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus",
            return_value=mock_data_bus,
        ), patch(
            "app.coordination.event_emitters.DataEvent", MockDataEvent
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ):
            payload = {"job_id": "test"}
            await _emit_data_event(
                MockDataEventType.TRAINING_STARTED,
                payload,
                source="test",
            )
            # Payload should have timestamp added
            assert "timestamp" in payload

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, mock_data_bus):
        """Test returns True on successful emission."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus",
            return_value=mock_data_bus,
        ), patch(
            "app.coordination.event_emitters.DataEvent", MockDataEvent
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ):
            result = await _emit_data_event(
                MockDataEventType.TRAINING_STARTED,
                {"job_id": "test"},
                source="test",
            )
            assert result is True
            assert len(mock_data_bus.published_events) == 1

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self, mock_data_bus):
        """Test returns False when publish raises exception."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event

        mock_data_bus.should_fail = True
        mock_data_bus.max_fail_count = 999  # Always fail

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus",
            return_value=mock_data_bus,
        ), patch(
            "app.coordination.event_emitters.DataEvent", MockDataEvent
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ):
            result = await _emit_data_event(
                MockDataEventType.TRAINING_STARTED,
                {"job_id": "test"},
                source="test",
            )
            assert result is False


class TestEmitDataEventWithRetry:
    """Tests for _emit_data_event_with_retry helper function."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self, mock_data_bus):
        """Test succeeds on first attempt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event_with_retry

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus",
            return_value=mock_data_bus,
        ), patch(
            "app.coordination.event_emitters.DataEvent", MockDataEvent
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ):
            result = await _emit_data_event_with_retry(
                MockDataEventType.TRAINING_COMPLETED,
                {"job_id": "test"},
                max_retries=3,
                base_delay=0.01,  # Short delay for testing
            )
            assert result is True
            assert len(mock_data_bus.published_events) == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, mock_data_bus):
        """Test retries on transient failures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event_with_retry

        # Fail first 2 attempts, succeed on 3rd
        mock_data_bus.should_fail = True
        mock_data_bus.max_fail_count = 2

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus",
            return_value=mock_data_bus,
        ), patch(
            "app.coordination.event_emitters.DataEvent", MockDataEvent
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ):
            result = await _emit_data_event_with_retry(
                MockDataEventType.TRAINING_COMPLETED,
                {"job_id": "test"},
                max_retries=3,
                base_delay=0.01,
            )
            assert result is True
            assert mock_data_bus.fail_count == 2

    @pytest.mark.asyncio
    async def test_returns_false_after_max_retries(self, mock_data_bus):
        """Test returns False when all retries exhausted."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event_with_retry

        # Always fail
        mock_data_bus.should_fail = True
        mock_data_bus.max_fail_count = 999

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.get_data_bus",
            return_value=mock_data_bus,
        ), patch(
            "app.coordination.event_emitters.DataEvent", MockDataEvent
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ):
            result = await _emit_data_event_with_retry(
                MockDataEventType.TRAINING_COMPLETED,
                {"job_id": "test"},
                max_retries=2,
                base_delay=0.01,
            )
            assert result is False
            assert mock_data_bus.fail_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_returns_false_when_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event_with_retry

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await _emit_data_event_with_retry(
                MockDataEventType.TRAINING_COMPLETED,
                {"job_id": "test"},
            )
            assert result is False


class TestEmitDataEventSync:
    """Tests for _emit_data_event_sync helper function."""

    def test_returns_false_when_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event_sync

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = _emit_data_event_sync(
                MockDataEventType.TRAINING_STARTED,
                {"job_id": "test"},
            )
            assert result is False

    def test_returns_false_when_no_event_loop(self):
        """Test returns False when no event loop is running."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import _emit_data_event_sync

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True):
            # No event loop running - should catch RuntimeError
            result = _emit_data_event_sync(
                MockDataEventType.TRAINING_STARTED,
                {"job_id": "test"},
            )
            assert result is False


# =============================================================================
# Training Event Emitter Tests
# =============================================================================


class TestEmitTrainingStarted:
    """Tests for emit_training_started function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_started

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_training_started(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_creates_stage_completion_result(self, mock_stage_bus):
        """Test creates proper StageCompletionResult."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_started

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            result = await emit_training_started(
                job_id="test-job",
                board_type="hex8",
                num_players=4,
                model_version="v2.1.0",
                node_name="gpu-node-1",
            )
            assert result is True
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            # Check the StageEvent argument
            assert call_args[0][0] == MockStageEvent.TRAINING_STARTED


class TestEmitTrainingComplete:
    """Tests for emit_training_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_training_complete(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_uses_training_complete_event_on_success(self):
        """Test uses TRAINING_COMPLETE event when success=True."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event_with_retry",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await emit_training_complete(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
                success=True,
                final_loss=0.05,
                final_elo=1650.0,
            )
            call_args = mock_emit.call_args
            assert call_args[0][0] == MockStageEvent.TRAINING_COMPLETE

    @pytest.mark.asyncio
    async def test_uses_training_failed_event_on_failure(self):
        """Test uses TRAINING_FAILED event when success=False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event_with_retry",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await emit_training_complete(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
                success=False,
            )
            call_args = mock_emit.call_args
            assert call_args[0][0] == MockStageEvent.TRAINING_FAILED

    @pytest.mark.asyncio
    async def test_calculates_elo_delta(self):
        """Test Elo delta is calculated from final_elo."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_complete

        captured_result = None

        async def capture_emit(event, result):
            nonlocal captured_result
            captured_result = result
            return True

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event_with_retry",
            side_effect=capture_emit,
        ):
            await emit_training_complete(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
                final_elo=1650.0,  # 150 above baseline
            )
            assert captured_result is not None
            # Elo delta should be 1650 - 1500 = 150
            assert captured_result.elo_delta == 150.0


class TestEmitTrainingStartedSync:
    """Tests for emit_training_started_sync function."""

    def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_started_sync

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = emit_training_started_sync(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
            )
            assert result is False


class TestEmitTrainingCompleteSync:
    """Tests for emit_training_complete_sync function."""

    def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_complete_sync

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = emit_training_complete_sync(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
            )
            assert result is False


# =============================================================================
# Selfplay Event Emitter Tests
# =============================================================================


class TestEmitSelfplayComplete:
    """Tests for emit_selfplay_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_selfplay_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_selfplay_complete(
                task_id="test-task",
                board_type="hex8",
                num_players=2,
                games_generated=100,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_maps_gpu_accelerated_type(self):
        """Test maps gpu_accelerated to GPU_SELFPLAY_COMPLETE event."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_selfplay_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await emit_selfplay_complete(
                task_id="test-task",
                board_type="hex8",
                num_players=2,
                games_generated=100,
                selfplay_type="gpu_accelerated",
            )
            call_args = mock_emit.call_args
            assert call_args[0][0] == MockStageEvent.GPU_SELFPLAY_COMPLETE

    @pytest.mark.asyncio
    async def test_maps_canonical_type(self):
        """Test maps canonical to CANONICAL_SELFPLAY_COMPLETE event."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_selfplay_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await emit_selfplay_complete(
                task_id="test-task",
                board_type="hex8",
                num_players=2,
                games_generated=100,
                selfplay_type="canonical",
            )
            call_args = mock_emit.call_args
            assert call_args[0][0] == MockStageEvent.CANONICAL_SELFPLAY_COMPLETE

    @pytest.mark.asyncio
    async def test_maps_standard_type(self):
        """Test maps standard to SELFPLAY_COMPLETE event."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_selfplay_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await emit_selfplay_complete(
                task_id="test-task",
                board_type="hex8",
                num_players=2,
                games_generated=100,
                selfplay_type="standard",
            )
            call_args = mock_emit.call_args
            assert call_args[0][0] == MockStageEvent.SELFPLAY_COMPLETE


# =============================================================================
# Evaluation Event Emitter Tests
# =============================================================================


class TestEmitEvaluationComplete:
    """Tests for emit_evaluation_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_evaluation_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_evaluation_complete(
                model_id="test-model",
                board_type="hex8",
                num_players=2,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_uses_retry_for_critical_event(self):
        """Test uses _emit_stage_event_with_retry for this critical event."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_evaluation_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ), patch(
            "app.coordination.event_emitters.StageCompletionResult",
            MockStageCompletionResult,
        ), patch(
            "app.coordination.event_emitters._emit_stage_event_with_retry",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_emit:
            await emit_evaluation_complete(
                model_id="test-model",
                board_type="hex8",
                num_players=2,
                win_rate=0.75,
                elo_delta=150.0,
            )
            # Should use retry version
            mock_emit.assert_called_once()


# =============================================================================
# Promotion Event Emitter Tests
# =============================================================================


class TestEmitPromotionComplete:
    """Tests for emit_promotion_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_promotion_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_promotion_complete(
                model_id="test-model",
                board_type="hex8",
                num_players=2,
            )
            assert result is False


class TestEmitPromotionCompleteSync:
    """Tests for emit_promotion_complete_sync function."""

    def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_promotion_complete_sync

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = emit_promotion_complete_sync(
                model_id="test-model",
                board_type="hex8",
                num_players=2,
            )
            assert result is False


# =============================================================================
# Export Event Emitter Tests
# =============================================================================


class TestEmitNpzExportComplete:
    """Tests for emit_npz_export_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_npz_export_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_npz_export_complete(
                board_type="hex8",
                num_players=2,
                samples_exported=10000,
                games_exported=500,
                output_path="/path/to/data.npz",
            )
            assert result is False


# =============================================================================
# Sync Event Emitter Tests
# =============================================================================


class TestEmitSyncComplete:
    """Tests for emit_sync_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_sync_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_sync_complete(
                sync_type="data",
                items_synced=100,
            )
            assert result is False


# =============================================================================
# Quality Event Emitter Tests
# =============================================================================


class TestEmitQualityUpdated:
    """Tests for emit_quality_updated function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_data_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_quality_updated

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_quality_updated(
                board_type="hex8",
                num_players=2,
                avg_quality=0.85,
                total_games=1000,
                high_quality_games=850,
            )
            assert result is False


class TestEmitGameQualityScore:
    """Tests for emit_game_quality_score function."""

    @pytest.mark.asyncio
    async def test_includes_is_per_game_flag(self):
        """Test payload includes is_per_game=True."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_game_quality_score

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ), patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_game_quality_score(
                game_id="game-123",
                quality_score=0.9,
                quality_category="high",
                training_weight=1.0,
            )
            assert captured_payload is not None
            assert captured_payload.get("is_per_game") is True


class TestEmitQualityPenaltyApplied:
    """Tests for emit_quality_penalty_applied function."""

    @pytest.mark.asyncio
    async def test_includes_required_fields(self):
        """Test payload includes all required fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_quality_penalty_applied

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_quality_penalty_applied(
                config_key="hex8_2p",
                penalty=0.15,
                reason="low_quality_score",
                current_weight=0.85,
            )
            assert captured_payload is not None
            assert captured_payload["config_key"] == "hex8_2p"
            assert captured_payload["penalty"] == 0.15
            assert captured_payload["reason"] == "low_quality_score"
            assert captured_payload["current_weight"] == 0.85


# =============================================================================
# Task Event Emitter Tests
# =============================================================================


class TestEmitTaskComplete:
    """Tests for emit_task_complete function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_stage_events_unavailable(self):
        """Test returns False when HAS_STAGE_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_complete

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_task_complete(
                task_id="test-task",
                task_type="selfplay",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_task_type(self):
        """Test returns False for unmapped task type."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_complete

        with patch(
            "app.coordination.event_emitters.HAS_STAGE_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.StageEvent", MockStageEvent
        ):
            result = await emit_task_complete(
                task_id="test-task",
                task_type="unknown_type",
            )
            assert result is False


class TestEmitTaskSpawned:
    """Tests for emit_task_spawned function."""

    @pytest.mark.asyncio
    async def test_includes_required_fields(self):
        """Test payload includes all required fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_spawned

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_task_spawned(
                task_id="task-123",
                task_type="selfplay",
                node_id="gpu-node-1",
                config={"board_type": "hex8"},
            )
            assert captured_payload is not None
            assert captured_payload["task_id"] == "task-123"
            assert captured_payload["task_type"] == "selfplay"
            assert captured_payload["node_id"] == "gpu-node-1"
            assert "started_at" in captured_payload


class TestEmitTaskCancelled:
    """Tests for emit_task_cancelled function."""

    @pytest.mark.asyncio
    async def test_includes_required_fields(self):
        """Test payload includes all required fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_cancelled

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_task_cancelled(
                task_id="task-123",
                task_type="selfplay",
                node_id="gpu-node-1",
                reason="user_requested",
                requested_by="admin",
            )
            assert captured_payload is not None
            assert captured_payload["task_id"] == "task-123"
            assert captured_payload["reason"] == "user_requested"
            assert captured_payload["requested_by"] == "admin"


class TestEmitTaskHeartbeat:
    """Tests for emit_task_heartbeat function."""

    @pytest.mark.asyncio
    async def test_includes_progress_fields(self):
        """Test payload includes progress fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_heartbeat

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_task_heartbeat(
                task_id="task-123",
                task_type="selfplay",
                node_id="gpu-node-1",
                progress_percent=50.0,
                games_completed=250,
                elapsed_seconds=120.0,
            )
            assert captured_payload is not None
            assert captured_payload["progress_percent"] == 50.0
            assert captured_payload["games_completed"] == 250


class TestEmitTaskFailed:
    """Tests for emit_task_failed function."""

    @pytest.mark.asyncio
    async def test_includes_error_fields(self):
        """Test payload includes error fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_failed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_task_failed(
                task_id="task-123",
                task_type="training",
                node_id="gpu-node-1",
                error="CUDA OOM",
                error_type="oom",
                retryable=True,
            )
            assert captured_payload is not None
            assert captured_payload["error"] == "CUDA OOM"
            assert captured_payload["error_type"] == "oom"
            assert captured_payload["retryable"] is True


class TestEmitTaskAbandoned:
    """Tests for emit_task_abandoned function."""

    @pytest.mark.asyncio
    async def test_includes_required_fields(self):
        """Test payload includes required fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_abandoned

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_task_abandoned(
                task_id="task-123",
                task_type="selfplay",
                node_id="gpu-node-1",
                reason="low_priority",
            )
            assert captured_payload is not None
            assert captured_payload["task_id"] == "task-123"
            assert captured_payload["reason"] == "low_priority"


class TestEmitTaskOrphaned:
    """Tests for emit_task_orphaned function."""

    @pytest.mark.asyncio
    async def test_includes_last_heartbeat(self):
        """Test payload includes last_heartbeat field."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_task_orphaned

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            last_hb = time.time() - 300  # 5 minutes ago
            await emit_task_orphaned(
                task_id="task-123",
                task_type="selfplay",
                node_id="gpu-node-1",
                last_heartbeat=last_hb,
                reason="worker_died",
            )
            assert captured_payload is not None
            assert captured_payload["last_heartbeat"] == last_hb


# =============================================================================
# Optimization Event Emitter Tests
# =============================================================================


class TestEmitOptimizationTriggered:
    """Tests for emit_optimization_triggered function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_data_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_optimization_triggered

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_optimization_triggered(
                optimization_type="cmaes",
                run_id="run-123",
                reason="plateau_detected",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_maps_cmaes_type(self):
        """Test maps 'cmaes' to CMAES_TRIGGERED event."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_optimization_triggered

        captured_event_type = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_event_type
            captured_event_type = event_type
            return True

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ), patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_optimization_triggered(
                optimization_type="cmaes",
                run_id="run-123",
                reason="plateau",
            )
            assert captured_event_type == MockDataEventType.CMAES_TRIGGERED

    @pytest.mark.asyncio
    async def test_maps_nas_type(self):
        """Test maps 'nas' to NAS_TRIGGERED event."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_optimization_triggered

        captured_event_type = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_event_type
            captured_event_type = event_type
            return True

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters.DataEventType", MockDataEventType
        ), patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_optimization_triggered(
                optimization_type="nas",
                run_id="run-123",
                reason="search",
            )
            assert captured_event_type == MockDataEventType.NAS_TRIGGERED


class TestEmitPlateauDetected:
    """Tests for emit_plateau_detected function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_data_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_plateau_detected

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_plateau_detected(
                metric_name="val_loss",
                current_value=0.5,
                best_value=0.4,
                epochs_since_improvement=10,
            )
            assert result is False


class TestEmitHyperparameterUpdated:
    """Tests for emit_hyperparameter_updated function."""

    @pytest.mark.asyncio
    async def test_includes_all_fields(self):
        """Test payload includes all required fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_hyperparameter_updated

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_hyperparameter_updated(
                config="hex8_2p",
                param_name="learning_rate",
                old_value=0.001,
                new_value=0.0005,
                optimizer="cmaes",
            )
            assert captured_payload is not None
            assert captured_payload["config"] == "hex8_2p"
            assert captured_payload["param_name"] == "learning_rate"
            assert captured_payload["old_value"] == 0.001
            assert captured_payload["new_value"] == 0.0005
            assert captured_payload["optimizer"] == "cmaes"


class TestEmitRegressionDetected:
    """Tests for emit_regression_detected function."""

    @pytest.mark.asyncio
    async def test_calculates_regression_amount(self):
        """Test regression_amount is calculated correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_regression_detected

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_regression_detected(
                metric_name="win_rate",
                current_value=0.6,
                previous_value=0.75,
                severity="moderate",
            )
            assert captured_payload is not None
            assert captured_payload["regression_amount"] == 0.15


# =============================================================================
# Backpressure Event Emitter Tests
# =============================================================================


class TestEmitBackpressureActivated:
    """Tests for emit_backpressure_activated function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_data_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_backpressure_activated

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_backpressure_activated(
                node_id="gpu-node-1",
                level="high",
                reason="gpu_memory",
            )
            assert result is False


class TestEmitBackpressureReleased:
    """Tests for emit_backpressure_released function."""

    @pytest.mark.asyncio
    async def test_includes_duration(self):
        """Test payload includes duration_seconds."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_backpressure_released

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_backpressure_released(
                node_id="gpu-node-1",
                previous_level="high",
                duration_seconds=120.0,
            )
            assert captured_payload is not None
            assert captured_payload["duration_seconds"] == 120.0


# =============================================================================
# Cache Event Emitter Tests
# =============================================================================


class TestEmitCacheInvalidated:
    """Tests for emit_cache_invalidated function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_data_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_cache_invalidated

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_cache_invalidated(
                invalidation_type="model",
                target_id="hex8_2p",
                count=5,
            )
            assert result is False


# =============================================================================
# Host/Node Event Emitter Tests
# =============================================================================


class TestEmitHostOnline:
    """Tests for emit_host_online function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_data_events_unavailable(self):
        """Test returns False when HAS_DATA_EVENTS is False."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_host_online

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_host_online(node_id="gpu-node-1")
            assert result is False

    @pytest.mark.asyncio
    async def test_includes_host_id_alias(self):
        """Test payload includes host_id as alias for node_id."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_host_online

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters.HAS_DATA_EVENTS", True
        ), patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_host_online(node_id="gpu-node-1")
            assert captured_payload is not None
            assert captured_payload["node_id"] == "gpu-node-1"
            assert captured_payload["host_id"] == "gpu-node-1"


class TestEmitHostOffline:
    """Tests for emit_host_offline function."""

    @pytest.mark.asyncio
    async def test_includes_reason(self):
        """Test payload includes reason."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_host_offline

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_host_offline(
                node_id="gpu-node-1",
                reason="graceful_shutdown",
            )
            assert captured_payload is not None
            assert captured_payload["reason"] == "graceful_shutdown"


class TestEmitNodeRecovered:
    """Tests for emit_node_recovered function."""

    @pytest.mark.asyncio
    async def test_includes_recovery_type(self):
        """Test payload includes recovery_type."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_node_recovered

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_node_recovered(
                node_id="gpu-node-1",
                recovery_type="automatic",
                offline_duration_seconds=300.0,
            )
            assert captured_payload is not None
            assert captured_payload["recovery_type"] == "automatic"
            assert captured_payload["offline_duration_seconds"] == 300.0


class TestEmitNodeActivated:
    """Tests for emit_node_activated function."""

    @pytest.mark.asyncio
    async def test_includes_activation_type(self):
        """Test payload includes activation_type."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_node_activated

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_node_activated(
                node_id="gpu-node-1",
                activation_type="selfplay",
                config_key="hex8_2p",
            )
            assert captured_payload is not None
            assert captured_payload["activation_type"] == "selfplay"
            assert captured_payload["config_key"] == "hex8_2p"


# =============================================================================
# Coordinator Event Emitter Tests
# =============================================================================


class TestEmitCoordinatorHealthy:
    """Tests for emit_coordinator_healthy function."""

    @pytest.mark.asyncio
    async def test_uses_atomic_state_guard(self):
        """Test uses _state.data_available guard."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_coordinator_healthy, _state

        # Save original state
        original_data_available = _state.data_available
        original_event_type = _state.event_type

        try:
            # Test when unavailable
            _state.data_available = False
            result = await emit_coordinator_healthy(coordinator_name="test")
            assert result is False

            # Test when event_type is None
            _state.data_available = True
            _state.event_type = None
            result = await emit_coordinator_healthy(coordinator_name="test")
            assert result is False
        finally:
            # Restore state
            _state.data_available = original_data_available
            _state.event_type = original_event_type


class TestEmitCoordinatorUnhealthy:
    """Tests for emit_coordinator_unhealthy function."""

    @pytest.mark.asyncio
    async def test_uses_atomic_state_guard(self):
        """Test uses _state.data_available guard."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_coordinator_unhealthy, _state

        original_data_available = _state.data_available
        original_event_type = _state.event_type

        try:
            _state.data_available = False
            result = await emit_coordinator_unhealthy(coordinator_name="test")
            assert result is False
        finally:
            _state.data_available = original_data_available
            _state.event_type = original_event_type


class TestEmitCoordinatorHeartbeat:
    """Tests for emit_coordinator_heartbeat function."""

    @pytest.mark.asyncio
    async def test_includes_health_metrics(self):
        """Test payload includes health metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_coordinator_heartbeat

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_coordinator_heartbeat(
                coordinator_name="test-coordinator",
                health_score=0.95,
                active_handlers=5,
                events_processed=1000,
            )
            assert captured_payload is not None
            assert captured_payload["health_score"] == 0.95
            assert captured_payload["active_handlers"] == 5
            assert captured_payload["events_processed"] == 1000


class TestEmitCoordinatorShutdown:
    """Tests for emit_coordinator_shutdown function."""

    @pytest.mark.asyncio
    async def test_includes_shutdown_info(self):
        """Test payload includes shutdown information."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_coordinator_shutdown

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_coordinator_shutdown(
                coordinator_name="test-coordinator",
                reason="graceful",
                remaining_tasks=3,
                state_snapshot={"key": "value"},
            )
            assert captured_payload is not None
            assert captured_payload["reason"] == "graceful"
            assert captured_payload["remaining_tasks"] == 3
            assert captured_payload["state_snapshot"]["key"] == "value"


# =============================================================================
# Handler Error Event Emitter Tests
# =============================================================================


class TestEmitHandlerFailed:
    """Tests for emit_handler_failed function."""

    @pytest.mark.asyncio
    async def test_includes_error_info(self):
        """Test payload includes error information."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_handler_failed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_handler_failed(
                handler_name="my_handler",
                event_type="TRAINING_STARTED",
                error="KeyError: 'model_id'",
                coordinator="DataPipelineOrchestrator",
            )
            assert captured_payload is not None
            assert captured_payload["handler_name"] == "my_handler"
            assert captured_payload["event_type"] == "TRAINING_STARTED"
            assert captured_payload["error"] == "KeyError: 'model_id'"


class TestEmitHandlerTimeout:
    """Tests for emit_handler_timeout function."""

    @pytest.mark.asyncio
    async def test_includes_timeout_seconds(self):
        """Test payload includes timeout_seconds."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_handler_timeout

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_handler_timeout(
                handler_name="slow_handler",
                event_type="DATA_SYNC_COMPLETED",
                timeout_seconds=30.0,
            )
            assert captured_payload is not None
            assert captured_payload["timeout_seconds"] == 30.0


# =============================================================================
# Cluster Health Event Emitter Tests
# =============================================================================


class TestEmitNodeUnhealthy:
    """Tests for emit_node_unhealthy function."""

    @pytest.mark.asyncio
    async def test_includes_health_metrics(self):
        """Test payload includes health metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_node_unhealthy

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_node_unhealthy(
                node_id="gpu-node-1",
                reason="high_gpu_temp",
                gpu_utilization=95.0,
                consecutive_failures=5,
            )
            assert captured_payload is not None
            assert captured_payload["gpu_utilization"] == 95.0
            assert captured_payload["consecutive_failures"] == 5


class TestEmitHealthCheckPassed:
    """Tests for emit_health_check_passed function."""

    @pytest.mark.asyncio
    async def test_includes_latency(self):
        """Test payload includes latency_ms."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_health_check_passed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_health_check_passed(
                node_id="gpu-node-1",
                check_type="gpu",
                latency_ms=15.5,
            )
            assert captured_payload is not None
            assert captured_payload["latency_ms"] == 15.5


class TestEmitHealthCheckFailed:
    """Tests for emit_health_check_failed function."""

    @pytest.mark.asyncio
    async def test_includes_error(self):
        """Test payload includes error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_health_check_failed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_health_check_failed(
                node_id="gpu-node-1",
                reason="connection_timeout",
                check_type="ssh",
                error="Connection timed out after 30s",
            )
            assert captured_payload is not None
            assert captured_payload["error"] == "Connection timed out after 30s"


class TestEmitP2pClusterHealthy:
    """Tests for emit_p2p_cluster_healthy function."""

    @pytest.mark.asyncio
    async def test_includes_node_counts(self):
        """Test payload includes node counts."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_p2p_cluster_healthy

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_p2p_cluster_healthy(
                healthy_nodes=30,
                node_count=35,
            )
            assert captured_payload is not None
            assert captured_payload["healthy"] is True
            assert captured_payload["healthy_nodes"] == 30
            assert captured_payload["node_count"] == 35


class TestEmitP2pClusterUnhealthy:
    """Tests for emit_p2p_cluster_unhealthy function."""

    @pytest.mark.asyncio
    async def test_includes_alerts(self):
        """Test payload includes alerts."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_p2p_cluster_unhealthy

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_p2p_cluster_unhealthy(
                healthy_nodes=10,
                node_count=35,
                alerts=["Quorum lost", "Leader offline"],
            )
            assert captured_payload is not None
            assert captured_payload["healthy"] is False
            assert len(captured_payload["alerts"]) == 2


class TestEmitSplitBrainDetected:
    """Tests for emit_split_brain_detected function."""

    @pytest.mark.asyncio
    async def test_includes_leader_info(self):
        """Test payload includes leader information."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_split_brain_detected

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_split_brain_detected(
                leaders_seen=["node-1", "node-2"],
                severity="critical",
                voter_count=5,
                resolution_action="step_down",
            )
            assert captured_payload is not None
            assert captured_payload["leaders_seen"] == ["node-1", "node-2"]
            assert captured_payload["leader_count"] == 2
            assert captured_payload["severity"] == "critical"


class TestEmitP2pNodeDead:
    """Tests for emit_p2p_node_dead function."""

    @pytest.mark.asyncio
    async def test_includes_offline_duration(self):
        """Test payload includes offline_duration_seconds."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_p2p_node_dead

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_p2p_node_dead(
                node_id="gpu-node-1",
                reason="timeout",
                offline_duration_seconds=600.0,
            )
            assert captured_payload is not None
            assert captured_payload["offline_duration_seconds"] == 600.0


# =============================================================================
# Repair Event Emitter Tests
# =============================================================================


class TestEmitRepairCompleted:
    """Tests for emit_repair_completed function."""

    @pytest.mark.asyncio
    async def test_includes_repair_details(self):
        """Test payload includes repair details."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_repair_completed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_repair_completed(
                game_id="game-123",
                source_nodes=["node-1", "node-2"],
                target_nodes=["node-3"],
                duration_seconds=5.5,
                new_replica_count=3,
            )
            assert captured_payload is not None
            assert captured_payload["game_id"] == "game-123"
            assert captured_payload["source_nodes"] == ["node-1", "node-2"]
            assert captured_payload["target_nodes"] == ["node-3"]
            assert captured_payload["new_replica_count"] == 3


class TestEmitRepairFailed:
    """Tests for emit_repair_failed function."""

    @pytest.mark.asyncio
    async def test_includes_error(self):
        """Test payload includes error."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_repair_failed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_repair_failed(
                game_id="game-123",
                source_nodes=["node-1"],
                target_nodes=["node-3"],
                error="Connection refused",
            )
            assert captured_payload is not None
            assert captured_payload["error"] == "Connection refused"


# =============================================================================
# Curriculum Event Emitter Tests
# =============================================================================


class TestEmitCurriculumUpdated:
    """Tests for emit_curriculum_updated function."""

    @pytest.mark.asyncio
    async def test_wraps_curriculum_rebalanced(self):
        """Test calls emit_curriculum_rebalanced internally."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_curriculum_updated

        with patch(
            "app.coordination.event_emitters.emit_curriculum_rebalanced",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_rebalanced:
            result = await emit_curriculum_updated(
                config_key="hex8_2p",
                new_weight=1.3,
                trigger="promotion",
            )
            assert result is True
            mock_rebalanced.assert_called_once()


class TestEmitCurriculumRebalanced:
    """Tests for emit_curriculum_rebalanced function."""

    @pytest.mark.asyncio
    async def test_includes_weight_changes(self):
        """Test payload includes weight changes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_curriculum_rebalanced

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_curriculum_rebalanced(
                config="hex8_2p",
                old_weights={"hex8_2p": 1.0, "square8_2p": 1.0},
                new_weights={"hex8_2p": 1.3, "square8_2p": 0.8},
                reason="elo_improvement",
                trigger="promotion",
            )
            assert captured_payload is not None
            assert captured_payload["old_weights"] == {"hex8_2p": 1.0, "square8_2p": 1.0}
            assert captured_payload["new_weights"] == {"hex8_2p": 1.3, "square8_2p": 0.8}


class TestEmitTrainingTriggered:
    """Tests for emit_training_triggered function."""

    @pytest.mark.asyncio
    async def test_includes_threshold_info(self):
        """Test payload includes threshold information."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_triggered

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_training_triggered(
                config="hex8_2p",
                job_id="job-123",
                trigger_reason="game_threshold",
                game_count=5000,
                threshold=4000,
                priority="high",
            )
            assert captured_payload is not None
            assert captured_payload["games"] == 5000
            assert captured_payload["threshold"] == 4000
            assert captured_payload["priority"] == "high"


# =============================================================================
# Model Event Emitter Tests
# =============================================================================


class TestEmitModelCorrupted:
    """Tests for emit_model_corrupted function."""

    @pytest.mark.asyncio
    async def test_includes_corruption_type(self):
        """Test payload includes corruption_type."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_model_corrupted

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_model_corrupted(
                model_id="hex8_2p_v2",
                model_path="/models/canonical_hex8_2p.pth",
                corruption_type="checksum_mismatch",
            )
            assert captured_payload is not None
            assert captured_payload["corruption_type"] == "checksum_mismatch"


class TestEmitTrainingRollbackNeeded:
    """Tests for emit_training_rollback_needed function."""

    @pytest.mark.asyncio
    async def test_includes_rollback_info(self):
        """Test payload includes rollback information."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_rollback_needed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_training_rollback_needed(
                model_id="hex8_2p_v2",
                reason="severe_regression",
                checkpoint_path="/checkpoints/epoch_20.pth",
                severity="severe",
            )
            assert captured_payload is not None
            assert captured_payload["reason"] == "severe_regression"
            assert captured_payload["severity"] == "severe"


class TestEmitTrainingRollbackCompleted:
    """Tests for emit_training_rollback_completed function."""

    @pytest.mark.asyncio
    async def test_includes_rollback_details(self):
        """Test payload includes rollback details."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_training_rollback_completed

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_training_rollback_completed(
                model_id="hex8_2p_v2",
                checkpoint_path="/checkpoints/epoch_20.pth",
                rollback_from="epoch_30",
                reason="regression_detected",
            )
            assert captured_payload is not None
            assert captured_payload["rollback_from"] == "epoch_30"
            assert captured_payload["reason"] == "regression_detected"


# =============================================================================
# New Games Event Emitter Tests
# =============================================================================


class TestEmitNewGames:
    """Tests for emit_new_games function."""

    @pytest.mark.asyncio
    async def test_includes_game_counts(self):
        """Test payload includes game counts."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import emit_new_games

        captured_payload = None

        async def capture_emit(event_type, payload, **kwargs):
            nonlocal captured_payload
            captured_payload = payload
            return True

        with patch(
            "app.coordination.event_emitters._emit_data_event",
            side_effect=capture_emit,
        ):
            await emit_new_games(
                host="gpu-node-1",
                new_games=500,
                total_games=10000,
            )
            assert captured_payload is not None
            assert captured_payload["new_games"] == 500
            assert captured_payload["total_games"] == 10000


# =============================================================================
# Module-Level Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_exist(self):
        """Test all items in __all__ are importable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination import event_emitters

        for name in event_emitters.__all__:
            assert hasattr(event_emitters, name), f"Missing export: {name}"

    def test_critical_emitters_exported(self):
        """Test critical emitters are exported."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import __all__

        critical = [
            "emit_training_started",
            "emit_training_complete",
            "emit_selfplay_complete",
            "emit_evaluation_complete",
            "emit_promotion_complete",
        ]
        for name in critical:
            assert name in __all__, f"Critical emitter not exported: {name}"

    def test_constants_exported(self):
        """Test constants are exported."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.coordination.event_emitters import __all__

        assert "CRITICAL_EVENT_TYPES" in __all__
        assert "CRITICAL_STAGE_EVENTS" in __all__
        assert "is_critical_event" in __all__


# =============================================================================
# Deprecation Warning Tests
# =============================================================================


class TestDeprecationWarning:
    """Tests for module deprecation warning."""

    def test_emits_deprecation_warning_on_import(self):
        """Test module emits deprecation warning on import."""
        import importlib
        import sys

        # Remove from cache if present
        module_name = "app.coordination.event_emitters"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import app.coordination.event_emitters  # noqa: F401

            # Check for deprecation warning
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()
